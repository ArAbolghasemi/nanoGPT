import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from tqdm import trange
from inference_util import print_model_layers, load_model_from_checkpoint
from optimizers import CholeskyCMAES_torch, CholeskyCMAES_numpy

def unit_maximization(model, layers_channels_dict, 
                      lr=1e-2, weight_decay=0e-4, max_iter=5000, 
                      block_size=1024, device='cuda', print_progress=True):
    """
    Maximize the response of specified units in a GPT model while keeping track of optimization history.
    
    Args:
        model (torch.nn.Module): Loaded GPT model.
        layers_channels_dict (dict): A dictionary where keys are layer indices and values are lists of channels to maximize.
            Example: {3: [10, 20], 5: [50]} means maximize channels 10 and 20 in layer 3 and channel 50 in layer 5.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        max_iter (int): Maximum number of optimization iterations.
        block_size (int): Length of the input sequence to optimize.
        device (str): Device for computation ('cuda' or 'cpu').
        print_progress (bool): Whether to print progress and loss during optimization.
    
    Returns:
        maximized_units (torch.Tensor): Tensor storing the optimized inputs at each iteration. 
                                        Shape: (max_iter, block_size)
    """
    # Move model to the specified device and set it to evaluation mode
    model.to(device)
    model.eval()

    # Initialize the input tensor with random noise (shape: (block_size,))
    input_tensor = torch.randn((block_size,), dtype=torch.float32, requires_grad=True, device=device)

    # Tensor to store input history across iterations
    maximized_units = torch.zeros((max_iter, block_size), dtype=torch.float32, device=device)

    # Adam optimizer for optimizing the input tensor
    optimizer = Adam([input_tensor], lr=lr, weight_decay=weight_decay)

    # Helper function to compute the loss for specified units
    def compute_loss(model, input_tensor, layers_channels_dict):
        total_loss = 0.0
        hooks = []

        # Hook function to capture activations
        activations = {}

        def hook_fn(layer_idx, module, input, output):
            activations[layer_idx] = output

        # Register hooks for specified layers
        for layer_idx in layers_channels_dict:
            layer = model.transformer.h[layer_idx]
            hooks.append(layer.register_forward_hook(lambda m, i, o, idx=layer_idx: hook_fn(idx, m, i, o)))

        # Forward pass through the model
        with torch.no_grad():
            logits, _ = model(input_tensor.unsqueeze(0).argmax(dim=-1))  # Add batch dimension

        # Compute loss as the sum of activations for specified channels
        for layer_idx, channels in layers_channels_dict.items():
            layer_activations = activations[layer_idx]  # Shape: (1, block_size, embedding_dim)
            for channel in channels:
                total_loss += layer_activations[0, :, channel].sum()

        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()

        return total_loss

    # Optimization loop
    pbar = trange(max_iter) if print_progress else range(max_iter)
    for i in pbar:
        optimizer.zero_grad()
        loss = compute_loss(model, input_tensor, layers_channels_dict)
        (-loss).backward()  # Negative loss for gradient ascent
        optimizer.step()

        # Store the current input tensor in the history
        maximized_units[i] = input_tensor.clone().detach()

        if print_progress:
            pbar.set_description(f"Iter {i+1}/{max_iter}, Loss: {loss.item():.4f}")

    return maximized_units

def optimize_single_unit_with_tv(model, layer_id, channel_id, 
                                 lr=1e-2, weight_decay=0e-4, max_iter=500, opt_type = "all",
                                 block_size=1024, vocab_size=50304, tv_weight=1e-4, 
                                 device='cuda', print_progress=True):
    """
    Optimize the input to maximize the activation of a single unit in the model,
    with Total Variation (TV) regularization.
    
    Args:
        model (torch.nn.Module): Loaded GPT model.
        layer_id (int): Layer index of the unit to optimize.
        channel_id (int): Channel index of the unit to optimize.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        max_iter (int): Number of optimization iterations.
        block_size (int): Length of the input sequence.
        vocab_size (int): Size of the vocabulary.
        tv_weight (float): Weight for the TV regularization term.
        device (str): Device to perform computation ('cuda' or 'cpu').
        print_progress (bool): Whether to display a progress bar.
    
    Returns:
        history (torch.Tensor): Tensor storing the optimized input at each iteration.
                                Shape: (max_iter, 1, block_size)
    """
    # Move the model to the specified device and set it to evaluation mode
    model.to(device)
    model.eval()

    # Initialize the input tensor with random integers in the vocabulary range
    #input_tensor = torch.randint(0, vocab_size, (1, block_size), device=device, dtype=torch.long)
    input_tensor = torch.zeros((1, block_size), dtype=torch.long, device=device)

    # Tensor to store input history across iterations
    history = torch.zeros((max_iter, 1, block_size), dtype=torch.long, device=device)
    score_hist = torch.zeros((max_iter, 1), dtype=torch.float32, device=device)

    # Create a float tensor for optimization
    float_input = input_tensor.float().to(device)
    float_input.requires_grad = True

    # Adam optimizer
    optimizer = Adam([float_input], lr=lr, weight_decay=weight_decay)

    # Hook to capture activations
    activations = None

    def hook_fn(module, input, output):
        nonlocal activations
        activations = output  # Keep activations connected to the computational graph

    # Attach a forward hook to the specified layer
    hook = model.transformer.h[layer_id].register_forward_hook(hook_fn)

    # Optimization loop
    pbar = trange(max_iter) if print_progress else range(max_iter)
    for i in pbar:
        optimizer.zero_grad()  # Zero the gradients

        print(':::::::::--> input_tensor: ', float_input)

        # Forward pass
        logits = model(float_input.argmax(dim=-1).unsqueeze(0))  # Ensure correct input shape

        # Loss: maximize the activation of the specific unit
        if opt_type == "all":
            activation_loss = -activations[:, :, channel_id].mean(dim=-1) 
        elif opt_type == "last":
            activation_loss = -activations[:, -1, channel_id]

        # TV regularization term
        tv_loss = tv_weight * torch.sum(torch.abs(float_input[:, 1:] - float_input[:, :-1]))

        # Total loss
        loss = activation_loss + tv_loss
        loss.backward()

        # Step the optimizer
        optimizer.step()


        # Clamp float_input values to keep them within valid range
        #with torch.no_grad():
        #    float_input.clamp_(0, vocab_size - 1)  # Clamp values to valid range

        # Convert to integer tokens and save to history
        input_tensor = float_input.argmax(dim=-1).long()  # Convert back to integers
        history[i] = input_tensor.clone()
        score_hist[i] = activation_loss.clone().detach()

        # Debug: Print loss
        if print_progress:
            #pbar.set_description(f"Iter {i+1}/{max_iter}, Loss: {-activation_loss.item():.4f}, TV Loss: {tv_loss.item():.4f}")
            pbar.set_description(f"Iter {i+1}/{max_iter}, Loss: {-activation_loss.item():.4f}")

    # Remove the hook after optimization
    hook.remove()

    return history, score_hist

def optimize_single_unit_debug(model, layer_id, channel_id, 
                                 lr=1e-2, weight_decay=0e-4, max_iter=500, opt_type="all",
                                 block_size=1024, vocab_size=50304, device='cuda', print_progress=True):
    """
    Optimize the input to maximize the activation of a single unit in the model,
    with proper gradient flow and no reliance on discrete indices during the forward pass.
    
    Args:
        model (torch.nn.Module): Loaded GPT model.
        layer_id (int): Layer index of the unit to optimize.
        channel_id (int): Channel index of the unit to optimize.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        max_iter (int): Number of optimization iterations.
        block_size (int): Length of the input sequence.
        vocab_size (int): Size of the vocabulary.
        device (str): Device to perform computation ('cuda' or 'cpu').
        print_progress (bool): Whether to display a progress bar.
    
    Returns:
        history (torch.Tensor): Tensor storing the optimized input at each iteration.
                                Shape: (max_iter, 1, block_size)
    """
    model.to(device)
    model.eval()

    # Initialize float_input as a learnable tensor
    float_input = torch.randn((1, block_size, vocab_size), device=device, requires_grad=True)

    # Tensor to store input history across iterations
    history = torch.zeros((max_iter, 1, block_size), dtype=torch.long, device=device)
    score_hist = torch.zeros((max_iter, 1), dtype=torch.float32, device=device)

    optimizer = Adam([float_input], lr=lr, weight_decay=weight_decay)

    activations = None

    def hook_fn(module, input, output):
        nonlocal activations
        activations = output

    hook = model.transformer.h[layer_id].register_forward_hook(hook_fn)

    pbar = trange(max_iter) if print_progress else range(max_iter)
    for i in pbar:
        optimizer.zero_grad()

        # Forward pass using softmax probabilities
        probs = torch.nn.functional.softmax(float_input, dim=-1)  # Convert to probabilities
        logits = model(probs.argmax(dim=-1))  # Discrete tokens passed to the model

        if activations is None:
            raise RuntimeError("Activations not captured by the hook! Check the layer index.")

        if opt_type == "all":
            activation_loss = -activations[:, :, channel_id].mean(dim=-1)
        elif opt_type == "last":
            activation_loss = -activations[:, -1, channel_id].sum()

        loss = activation_loss
        loss.backward()

        # Debug: Check gradients
        if float_input.grad is None:
            raise RuntimeError("Gradients not computed for float_input!")

        # Step the optimizer
        optimizer.step()

        # Convert to integer tokens and save to history
        with torch.no_grad():
            discrete_input = probs.argmax(dim=-1).long()
            history[i] = discrete_input.clone()
            score_hist[i] = activation_loss.clone().detach()

        if print_progress:
            pbar.set_description(f"Iter {i+1}/{max_iter}, Loss: {-activation_loss.item():.4f}")

    hook.remove()

    return history, score_hist

def optimize_single_unit_CholeskyCMAES(
    model, layer_id, channel_id, max_iter=100, opt_type="all",
    block_size=1024, vocab_size=50304, device='cuda', print_progress=True,
    init_sigma=3.0, Aupdate_freq=10, maximize=True, random_seed=None, optim_params={},
    penalty_weight=1e-2, penalty_type="syll_change", init_code=None, init_code_type = 'zeros'
    ):
    """
    Optimize the input to maximize the activation of a single unit in the model using CholeskyCMAES with TV normalization.

    Args:
        model (torch.nn.Module): Loaded GPT model.
        layer_id (int): Layer index of the unit to optimize.
        channel_id (int): Channel index of the unit to optimize.
        max_iter (int): Number of optimization iterations.
        block_size (int): Length of the input sequence.
        vocab_size (int): Size of the vocabulary.
        device (str): Device to perform computation ('cuda' or 'cpu').
        print_progress (bool): Whether to display a progress bar.
        init_sigma (float): Initial standard deviation for CMA-ES.
        Aupdate_freq (int): Frequency of covariance matrix updates.
        maximize (bool): Whether to maximize or minimize the score.
        random_seed (int): Seed for random number generation.
        optim_params (dict): Additional parameters for the optimizer.
        tv_weight (float): Weight for Total Variation (TV) normalization.
        penalty_weight (float): Weight for the penalty term.
        penalty_type (str): Type of penalty term ('syll_change' or None).
        init_code (torch.Tensor): Initial input tensor for optimization.
        init_code_type (str): Type of initialization ('zeros' or 'random').

    Returns:
        opt_hist (torch.Tensor): History of optimized inputs across iterations.
                                 Shape: (max_iter, population_size, block_size).
        score_hist (torch.Tensor): History of scores across iterations.
                                   Shape: (max_iter, population_size).
    """
    # Move the model to the specified device and set it to evaluation mode
    model.to(device)
    model.eval()

    # Initialize input tensor (you can also use random initialization if needed) if not provided
    if init_code is not None:
        if init_code_type == 'zeros':
            input_tensor = torch.zero((1, block_size), dtype=torch.long, device=device)
        elif init_code_type == 'random':
            input_tensor = torch.randint(0, vocab_size, (1, block_size), device=device, dtype=torch.long)
        else:
            raise ValueError(f"Invalid init_code_type: {init_code_type}") 

    # Tensor to store input history across iterations
    opt_hist = torch.zeros((max_iter, 30, block_size), dtype=torch.long, device=device)
    score_hist = torch.zeros((max_iter, 30), dtype=torch.float32, device=device)

    # Initialize the optimizer with CMA-ES
    new_codes = torch.zeros([1, block_size], dtype=torch.long)

    optimizer = CholeskyCMAES_torch(
        block_size, population_size=None, init_sigma=init_sigma,
        init_code=new_codes, Aupdate_freq=Aupdate_freq,
        maximize=maximize, random_seed=random_seed, optim_params=optim_params
    )

    # Hook to capture activations
    activations = None

    def hook_fn(module, input, output):
        nonlocal activations
        activations = output  # Capture activations from the specified layer

    # Attach a forward hook to the specified layer
    hook = model.transformer.h[layer_id].register_forward_hook(hook_fn)

    # Optimization loop
    pbar = trange(max_iter) if print_progress else range(max_iter)
    for i in pbar:
        # Ensure input_tensor is on the correct device
        input_tensor = input_tensor.to(device)

        # Perform forward pass and compute the score
        with torch.no_grad():
            _ = model(input_tensor)
            if opt_type == "all":
                score_act = activations[:, :, channel_id].mean(dim=-1)
            elif opt_type == "last":
                score_act = activations[:, -1, channel_id]
            else:
                raise ValueError(f"Invalid opt_type: {opt_type}")

        # Apply normalization pemalty
        score = score_act
        penalty = 0
        if penalty_type is not None:
            if penalty_type == "syll_change":
                # Penalize for syllable changes in the input -> idea is to prevent rapid changes
                penalty = penalty_weight*torch.sum((input_tensor[:, 1:] != input_tensor[:, :-1]).float(), dim=-1)
                score = score_act - penalty

        # Update the codes using the optimizer and the score
        new_codes = optimizer.step_simple(score, new_codes)

        # Update the input tensor and history
        input_tensor = new_codes.clone().detach().long()
        input_tensor.clamp_(0, vocab_size - 1)  # Clamp values to valid range

        opt_hist[i, 0: input_tensor.shape[0], 0: input_tensor.shape[1]] = input_tensor.clone()
        score_hist[i, 0: score.shape[0]] = score.clone()

        # Debug: Print progress
        if print_progress:
            pbar.set_description(f"Iter {i+1}/{max_iter}, act = {score_act.mean().item():.4f}, Score mean: {score.mean().item():.4f},  penalty: {penalty.mean().item():.4f}")

    # Remove the hook after optimization
    hook.remove()

    return opt_hist, score_hist


if __name__ == "__main__":

    # Load model
    checkpoint_path = "out-moSeq-syll_e1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model_from_checkpoint(checkpoint_path, device)

    block_size = model.config.block_size
    vocab_size = model.config.vocab_size
    print(f"Block size: {block_size}, Vocab size: {vocab_size}")
    """
    # Define target units (layer-channel pairs)
    layers_channels_dict = {
        0: [1, 5],  # Maximize channels 10 and 20 in layer 3
        1: [2],      # Maximize channel 50 in layer 5
        2: [3, 4]       # Maximize channel 100 in layer 7
    }

    # Run unit maximization
    optimized_history = unit_maximization(
        model=model,
        layers_channels_dict=layers_channels_dict,
        lr=1e-2,
        weight_decay=1e-2,
        max_iter=50,
        block_size=block_size,  # Input sequence length
        device=device,
        print_progress=True
    )

    # Analyze the optimized history
    print("Optimized History Shape:", optimized_history.shape)  # (1000, 1024)
    """

     # Parameters
    layer_id = 5      # Layer index
    channel_id = 10    # Channel index

    # Optimize the unit
    """optimized_input = optimize_single_unit_1(
        model=model,
        layer_id=layer_id,
        channel_id=channel_id,
        vocab_size=vocab_size,
        lr=1e-2,
        weight_decay=1e-4,
        max_iter=50,
        block_size=block_size,
        device=device,
        print_progress=True
    )

    

    optimized_input, score_hist = optimize_single_unit_with_tv_debug(
        model=model,
        layer_id=layer_id,
        channel_id=channel_id,
        vocab_size=vocab_size,
        block_size=block_size,
        device=device,
        opt_type="last",
        max_iter = 300,
        print_progress=True,
        lr=1e-2, 
        weight_decay=0e-4, 
    )"""

    optimized_input, score_hist = optimize_single_unit_CholeskyCMAES(
        model=model,
        layer_id=layer_id,
        channel_id=channel_id,
        vocab_size=vocab_size,
        block_size=block_size,
        device=device,
        opt_type="last",
        max_iter = 5000,
        print_progress=True,
        init_sigma=15.0, 
        Aupdate_freq=10, 
        maximize=True, 
        random_seed=None,
        penalty_weight=1, 
    )
    
    # let print the first the optimized input
    print(" optimized input shape:", optimized_input.shape)
    print(" optimized input:", optimized_input[-1, 0, :])
    print(" optimized input example:", optimized_input[-1, :, :])
    # let plot the score history 
    import matplotlib.pyplot as plt
    #import time
    score_mean = torch.mean(score_hist, dim=-1).cpu().numpy()
    plt.plot(score_mean[2:])
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Score History")
    plt.show()
    # wait for 5 seconds and then close the plot
    #time.sleep(5)
    #plt.close()
    # let print the last optimized input


    



