import torch
import os
from model import GPTConfig, GPT

def load_model_from_checkpoint(out_dir, device='cuda'):
    """
    Load a GPT model from a saved checkpoint.
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to load the model onto.
    Returns:
        model (torch.nn.Module): Loaded GPT model.
        model_config (dict): Configuration of the model.
    """
    # Load the checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    model_config = GPTConfig(**model_args)
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded and ready for inference!")
    return model, model_config

def print_model_layers(model):
    """
    Print all the layers in the model and their details.
    Args:
        model (torch.nn.Module): The GPT model.
    """
    print("\nModel Layers:")
    for idx, layer in enumerate(model.transformer.h):
        print(f"Layer {idx}: {layer}")

def get_layer_activations(model, layer_idx, input_tensor, submodule='final', device='cuda'):
    """
    Get activations of a specific part of a layer for given inputs.
    Args:
        model (torch.nn.Module): Loaded GPT model.
        layer_idx (int): Index of the desired layer (0-based indexing).
        input_tensor (torch.Tensor): Input tensor of shape (B, block_size).
        submodule (str): Specify which part of the layer to capture ('ln_1', 'attn', 'residual_1',
                         'ln_2', 'mlp', 'residual_2', 'final').
        device (str): Device for computation.
    Returns:
        activations (torch.Tensor): Activations of the specified submodule or final output.
    """
    activations = None

    # Hook to capture activations
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.clone().detach()

    # Attach the hook based on the specified submodule
    if submodule == 'ln_1':
        handle = model.transformer.h[layer_idx].ln_1.register_forward_hook(hook_fn)
    elif submodule == 'attn':
        handle = model.transformer.h[layer_idx].attn.register_forward_hook(hook_fn)
    elif submodule == 'residual_1':
        # Custom hook to capture after Residual Connection 1
        def residual_1_hook(module, input, output):
            nonlocal activations
            activations = input[0] + module.attn(output).clone().detach()

        handle = model.transformer.h[layer_idx].register_forward_hook(residual_1_hook)
    elif submodule == 'ln_2':
        handle = model.transformer.h[layer_idx].ln_2.register_forward_hook(hook_fn)
    elif submodule == 'mlp':
        handle = model.transformer.h[layer_idx].mlp.register_forward_hook(hook_fn)
    elif submodule == 'residual_2':
        # Custom hook to capture after Residual Connection 2
        def residual_2_hook(module, input, output):
            nonlocal activations
            activations = input[0] + module.mlp(output).clone().detach()

        handle = model.transformer.h[layer_idx].register_forward_hook(residual_2_hook)
    elif submodule == 'final':
        handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Invalid submodule '{submodule}'. Choose from 'ln_1', 'attn', 'residual_1', 'ln_2', 'mlp', 'residual_2', or 'final'.")

    # Forward pass
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove the hook after forward pass
    handle.remove()

    return activations
"""
# Example usage
if __name__ == "__main__":
    # Parameters
    checkpoint_path = "out-moSeq-syll_e1"
    layer_index = 0  # Example: Get activations from the 4th layer
    B, block_size = 8, 256  # Batch size and sequence length
    vocab_size = 30  # Example GPT vocab size

    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, model_config = load_model_from_checkpoint(checkpoint_path, device)

    # Print model layers
    print_model_layers(model)

    # Create dummy input tensor
    input_tensor = torch.randint(0, vocab_size, (B, block_size), dtype=torch.long)

    # Get activations
    activations = get_layer_activations(model, layer_index, input_tensor, device=device)

    print(f"Activations shape: {activations.shape}")
    # let print the first 5 elements of the activations
    print(activations[0, :5])


    # Get activations for LayerNorm 1
    ln_1_activations = get_layer_activations(model, layer_index, input_tensor, submodule='ln_1', device=device)
    print(f"LayerNorm 1 Activations shape: {ln_1_activations.shape}")
    print(ln_1_activations[0, :5])

    # Get activations for Attention
    attn_activations = get_layer_activations(model, layer_index, input_tensor, submodule='attn', device=device)
    print(f"Attention Activations shape: {attn_activations.shape}")
    print(attn_activations[0, :5])

    # Get activations for Residual Connection 1
    residual_1_activations = get_layer_activations(model, layer_index, input_tensor, submodule='residual_1', device=device)
    print(f"Residual Connection 1 Activations shape: {residual_1_activations.shape}")
    print(residual_1_activations[0, :5])

    # Get activations for LayerNorm 2
    ln_2_activations = get_layer_activations(model, layer_index, input_tensor, submodule='ln_2', device=device)
    print(f"LayerNorm 2 Activations shape: {ln_2_activations.shape}")

    # Get activations for MLP
    mlp_activations = get_layer_activations(model, layer_index, input_tensor, submodule='mlp', device=device)
    print(f"MLP Activations shape: {mlp_activations.shape}")
    print(mlp_activations[0, :5])

    # Get activations for Residual Connection 2
    residual_2_activations = get_layer_activations(model, layer_index, input_tensor, submodule='residual_2', device=device)
    print(f"Residual Connection 2 Activations shape: {residual_2_activations.shape}")

    # Get final layer output
    final_activations = get_layer_activations(model, layer_index, input_tensor, submodule='final', device=device)
    print(f"Final Layer Activations shape: {final_activations.shape}")
"""