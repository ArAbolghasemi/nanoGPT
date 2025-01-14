import time
import numpy as np
#from numpy import sqrt, zeros, floor, log, log2, eye, exp, linspace, logspace, log10, mean, std
#from numpy.linalg import norm
#from numpy.random import randn


class CholeskyCMAES_numpy:
    """ Note this is a variant of CMAES Cholesky suitable for high dimensional optimization"""
    def __init__(self, space_dimen, population_size=None, init_sigma=3.0, init_code=None, Aupdate_freq=10,
                 maximize=True, random_seed=None, optim_params={}):
        N = space_dimen
        self.space_dimen = space_dimen
        # Overall control parameter
        self.maximize = maximize  # if the program is to maximize or to minimize
        # Strategy parameter setting: Selection
        if population_size is None:
            self.lambda_ = int(4 + floor(3 * log2(N)))  # population size, offspring number
            # the relation between dimension and population size.
        else:
            self.lambda_ = population_size  # use custom specified population size
        mu = self.lambda_ / 2  # number of parents/points for recombination
        #  Select half the population size as parents
        weights = log(mu + 1 / 2) - (log(np.arange(1, 1 + floor(mu))))  # muXone array for weighted recombination
        self.mu = int(floor(mu))
        self.weights = weights / sum(weights)  # normalize recombination weights array
        mueff = self.weights.sum() ** 2 / sum(self.weights ** 2)  # variance-effectiveness of sum w_i x_i
        self.weights.shape = (1, -1)  # Add the 1st dim 1 to the weights mat
        self.mueff = mueff  # add to class variable
        self.sigma = init_sigma  # Note by default, sigma is None here.
        print("Space dimension: %d, Population size: %d, Select size:%d, Optimization Parameters:\nInitial sigma: %.3f"
              % (self.space_dimen, self.lambda_, self.mu, self.sigma))
        # Strategy parameter setting: Adaptation
        self.cc = 4 / (N + 4)  # defaultly  0.0009756
        self.cs = sqrt(mueff) / (sqrt(mueff) + sqrt(N))  # 0.0499
        self.c1 = 2 / (N + sqrt(2)) ** 2  # 1.1912701410022985e-07
        if "cc" in optim_params.keys():  # if there is outside value for these parameter, overwrite them
            self.cc = optim_params["cc"]
        if "cs" in optim_params.keys():
            self.cs = optim_params["cs"]
        if "c1" in optim_params.keys():
            self.c1 = optim_params["c1"]
        self.damps = 1 + self.cs + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1)  # damping for sigma usually  close to 1

        print("cc=%.3f, cs=%.3f, c1=%.3f damps=%.3f" % (self.cc, self.cs, self.c1, self.damps))
        if init_code is not None:
            self.init_x = np.asarray(init_code)
            self.init_x.shape = (1, N)
        else:
            self.init_x = None  # FIXED Nov. 1st
        self.xmean = zeros((1, N))
        self.xold = zeros((1, N))
        # Initialize dynamic (internal) strategy parameters and constants
        self.pc = zeros((1, N))
        self.ps = zeros((1, N))  # evolution paths for C and sigma
        self.A = eye(N, N)  # covariant matrix is represent by the factors A * A '=C
        self.Ainv = eye(N, N)

        self.eigeneval = 0  # track update of B and D
        self.counteval = 0
        if Aupdate_freq is None:
            self.update_crit = self.lambda_ / self.c1 / N / 10
        else:
            self.update_crit = Aupdate_freq * self.lambda_
        self.chiN = sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
        # expectation of ||N(0,I)|| == norm(randn(N,1)) in 1/N expansion formula
        self._istep = 0

    def get_init_pop(self):
        return self.init_x

    def step_simple(self, scores, codes, verbosity=1, printmessage=False):
        """ Taking scores and codes to return new codes, without generating images
        Used in cases when the images are better handled in outer objects like Experiment object
        """
        # Note it's important to decide which variable is to be saved in the `Optimizer` object
        # Note to confirm with other code, this part is transposed.
        # set short name for everything to simplify equations
        N = self.space_dimen
        # lambda_, mu, mueff, chiN = self.lambda_, self.mu, self.mueff, self.chiN
        # cc, cs, c1, damps = self.cc, self.cs, self.c1, self.damps
        # sigma, A, Ainv, ps, pc, = self.sigma, self.A, self.Ainv, self.ps, self.pc,
        # Sort by fitness and compute weighted mean into xmean
        if self.maximize is False:
            code_sort_index = np.argsort( scores)  # add - operator it will do maximization.
        else:
            code_sort_index = np.argsort(-scores)
        # scores = scores[code_sort_index]  # Ascending order. minimization
        if self._istep == 0:
            # Population Initialization: if without initialization, the first xmean is evaluated from weighted average all the natural images
            if self.init_x is None:
                select_n = len(code_sort_index[0:self.mu])
                temp_weight = self.weights[:, :select_n] / np.sum(self.weights[:, :select_n]) # in case the codes is not enough
                self.xmean = temp_weight @ codes[code_sort_index[0:self.mu], :]
            else:
                self.xmean = self.init_x
        else:
            self.xold = self.xmean
            self.xmean = self.weights @ codes[code_sort_index[0:self.mu], :]  # Weighted recombination, new mean value
            # Cumulation statistics through steps: Update evolution paths
            randzw = self.weights @ self.randz[code_sort_index[0:self.mu], :]
            self.ps = (1 - self.cs) * self.ps + sqrt(self.cs * (2 - self.cs) * self.mueff) * randzw
            self.pc = (1 - self.cc) * self.pc + sqrt(self.cc * (2 - self.cc) * self.mueff) * randzw @ self.A
            # Adapt step size sigma
            self.sigma = self.sigma * exp((self.cs / self.damps) * (norm(self.ps) / self.chiN - 1))
            # self.sigma = self.sigma * exp((self.cs / self.damps) * (norm(ps) / self.chiN - 1))
            if verbosity:
                if printmessage: print("sigma: %.2f" % self.sigma)
            # Update A and Ainv with search path
            if self.counteval - self.eigeneval > self.update_crit:  # to achieve O(N ^ 2) do decomposition less frequently
                self.eigeneval = self.counteval
                t1 = time.time()
                v = self.pc @ self.Ainv
                normv = v @ v.T
                # Directly update the A Ainv instead of C itself
                self.A = sqrt(1 - self.c1) * self.A + sqrt(1 - self.c1) / normv * (
                            sqrt(1 + normv * self.c1 / (1 - self.c1)) - 1) * v.T @ self.pc  # FIXME, dimension error, # FIXED aug.13th
                self.Ainv = 1 / sqrt(1 - self.c1) * self.Ainv - 1 / sqrt(1 - self.c1) / normv * (
                            1 - 1 / sqrt(1 + normv * self.c1 / (1 - self.c1))) * self.Ainv @ v.T @ v
                t2 = time.time()
                if printmessage: print("A, Ainv update! Time cost: %.2f s" % (t2 - t1))
        # Generate new sample by sampling from Gaussian distribution
        # new_samples = zeros((self.lambda_, N))
        self.randz = randn(self.lambda_, N)  # save the random number for generating the code.
        new_samples = self.xmean + self.sigma * self.randz @ self.A
        self.counteval += self.lambda_
        # for k in range(self.lambda_):
        #     new_samples[k:k + 1, :] = self.xmean + sigma * (self.randz[k, :] @ A)  # m + sig * Normal(0,C)
        #     # Clever way to generate multivariate gaussian!!
        #     # Stretch the guassian hyperspher with D and transform the
        #     # ellipsoid by B mat linear transform between coordinates
        #     self.counteval += 1
        # self.sigma, self.A, self.Ainv, self.ps, self.pc = sigma, A, Ainv, ps, pc,
        self._istep += 1
        return new_samples

import torch

class CholeskyCMAES_torch:
    """Variant of CMAES with Cholesky decomposition, adapted for PyTorch."""
    def __init__(self, space_dimen, population_size=None, init_sigma=3.0, init_code=None, Aupdate_freq=10,
                 maximize=True, random_seed=None, optim_params={}):
        """Initialize the CMA-ES optimizer.

        Args:
            space_dimen (int): Dimensionality of the search space.
            population_size (int, optional): Number of candidate solutions per generation. Defaults to a function of space_dimen.
            init_sigma (float): Initial step size.
            init_code (list or torch.Tensor, optional): Initial mean of the population.
            Aupdate_freq (int): Frequency of covariance matrix updates.
            maximize (bool): Whether to maximize or minimize the objective.
            optim_params (dict): Custom parameters to override default strategy settings.
            random_seed (int): Random seed for reproducibility.
        """
        # Set random seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        self.space_dimen = torch.tensor(space_dimen)
        self.maximize = torch.tensor(maximize)

        # Determine population size
        self.lambda_ = torch.tensor(population_size if population_size is not None else
                        int(4 + torch.floor(3 * torch.log2(torch.tensor(self.space_dimen, dtype=torch.float32)))))

        # Set up recombination weights
        mu = self.lambda_ / 2
        weights = torch.log(mu + 0.5) - torch.log(torch.arange(1, 1 + torch.floor(mu), dtype=torch.float32))
        #weights = torch.log(torch.tensor(mu + 0.5)) - torch.log(torch.arange(1, 1 + torch.floor(torch.tensor(mu)), dtype=torch.float32))
        self.mu = int(torch.floor(mu))
        self.weights = weights / weights.sum()
        self.weights = self.weights.view(1, -1)
        self.mueff = self.weights.sum() ** 2 / (self.weights ** 2).sum()

        # Strategy parameters
        self.sigma = torch.tensor(init_sigma)
        self.cc = optim_params.get("cc", 4 / (self.space_dimen + 4))
        self.cs = optim_params.get("cs", torch.sqrt(self.mueff) / (torch.sqrt(self.mueff) + torch.sqrt(self.space_dimen)))
        self.c1 = optim_params.get("c1", 2 / (self.space_dimen + torch.sqrt(torch.tensor(2))) ** 2)
        self.damps = 1 + self.cs + 2 * max(0, torch.sqrt((self.mueff - 1) / (self.space_dimen + 1)) - 1)

        # Initialization messages
        print(f"Space dimension: {self.space_dimen}, Population size: {self.lambda_}, Select size: {self.mu},\nInitial sigma: {self.sigma:.3f}")
        print(f"cc={self.cc:.3f}, cs={self.cs:.3f}, c1={self.c1:.3f}, damps={self.damps:.3f}")

        # Initialize population mean
        if init_code is not None:
            self.init_x = torch.tensor(init_code, dtype=torch.float32).view(1, self.space_dimen)
        else:
            self.init_x = None
        self.xmean = torch.zeros((1, self.space_dimen), dtype=torch.float32)
        self.xold = torch.zeros((1, self.space_dimen), dtype=torch.float32)

        # Initialize dynamic strategy parameters
        self.pc = torch.zeros((1, self.space_dimen), dtype=torch.float32)
        self.ps = torch.zeros((1, self.space_dimen), dtype=torch.float32)
        self.A = torch.eye(self.space_dimen, dtype=torch.float32)
        self.Ainv = torch.eye(self.space_dimen, dtype=torch.float32)

        # Miscellaneous parameters
        self.eigeneval = 0
        self.counteval = 0
        self.update_crit = Aupdate_freq * self.lambda_ if Aupdate_freq is not None else self.lambda_ / self.c1 / self.space_dimen / 10
        self.chiN = torch.sqrt(self.space_dimen) * (1 - 1 / (4 * self.space_dimen) + 1 / (21 * self.space_dimen ** 2))
        self._istep = 0

    def get_init_pop(self):
        """Retrieve the initial population."""
        return self.init_x

    def step_simple(self, scores, codes, verbosity=1, printmessage=False):
        """Perform a single optimization step.

        Args:
            scores (list or torch.Tensor): Fitness values of the population.
            codes (list or torch.Tensor): Candidate solutions.
            verbosity (int): Verbosity level for logging.
            printmessage (bool): Whether to print intermediate messages.

        Returns:
            torch.Tensor: New population of candidate solutions.
        """
        N = self.space_dimen
        # convert inputs to tensors if necessary
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, dtype=torch.float32)
        if not isinstance(codes, torch.Tensor):
            codes = torch.tensor(codes, dtype=torch.float32)

        # Sort population by fitness
        code_sort_index = torch.argsort(-scores if self.maximize else scores)

        if self._istep == 0:
            # Initialize mean if not provided
            if self.init_x is None:
                select_n = len(code_sort_index[:self.mu])
                temp_weight = self.weights[:, :select_n] / self.weights[:, :select_n].sum()
                self.xmean = temp_weight @ codes[code_sort_index[:self.mu]]
            else:
                self.xmean = self.init_x
        else:
            # Update mean
            self.xold = self.xmean
            self.xmean = self.weights @ codes[code_sort_index[:self.mu].detach().cpu()]

            # Update evolution paths
            randzw = self.weights @ self.randz[code_sort_index[:self.mu].detach().cpu()]
            self.ps = (1 - self.cs) * self.ps + torch.sqrt(self.cs * (2 - self.cs) * self.mueff) * randzw
            self.pc = (1 - self.cc) * self.pc + torch.sqrt(self.cc * (2 - self.cc) * self.mueff) * randzw @ self.A

            # Adapt step size
            self.sigma *= torch.exp((self.cs / self.damps) * (torch.norm(self.ps) / self.chiN - 1))

            if verbosity and printmessage:
                print(f"sigma: {self.sigma:.2f}")

            # Update covariance matrix if necessary
            if self.counteval - self.eigeneval > self.update_crit:
                self.eigeneval = self.counteval
                v = self.pc @ self.Ainv
                normv = (v @ v.T).item()

                # Update Cholesky factors
                self.A = torch.sqrt(1 - self.c1) * self.A + torch.sqrt(1 - self.c1) / normv * (
                    torch.sqrt(1 + normv * self.c1 / (1 - self.c1)) - 1) * v.T @ self.pc
                self.Ainv = 1 / torch.sqrt(1 - self.c1) * self.Ainv - 1 / torch.sqrt(1 - self.c1) / normv * (
                    1 - 1 / torch.sqrt(1 + normv * self.c1 / (1 - self.c1))) * self.Ainv @ v.T @ v

                if printmessage:
                    print("A, Ainv updated.")

        # Generate new samples
        self.randz = torch.randn(self.lambda_, N, dtype=torch.float32)
        new_samples = self.xmean + self.sigma * self.randz @ self.A
        self.counteval += self.lambda_
        self._istep += 1

        return new_samples
