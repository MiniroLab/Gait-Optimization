import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# =============================================================================
#            (Re)Use your simulation code in a multi-objective function
# =============================================================================
def simulate_speed_and_cost(a_param_opt, stance_freq_opt, swing_freq_opt, 
                              in_phase_coupling_opt, out_phase_coupling_opt):
    """
    Runs a headless simulation using provided CPG parameters and returns two objectives:
      1. Average forward speed (to maximize).
      2. Negative cost of transport (so that maximizing it is equivalent to minimizing cost).
      
    (Replace the dummy simulation below with your full simulation code that 
    computes these metrics.)
    """
    # === REPLACE THIS DUMMY EVALUATION WITH YOUR MUJOCO SIMULATION CODE ===
    # For demonstration, we use random numbers:
    avg_speed = np.random.uniform(0.01, 0.05)  # e.g., m/s
    # Let's assume the cost of transport (CoT) is computed as:
    # CoT = total_energy / (weight * distance_traveled)
    # Here we simulate it as a random number:
    cost_of_transport = np.random.uniform(5, 10)
    # Since we want to maximize speed and minimize cost, we return:
    return np.array([avg_speed, -cost_of_transport])
    # ========================================================================

# =============================================================================
#          Define parameter bounds and keys for the optimization
# =============================================================================
pbounds = {
    'a_param_opt': (5, 25),
    'stance_freq_opt': (1.0, 8.0),
    'swing_freq_opt': (1.0, 8.0),
    'in_phase_coupling_opt': (0.2, 1.5),
    'out_phase_coupling_opt': (-1.5, -0.2)
}
keys = list(pbounds.keys())
d = len(keys)

# =============================================================================
#                    Initial Design via Sobol Sampling
# =============================================================================
N_INIT = 10
# Create bounds tensor of shape (2, d) where first row is lower bounds and second is upper bounds.
bounds = torch.tensor(
    [[pbounds[key][0] for key in keys],
     [pbounds[key][1] for key in keys]], dtype=torch.double
)

# Draw N_INIT Sobol samples
sobol_samples = draw_sobol_samples(bounds=bounds, n=1, q=N_INIT).squeeze(0)
initial_X = sobol_samples  # shape (N_INIT, d)

# Evaluate the simulation at the initial points
def evaluate_simulation(X):
    # X: tensor of shape (n, d)
    values = []
    for i in range(X.shape[0]):
        params = {keys[j]: X[i, j].item() for j in range(d)}
        out = simulate_speed_and_cost(**params)
        values.append(out)
    return torch.tensor(values, dtype=torch.double)

Y_init = evaluate_simulation(initial_X)  # shape (N_INIT, 2)

# =============================================================================
#                   Fit initial multi-output GP model
# =============================================================================
# Here we fit independent GPs for each objective.
models = []
for i in range(Y_init.shape[1]):
    gp = SingleTaskGP(initial_X, Y_init[:, i:i+1])
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    models.append(gp)
model = ModelListGP(*models)

# For multi-objective acquisition (EHVI) all objectives must be maximized.
# Our simulation returns [speed, -cost_of_transport], so both are to be maximized.
# Set a reference point that is dominated by all observations.
ref_point = torch.min(Y_init, dim=0)[0] - 0.1

acq_func = qExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    sampler=None,  # uses default Monte Carlo sampler
)

# =============================================================================
#                    Multi-Objective Bayesian Optimization Loop
# =============================================================================
N_ITER = 20
all_X = [initial_X]
all_Y = [Y_init]

for iteration in range(N_ITER):
    # Optimize the acquisition function to obtain one candidate (q=1)
    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )
    new_x = candidate.detach()  # shape (1, d)
    new_y = evaluate_simulation(new_x)
    
    # Update our data
    all_X.append(new_x)
    all_Y.append(new_y)
    X_all = torch.cat(all_X, dim=0)
    Y_all = torch.cat(all_Y, dim=0)
    
    # Refit each GP model with the updated data
    models = []
    for i in range(Y_all.shape[1]):
        gp = SingleTaskGP(X_all, Y_all[:, i:i+1])
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        models.append(gp)
    model = ModelListGP(*models)
    
    # Update reference point and acquisition function
    ref_point = torch.min(Y_all, dim=0)[0] - 0.1
    acq_func = qExpectedHypervolumeImprovement(model=model, ref_point=ref_point)
    
    print(f"Iteration {iteration+1}: Candidate {new_x.numpy()}  Objectives {new_y.numpy()}")

# =============================================================================
#                    Extract and Plot the Pareto Front
# =============================================================================
def is_pareto_efficient(Y):
    """
    Identify Pareto-efficient points.
    Y: NumPy array of shape (n_points, n_objectives) where all objectives are to be maximized.
    Returns a boolean array of shape (n_points,) indicating whether each point is Pareto-efficient.
    """
    is_efficient = np.ones(Y.shape[0], dtype=bool)
    for i, c in enumerate(Y):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(Y[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

Y_all_np = Y_all.numpy()
pareto_mask = is_pareto_efficient(Y_all_np)
pareto_Y = Y_all_np[pareto_mask]
pareto_X = X_all.numpy()[pareto_mask]

print("Pareto front (objectives):")
print(pareto_Y)

# For plotting, recall our objectives are [speed, -cost_of_transport].
# To plot cost of transport in its natural (positive) form, we flip the sign.
plt.figure(figsize=(8,6))
plt.scatter(Y_all_np[:, 0], -Y_all_np[:, 1], label="All samples")
plt.scatter(pareto_Y[:, 0], -pareto_Y[:, 1], color='red', label="Pareto front")
plt.xlabel("Average Speed (m/s)")
plt.ylabel("Cost of Transport")
plt.legend()
plt.title("Pareto Front: Speed vs. Cost of Transport")
plt.show()
