import time
import botorch.fit
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

import pandas as pd
import torch
import botorch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from botorch.acquisition import (
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound
)
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize


# ----------------------------
# Part 1: Read CSV Data & Prepare Training Data
# ----------------------------

# Read the CSV file generated from your LHS simulation trials.
csv_path = (
    "C:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/"
    "NnamdiFiles/mujocotest1/assets/Gait-Optimization/data/lhs_simulation_results_sync.csv"
)
df = pd.read_csv(csv_path)
print("CSV columns:", df.columns.tolist())

# Clean NaNs or bad rows (optional safety)
df = df.dropna(subset=["Average_Forward_Speed"]) # critical column
df = df.reset_index(drop=True)

# Extract design parameters and the objective.
# Parameters: stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip
# Objective of interest: Average_Forward_Speed (to be maximized)
# We convert the objective to negative values for minimization.

# Parameter and objective columns
param_cols = ['stance_freq', 'swing_freq', 'a_param', 'lambda_cpl', 'A_front', 'A_back', 'A_hip']
objective_col = 'Average_Forward_Speed'


# Extract input and output data
X_raw = df[param_cols].values
Y_raw = df[[objective_col]].values
Y_raw = -Y_raw  # Negate for minimization


## Normalize inputs to [0, 1] using MinMaxScaler
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)

# Convert to torch tensors
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

# Convert bounds to numpy array first, then to tensor
bounds_array = np.vstack([
    scaler_X.data_min_,
    scaler_X.data_max_
])
bounds_tensor = torch.from_numpy(bounds_array, dtype=torch.double)

# Print basic info
print(f"Training set shape: X = {train_x.shape}, Y = {train_y.shape}")
print(f"Objective stats (original): min = {df[objective_col].min():.3f}, max = {df[objective_col].max():.3f}")
print("First 3 training samples:\n", train_x[:3])
print("First 3 objectives (negative speed):\n", train_y[:3])


# ----------------------------
# Part 2: Build and Fit the Gaussian Process Model
# ----------------------------

# Create a GP model with a standardization transform on outcomes.
model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1)).to(dtype=torch.double)

# Set model to training mode
model.train()

# Create the marginal log likelihood (MLL) object.
mll = ExactMarginalLogLikelihood(model.likelihood, model)

# Fit the GP model using maximum likelihood estimation.
fit_gpytorch_mll(mll)

print("\nGP model fitted.")
print("Model hyperparameters:")
print(f"Lengthscale: {model.covar_module.base_kernel.lengthscale.item():.4f}")
print(f"Noise: {model.likelihood.noise.item():.4f}")
print(f"Log Marginal Likelihood: {mll(model(train_x), train_y).item():.4f}")
print("Model parameters:")
print(model.state_dict())
print("Model likelihood parameters:")
print(model.likelihood.state_dict())


# # ----------------------------
# # Part 3: Evaluate the Quality of the GP Model
# # ----------------------------

# # Set the model to evaluation mode:
# model.eval()

# # Compute GP predictions on the training data (using no-grad context)
# with torch.no_grad():
#     posterior = model(train_x)
#     # Extract the predictive mean and standard deviation
#     mean_pred = posterior.mean.squeeze().numpy()
#     std_pred = posterior.variance.sqrt().squeeze().numpy()

# # Convert true values from tensor to numpy
# y_true = train_y.squeeze().numpy()

# # Calculate RMSE and R² metrics on training data
# rmse = np.sqrt(mean_squared_error(y_true, mean_pred))
# r2 = r2_score(y_true, mean_pred)

# print("\n=== GP Model Quality Metrics ===")
# print(f"Training RMSE: {rmse:.4f}")
# print(f"Training R²: {r2:.4f}")
# # Note: A very negative R² (like -20168.85) indicates a very poor fit.


# # Also print the log marginal likelihood (should be higher for a better fit)
# # Compute the log marginal likelihood properly by passing the target as well.
# with torch.no_grad():
#     # Get predictions as a GPPosterior object
#     posterior = model(train_x)
#     lml = ExactMarginalLogLikelihood(model.likelihood, model)(posterior, train_y)
#     # If 'lml' contains more than one element, sum them up:
#     lml_value = lml.sum().item()

# print("Log Marginal Likelihood: {:.4f}".format(lml_value))

# # Optional: Plotting predicted vs. true values for visual inspection
# plt.figure(figsize=(8, 6))
# plt.errorbar(range(len(y_true)), mean_pred, yerr=2*std_pred, fmt='o', 
#              label='GP predictions (±2σ)', capsize=5)
# plt.plot(range(len(y_true)), y_true, 'rx', markersize=8, label='True values')
# plt.xlabel('Training sample index')
# plt.ylabel('Negative Average Forward Speed')
# plt.title('GP Predictions vs. True Values')
# plt.legend()
# plt.grid(True)
# plt.show()


# # ----------------------------
# # Part 4: (Optional) Use the GP for Bayesian Optimization
# # ----------------------------
# # For now, we focus on the single-objective optimization of maximum forward speed.
# # The GP model can serve as a surrogate function within an optimization loop.
# # (You would use acquisition functions like EI, PI, or UCB with optimize_acqf to propose new candidate points.)


# # =============================================================================
# #                         LOAD MUJOCO MODEL
# # =============================================================================
# model_path = (
#     "c:/Users/chike/Box/TurtleRobotExperiments/"
#     "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
#     "assets/Gait-Optimization/turtlev1/xmls/testrobot1.xml"
# )
# model = mujoco.MjModel.from_xml_path(model_path)
# data = mujoco.MjData(model)

# # =============================================================================
# #         SENSOR & ACTUATOR LOOKUP
# # =============================================================================
# sensor_name2id = {}
# for i in range(model.nsensor):
#     name_adr = model.sensor_adr[i]
#     name_chars = []
#     for c in model.names[name_adr:]:
#         if c == 0:
#             break
#         name_chars.append(chr(c))
#     sensor_name = "".join(name_chars)
#     sensor_name2id[sensor_name] = i

# def get_sensor_data(data, model, sensor_name2id, sname):
#     if sname not in sensor_name2id:
#         return None
#     sid = sensor_name2id[sname]
#     dim = model.sensor_dim[sid]
#     start_idx = model.sensor_adr[sid]
#     return data.sensordata[start_idx : start_idx + dim].copy()

# def get_actuator_index(model, name):
#     for i in range(model.nu):
#         actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
#         if actuator_name == name:
#             return i
#     raise ValueError(f"Actuator '{name}' not found in model.")

# # =============================================================================
# #             DEFINE ACTUATOR NAMES, INDICES, AND JOINT LIMITS
# # =============================================================================
# actuator_names = [
#     "pos_frontleftflipper",
#     "pos_frontrightflipper",
#     "pos_backleft",
#     "pos_backright",
#     "pos_frontlefthip",
#     "pos_frontrighthip"
# ]
# actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

# joint_limits = {}
# for i, name in enumerate(actuator_names):
#     ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
#     joint_limits[name] = (ctrl_min, ctrl_max)
#     print(f"{name}: ctrl range = [{ctrl_min:.3f}, {ctrl_max:.3f}]")

# # =============================================================================
# #   SELECT BODY FOR COM/ORIENTATION & TOTAL MASS
# # =============================================================================
# main_body_name = "base"
# try:
#     main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)
# except Exception as e:
#     print("Could not find body named", main_body_name, ":", e)
#     main_body_id = 0
# base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
# base_mass = model.body_mass[base_body_id]
# total_mass = np.sum(base_mass)


# # Phase offsets (diagonal phase offsets)
# phase_offsets_d = {
#     "pos_frontleftflipper":  0.0,
#     "pos_frontrightflipper": 0.5,
#     "pos_backleft":          0.5,
#     "pos_backright":         0.0,
#     "pos_frontlefthip":      0.75,
#     "pos_frontrighthip":     0.25
# }

# # Phase offsets (sync phase offsets)
# phase_offsets_s = {
#     "pos_frontleftflipper":  0.0,
#     "pos_frontrightflipper": 0.0,
#     "pos_backleft":          0.0,
#     "pos_backright":         0.0,
#     "pos_frontlefthip":      0.75,
#     "pos_frontrighthip":     0.75
# }

# phase_offsets = phase_offsets_s  # Choose sync or diagonal phase offsets

# def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
#     """
#     Integrate one step of the Hopf oscillator (with coupling) using Euler's method.
    
#     Base dynamics:
#       dx/dt = α (μ - (x² + y²)) x - ω y
#       dy/dt = α (μ - (x² + y²)) y + ω x
      
#     Coupling term:
#       dy += λ * Δ_i, where Δ_i = Σ_{j≠i}[ y_j cos(θ_ji) - x_j sin(θ_ji) ]
#       with θ_ji = 2π (φ_i - φ_j), φ's defined in phase_offsets.
#     """
#     r_sq = x * x + y * y
#     dx = alpha * (mu - r_sq) * x - omega * y
#     dy = alpha * (mu - r_sq) * y + omega * x

#     delta = 0.0
#     phase_list = [phase_offsets[name] for name in actuator_names]
#     for j in range(len(x_all)):
#         if j == index:
#             continue
#         theta_ji = 2.0 * np.pi * (phase_list[index] - phase_list[j])
#         delta += y_all[j] * np.cos(theta_ji) - x_all[j] * np.sin(theta_ji)
#     dy += coupling * delta

#     x_new = x + dx * dt
#     y_new = y + dy * dt
#     return x_new, y_new

# # Map oscillator output to joint control command
# joint_output_map = {
#     "pos_frontleftflipper":  {"offset": 0, "gain": 1},
#     "pos_frontrightflipper": {"offset": 0, "gain": 1},
#     "pos_backleft":          {"offset": 0, "gain": 1},
#     "pos_backright":         {"offset": 0, "gain": 1},
#     "pos_frontlefthip":      {"offset": 0, "gain": 1},
#     "pos_frontrighthip":     {"offset": 0, "gain": 1}
# }

# # Hopf oscillator parameters
# alpha = 10.0
# mu = 1.0

# ##############################################################################
# #                          4A. Simulation Wrapper
# ###############################################################################
# def run_simulation(params, sim_duration=30.0, seed=123):
#     """
#     Evaluates forward speed by running the MuJoCo simulation for sim_duration seconds
#     with the given parameter vector (unpacked).
#     Returns average forward speed in m/s.
#     """
#     np.random.seed(seed)
#     mujoco.mj_resetData(model, data)
#     data.time = 0.0

#     # Unpack parameters
#     stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip = params

#     # Initialize oscillator states
#     oscillators = {}
#     for name in actuator_names:
#         phase0 = phase_offsets[name]*2.0*np.pi
#         oscillators[name] = {
#             "x": np.sqrt(mu)*np.cos(phase0) + 0.002*np.random.randn(),
#             "y": np.sqrt(mu)*np.sin(phase0) + 0.002*np.random.randn()
#         }

#     amplitude_factors = {
#         "pos_frontleftflipper": A_front,
#         "pos_frontrightflipper": A_front,
#         "pos_backleft": A_back,
#         "pos_backright": A_back,
#         "pos_frontlefthip": A_hip,
#         "pos_frontrighthip": A_hip
#     }

#     dt_cpg = 0.001
#     start_time_sim = time.time()
#     times_log = []

#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         while viewer.is_running():
#             now = time.time()
#             sim_time = now - start_time_sim
#             if sim_time >= sim_duration:
#                 break

#             steps = 1  # e.g. we can do finer time stepping if desired
#             for _ in range(steps):
#                 x_all = [oscillators[name]["x"] for name in actuator_names]
#                 y_all = [oscillators[name]["y"] for name in actuator_names]
#                 for i, name in enumerate(actuator_names):
#                     x_i = oscillators[name]["x"]
#                     y_i = oscillators[name]["y"]
#                     freq = (stance_freq/(1.0 + np.exp(-a_param*y_i))) + \
#                            (swing_freq/(1.0 + np.exp(a_param*y_i)))
#                     x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg, lambda_cpl, x_all, y_all, i)
#                     oscillators[name]["x"] = x_new
#                     oscillators[name]["y"] = y_new

#             # Map oscillator outputs to joint commands
#             for name in actuator_names:
#                 offset = joint_output_map[name]["offset"]
#                 gain = joint_output_map[name]["gain"]
#                 min_angle, max_angle = joint_limits[name]
#                 desired_angle = offset + gain*amplitude_factors[name]*np.tanh(oscillators[name]["x"])
#                 data.ctrl[actuator_indices[name]] = np.clip(desired_angle, min_angle, max_angle)

#             mujoco.mj_step(model, data)
#             viewer.sync()
#             times_log.append(sim_time)

#     # Compute average forward speed in x-direction
#     # We have access to data.xpos[main_body_id], so let's do final - initial over sim_duration
#     # But we didn't log every step, so let's just do the final position - the initial position.
#     # data.xpos is the position of each body: data.xpos[body_id, 0] = x, 1 = y, 2 = z
#     # Since we didn't store them, let's do:
#     # Actually we can do the final and initial once the loop ends:
#     # But we must be careful: the final data.xpos might reflect the last step in the viewer sync

#     # Let's do a simpler approach: store com positions at start and end
#     # For demonstration, we just do:
#     x_disp = data.xpos[main_body_id][0]  # final
#     # The initial was near 0. So let's approximate average speed
#     avg_speed = x_disp / sim_duration
#     return avg_speed

# ###############################################################################
# #                          4B. Objective Function
# ###############################################################################
# def objective_function(X_norm, bounds_tensor):
#     """
#     Single-objective function for BoTorch. Minimizing negative speed.
#     X_norm: shape (N, D), normalized in [0,1].
#     bounds_tensor: shape (2, D), storing original parameter min & max.
#     """
#     # Convert to original scale
#     X_unnorm = unnormalize(X_norm, bounds_tensor)
#     X_np = X_unnorm.detach().numpy()

#     results = []
#     for i in range(X_np.shape[0]):
#         speed = run_simulation(X_np[i, :], sim_duration=20.0, seed=123)  # shorter test sim
#         results.append([-speed])  # negative -> minimization
#     return torch.tensor(results, dtype=torch.double)


# # Normalized bounds for optimization in [0,1] (same dimension as train_x)
# norm_bounds = torch.stack([
#     torch.zeros(train_x.shape[1], dtype=torch.double),
#     torch.ones(train_x.shape[1], dtype=torch.double)
# ])

# # Number of iterations for Bayesian optimization
# num_iterations = 20

# # Initialize history logs
# best_speed_history = []
# actual_speed_history = []
# param_history = []

# # Clone LHS data for further appending
# bo_train_x = train_x.clone()
# bo_train_y = train_y.clone()


# # Track the best values so far
# best_speed = -bo_train_y.min().item()  # Convert from negative to positive speed
# best_speed_history.append(best_speed)
# best_idx = torch.argmin(bo_train_y)
# best_params = unnormalize(bo_train_x[best_idx], bounds_tensor).detach().numpy()
# param_history.append(best_params)
# actual_speed_history.append(best_speed)
# print(f"Starting BO with {train_x.shape[0]} LHS points. Initial best speed: {best_speed:.4f} m/s")
# print(f"Best speed = {best_speed:.4f} m/s, best_params = {best_params}")


# # Choose acquisition function type: "LogEI", "PI", or "UCB"
# acq_name = "LogEI"  # Change here to switch acquisition function

# for iteration in range(1, num_iterations + 1):
#     # Refit GP model on current data
#     model_bo = SingleTaskGP(bo_train_x, bo_train_y, outcome_transform=Standardize(m=1)).double()
#     mll_bo = ExactMarginalLogLikelihood(model_bo.likelihood, model_bo)
#     fit_gpytorch_mll(mll_bo)
#     model_bo.eval()

#     # Define acquisition function
#     if acq_name == "LogEI":
#         acq_func = LogExpectedImprovement(model=model_bo, best_f=bo_train_y.min(), maximize=False)
#     elif acq_name == "PI":
#         acq_func = ProbabilityOfImprovement(model=model_bo, best_f=bo_train_y.min(), maximize=False)
#     elif acq_name == "UCB":
#         acq_func = UpperConfidenceBound(model=model_bo, beta=2.0, maximize=False)
#     else:
#         raise ValueError(f"Unknown acquisition function: {acq_name}")


#     # Optimize acquisition function
#     candidate, _ = optimize_acqf(
#         acq_function=acq_func,
#         bounds=norm_bounds,
#         q=1,
#         num_restarts=5,
#         raw_samples=64
#     )

#     # Evaluate candidate with actual simulation
#     new_obj = objective_function(candidate, bounds_tensor)
#     new_speed = -new_obj.item()  # Convert to positive speed

#     # Append to training data
#     bo_train_x = torch.cat([bo_train_x, candidate], dim=0)
#     bo_train_y = torch.cat([bo_train_y, new_obj.view(1, 1)], dim=0)

#     # Unnormalize for logging
#     new_params = unnormalize(candidate, bounds_tensor).detach().numpy().squeeze()
#     param_history.append(new_params)
#     actual_speed_history.append(new_speed)

#     if new_speed > best_speed:
#             best_speed = new_speed
#     best_speed_history.append(best_speed)

#     print(f"Iteration {iteration}: Speed = {new_speed:.4f} m/s, Best = {best_speed:.4f} m/s")

# # Convert history lists to arrays for further analysis or plotting
# param_history = np.array(param_history)
# actual_speed_history = np.array(actual_speed_history)
# best_speed_history = np.array(best_speed_history)

# print("\nFinal best parameters:", param_history[-1])
# print("Final best speed:", best_speed_history[-1])
















# # ----------------------------
# # Part 5: Integration with Warm-Up in Simulation (Overview)
# # ----------------------------
# # In your simulation code (see your provided simulation loop), you can incorporate a warm-up phase
# # before logging data. For example, before starting to log, run the simulation (e.g., for 2 seconds)
# # without storing data. Then, reset the timers and logging arrays before the main simulation phase.
# #
# # This code is part of your simulation function, which you already have; you would integrate the warm-up
# # loop as shown in previous examples.

# # ----------------------------
# # Part 6: Save and Print Metrics
# # ----------------------------
# # After running your simulation using new candidates from the GP model, the performance metrics (such as:
# # - Time Duration,
# # - Straight-line Distance Traveled,
# # - Average Forward Speed (x-direction),
# # - Average Lateral Deviation in Y Direction,
# # - Total Energy Consumed,
# # - Cost of Transport,
# # - Total COM Displacement,
# # - Average Frequency per joint,
# #
# # ) would be computed and saved to a CSV file.
# #
# # For example, assuming you have collected a results_list (list of dictionaries), you could save as follows:

# # results_list = [...]  # List containing dictionaries for each trial
# # results_df = pd.DataFrame(results_list)
# # results_df.to_csv("final_optimization_results.csv", index=False)
# # print("Final results saved to final_optimization_results.csv")
