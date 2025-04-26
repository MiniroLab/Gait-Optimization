import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import torch
import pandas as pd
from botorch.utils.transforms import unnormalize, normalize
from pathlib import Path


# =============================================================================
#                1. MUJOCO & SIMULATION SETUP
# =============================================================================
# MuJoCo model XML file path
model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/Gait-Optimization/turtlev1/xmls/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Actuator names 
actuator_names = [
    "pos_frontleftflipper",
    "pos_frontrightflipper",
    "pos_backleft",
    "pos_backright",
    "pos_frontlefthip",
    "pos_frontrighthip"
]

# Retrieve an actuator's index in the model
def get_actuator_index(model, name):
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if act_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

# Extract joint limits from the XML (used for clamping control signals)
joint_limits = {}
for i, name in enumerate(actuator_names):
    ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
    joint_limits[name] = (ctrl_min, ctrl_max)

# Identify the body whose position represents the center of mass (COM)
main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

# Phase offsets (diagonal phase offsets)
phase_offsets_d = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.5,
    "pos_backleft":          0.5,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.25
}


# Phase offsets (sync phase offsets)
phase_offsets_s = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.0,
    "pos_backleft":          0.0,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.75
}


phase_offsets = phase_offsets_s  # Use sync phase offsets for the simulation    

# Hopf oscillator parameters and integration step function
alpha = 20.0       # Convergence speed
mu = 1.0          # Radius^2

def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
    r_sq = x * x + y * y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    # Coupling term based on phase offsets
    delta = 0.0
    phase_list = [phase_offsets[name] for name in actuator_names]
    for j in range(len(x_all)):
        if j == index:
            continue
        theta_ji = 2.0 * np.pi * (phase_list[index] - phase_list[j])
        delta += y_all[j] * np.cos(theta_ji) - x_all[j] * np.sin(theta_ji)
    dy += coupling * delta

    # Euler integration step
    x_new = x + dx * dt
    y_new = y + dy * dt
    return x_new, y_new

# Mapping from oscillator outputs to joint angles
joint_output_map = {
    "pos_frontleftflipper":  {"offset": 0, "gain": 1},
    "pos_frontrightflipper": {"offset": 0, "gain": 1},
    "pos_backleft":          {"offset": 0, "gain": 1},
    "pos_backright":         {"offset": 0, "gain": 1},
    "pos_frontlefthip":      {"offset": 0, "gain": 1},
    "pos_frontrighthip":     {"offset": 0, "gain": 1}
}


# =============================================================================
#           2. OFFLINE DATA IMPORT & PREPROCESSING FOR BO MODEL
# =============================================================================
# Read the offline simulation data (LHS results) from CSV
csv_path = (
    "C:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/"
    "NnamdiFiles/mujocotest1/assets/Gait-Optimization/data/lhs_simulation_results_sync_set_1.csv"
)
plot_path = Path(
    "C:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/"
    "NnamdiFiles/mujocotest1/figs/optplots_v1"
)

offline_df = pd.read_csv(csv_path)
param_cols = ['stance_freq', 'swing_freq', 'a_param', 'lambda_cpl', 'A_front', 'A_back', 'A_hip']
objective_col = 'Average_Forward_Speed'

# Extract parameter and objective data from the CSV
offline_params_orig = offline_df[param_cols].values  # shape: (N, 7)
offline_objectives = offline_df[[objective_col]].values  # shape: (N, 1)

# =============================================================================
#           3. SINGLE-OBJECTIVE SIMULATION & EVALUATION FUNCTIONS
# =============================================================================
def run_simulation(params, sim_duration=30.0, seed=0, warmup_duration=4.0):
    """
    Runs the simulation with the given parameter vector:
      params = [stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip]
    Includes a warm-up phase (to stabilize the oscillators) before the main simulation.
    Returns the average forward speed (computed from the x-axis displacement).
    """
    np.random.seed(seed)
    mujoco.mj_resetData(model, data)
    data.time = 0.0

    # Unpack parameters
    stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip = params

    # Initialize oscillators for each actuator
    oscillators = {}
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name] = {
            "x": np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn(),
            "y": np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()
        }
    # For mapping control signals (gains etc.)
    amplitude_factors = {
        "pos_frontleftflipper":  A_front,
        "pos_frontrightflipper": A_front,
        "pos_backleft":          A_back,
        "pos_backright":         A_back,
        "pos_frontlefthip":      A_hip,
        "pos_frontrighthip":     A_hip
    }

    dt_cpg = 0.001

    ######################
    # Warm-up Phase
    ######################
    with mujoco.viewer.launch_passive(model, data) as viewer:
        warmup_start = time.time()
        last_loop_time = warmup_start
        print(f"[INFO] Starting warm-up for {warmup_duration:.1f}s...")
        while viewer.is_running():
            now = time.time()
            sim_time = now - warmup_start
            loop_dt = now - last_loop_time
            last_loop_time = now
            if sim_time >= warmup_duration:
                print("[INFO] Warm-up complete.")
                break

            steps = int(np.floor(loop_dt / dt_cpg))
            for _ in range(steps):
                x_all = [oscillators[name]["x"] for name in actuator_names]
                y_all = [oscillators[name]["y"] for name in actuator_names]
                for i, name in enumerate(actuator_names):
                    x_i = oscillators[name]["x"]
                    y_i = oscillators[name]["y"]
                    # Frequency calculation as in your model dynamics
                    freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                           (swing_freq  / (1.0 + np.exp(a_param * y_i)))
                    x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                             lambda_cpl, x_all, y_all, i, phase_offsets)
                    oscillators[name]["x"] = x_new
                    oscillators[name]["y"] = y_new

            mujoco.mj_step(model, data)
            viewer.sync()

    ######################
    # Main Simulation Phase
    ######################
    time_data = []
    com_positions = []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        last_loop_time = start_time
        while viewer.is_running():
            now = time.time()
            sim_time = now - start_time
            if sim_time >= sim_duration:
                break

            loop_dt = now - last_loop_time
            last_loop_time = now
            steps = int(np.floor(loop_dt / dt_cpg))
            for _ in range(steps):
                x_all = [oscillators[name]["x"] for name in actuator_names]
                y_all = [oscillators[name]["y"] for name in actuator_names]
                for i, name in enumerate(actuator_names):
                    x_i = oscillators[name]["x"]
                    y_i = oscillators[name]["y"]
                    freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                           (swing_freq  / (1.0 + np.exp(a_param * y_i)))
                    # Update oscillator states
                    x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                             lambda_cpl, x_all, y_all, i, phase_offsets)
                    oscillators[name]["x"] = x_new
                    oscillators[name]["y"] = y_new

            # Map oscillator outputs to actuator controls
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                off = joint_output_map[name]["offset"]
                gain = joint_output_map[name]["gain"]
                amp_factor = amplitude_factors[name]
                desired_angle = off + gain * amp_factor * np.tanh(oscillators[name]["x"])
                desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
                data.ctrl[actuator_indices[name]] = desired_angle_clamped

            mujoco.mj_step(model, data)
            viewer.sync()

            time_data.append(sim_time)
            com_positions.append(data.xpos[main_body_id].copy())

    # Compute forward speed using x-axis displacement
    if len(com_positions) > 1:
        x_disp = com_positions[-1][0] - com_positions[0][0]
        avg_speed = x_disp / sim_duration
    else:
        avg_speed = 0.0

    return avg_speed

def objective_function(params, orig_bounds):
    """
    Objective function for BO. Unnormalizes the parameter values and runs simulation.
    Returns negative forward speed so that maximizing speed corresponds to minimization.
    """
    # from botorch.utils.transforms import unnormalize
    params_orig = unnormalize(params, orig_bounds)
    params_np = params_orig.detach().numpy()
    n = params_np.shape[0]
    results = []
    for i in range(n):
        speed = run_simulation(params_np[i, :], sim_duration=30.0, seed=0)
        results.append([-speed])  # negative for minimization
    return torch.tensor(results, dtype=torch.double)

# =============================================================================
#         4. PREPARE OFFLINE DATA AS INITIAL TRAINING DATA
# =============================================================================
# Define original parameter bounds:
# Order: [stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip]
orig_bounds = torch.tensor([
    [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],  # lower bounds
    [6.0, 6.0, 20.0, 1.0, 2.0, 2.0, 2.0]   # upper bounds
], dtype=torch.double)

# Normalized bounds: all inputs will reside in [0,1]
norm_bounds = torch.stack([
    torch.zeros(orig_bounds.shape[1], dtype=torch.double),
    torch.ones(orig_bounds.shape[1], dtype=torch.double)
])

# Convert offline parameters and objectives into tensors
offline_params = torch.tensor(offline_params_orig, dtype=torch.double)
# Normalize offline parameters using the original bounds.
from botorch.utils.transforms import normalize
offline_params_norm = normalize(offline_params, orig_bounds)

offline_obj = torch.tensor(offline_objectives, dtype=torch.double)
# Convert forward speed to negative value for minimization.
offline_obj = -offline_obj

# =============================================================================
#           5. BAYESIAN OPTIMIZATION WITH OFFLINE DATA WARM-START
# =============================================================================
def run_bayesian_opt(acq_name, num_iterations=30):
    """
    Run BO to maximize forward speed using offline data as initial training data.
    acq_name: "PI", "LogEI", or "UCB".
    Returns best speed history, parameter history (in original scale), and candidate speeds.
    """
    # Initialize training data using the offline (warm-start) dataset.
    train_x = offline_params_norm.clone()
    train_obj = offline_obj.clone()

    best_speed_history = []
    best_param_history = []
    actual_speed_history = []
    candidate_param_history = []
    
    # Determine current best (note: train_obj stores negative speeds)

    current_speed = -train_obj.min().item()
    best_speed = current_speed
    best_speed_history.append(best_speed)
    best_index = torch.argmin(train_obj)

    best_params = unnormalize(train_x[best_index].unsqueeze(0), orig_bounds).detach().numpy()[0]
    best_param_history.append(best_params)
    actual_speed_history.append(current_speed)
    
    print(f"\n--- {acq_name} Optimization ---")
    print(f"Initial offline data: best params = {best_params}, speed = {current_speed:.4f} m/s")

    candidate_param_history.append(best_params)  # Store initial best parameters


    for i in range(1, num_iterations + 1):
        # Fit a GP model on current training data.
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.optim import optimize_acqf

        model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=1)).double()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Select acquisition function.
        if acq_name == "PI":
            from botorch.acquisition import ProbabilityOfImprovement
            acq_func = ProbabilityOfImprovement(model=model, best_f=train_obj.min(), maximize=False)
        elif acq_name == "LogEI":
            from botorch.acquisition import LogExpectedImprovement
            acq_func = LogExpectedImprovement(model=model, best_f=train_obj.min(), maximize=False)
        elif acq_name == "UCB":
            from botorch.acquisition import UpperConfidenceBound
            acq_func = UpperConfidenceBound(model=model, beta=2.0, maximize=False)
        else:
            raise ValueError(f"Unknown acquisition function: {acq_name}")

        # Optimize acquisition function in the normalized parameter space.
        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=norm_bounds,
            q=1,
            num_restarts=5,
            raw_samples=64
        )


        # candidate_params = unnormalize(candidate[0], orig_bounds).detach().numpy()
        # candidate_param_history.append(candidate_params)

        # Evaluate candidate with simulation.
        new_obj = objective_function(candidate, orig_bounds)
        new_speed = -new_obj[0, 0].item()  # convert back to positive speed

        # Append candidate to training data.
        train_x = torch.cat([train_x, candidate], dim=0)
        train_obj = torch.cat([train_obj, new_obj], dim=0)

        current_params = unnormalize(candidate, orig_bounds).detach().numpy()[0]
        actual_speed_history.append(new_speed)
        if new_speed > best_speed:
            best_speed = new_speed
        best_speed_history.append(best_speed)
        best_index = torch.argmin(train_obj)

        candidate_params = current_params
        candidate_param_history.append(candidate_params)

        best_params = unnormalize(train_x[best_index].unsqueeze(0), orig_bounds).detach().numpy()[0]
        best_param_history.append(best_params)
        
        print(f"Iteration {i}: Candidate params = {current_params}, candidate speed = {new_speed:.4f} m/s, best_speed = {best_speed:.4f} m/s")

    print(f"Final best speed for {acq_name}: {best_speed:.4f} m/s, Params = {best_params}")
    return best_speed_history, best_param_history, actual_speed_history, candidate_param_history

# =============================================================================
#           6. RUN EXPERIMENTS & COMPARE ACQUISITION FUNCTIONS
# =============================================================================
acq_functions = ["PI", "LogEI", "UCB"]
results = {}
num_iterations = 30  # BO iterations for each acquisition function

for acqf in acq_functions:
    print(f"\n===== Running Bayesian Optimization with {acqf} =====")
    best_speed_hist, param_hist, actual_speed_hist, candidate_param_hist = run_bayesian_opt(acqf, num_iterations=num_iterations)
    results[acqf] = {
        "best_speed_history": np.array(best_speed_hist),
        "best_param_history": np.array(param_hist),  # each row is a [7]-dim parameter vector
        "actual_speed_history": np.array(actual_speed_hist),
        "candidate_param_history": np.array(candidate_param_hist)  # each row is a [7]-dim parameter vector

    }

iters = np.arange(num_iterations + 1)

param_names = ["stance_freq", "swing_freq", "a_param", "lambda_cpl", "A_front", "A_back", "A_hip"]

# For each acquisition function, create an iteration index based on the length of one history.
for acq in acq_functions:
    res = results[acq]
    # Determine the number of entries; assuming all histories have the same length.
    n_iter = res["best_speed_history"].shape[0]
    iters = np.arange(n_iter)  # use actual number of entries

    # Create DataFrames with the iteration as a column.
    df_best_speed = pd.DataFrame({
        "Iteration": iters,
        "Best_Speed": res["best_speed_history"]
    })

    df_actual_speed = pd.DataFrame({
        "Iteration": iters,
        "Candidate_Speed": res["actual_speed_history"]
    })

    # For parameter histories, each row is a parameter vector of length len(param_names).
    df_best_params = pd.DataFrame(res["best_param_history"], columns=param_names)
    df_best_params.insert(0, "Iteration", iters)
    
    df_candidate_params = pd.DataFrame(res["candidate_param_history"], columns=param_names)
    df_candidate_params.insert(0, "Iteration", iters)

    # Save the DataFrames as CSV files.
    df_best_speed.to_csv(plot_path/f"{acq}_best_speed_history.csv", index=False)
    df_actual_speed.to_csv(plot_path/f"{acq}_actual_speed_history.csv", index=False)
    df_best_params.to_csv(plot_path/f"{acq}_best_param_history.csv", index=False)
    df_candidate_params.to_csv(plot_path/f"{acq}_candidate_param_history.csv", index=False)

# Plotting evolution of parameters and candidate speeds for each acquisition function.
for acq in acq_functions:
    res = results[acq]
    param_hist = res["candidate_param_history"]
    actual_speed_hist = res["actual_speed_history"]
    dim = param_hist.shape[1]
    nrows = (dim + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3 * nrows))
    axes = axes.flatten()
    for d in range(dim):
        ax = axes[d]
        ax.plot(iters, param_hist[:, d], marker='o', color='tab:blue', label=param_names[d])
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{param_names[d]}", color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.set_title(f"{acq}: {param_names[d]} vs. Iteration")
        ax.grid(True)
        final_val = param_hist[-1, d]
        ax.annotate(f"{final_val:.2f}", xy=(iters[-1], final_val),
                    xytext=(5, 0), textcoords="offset points", color='tab:blue')
        ax2 = ax.twinx()
        ax2.plot(iters, actual_speed_hist, marker='x', linestyle='--', color='tab:red', label="Candidate Speed")
        ax2.set_ylabel("Candidate Speed (m/s)", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.suptitle(f"{acq} Candidate Parameter Evolution")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the figure as both PNG and SVG
    fig.savefig(plot_path/f"{acq}_candidate_parameters_vs_iteration.png", format='png')
    fig.savefig(plot_path/f"{acq}_candidate_parameters_vs_iteration.svg", format='svg')
    plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for acq in acq_functions:
    res = results[acq]
    axs[0].plot(iters, res["best_speed_history"], marker='o', label=acq)
axs[0].set_title("Best Speed vs. Iteration")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Best Speed (m/s)")
axs[0].grid(True)
axs[0].legend()

for acq in acq_functions:
    res = results[acq]
    axs[1].plot(iters, res["actual_speed_history"], marker='o', label=acq)
axs[1].set_title("Candidate Speed vs. Iteration")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Candidate Speed (m/s)")
axs[1].grid(True)
axs[1].legend()

fig.suptitle("Speed History Comparisons")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(plot_path/f"Speed_History_Comparisons.png", format='png')
fig.savefig(plot_path/f"Speed_History_Comparisons.svg", format='svg')
plt.show()

# =============================================================================
#       7. FINAL SIMULATION WITH LOGGING USING BEST PARAMETERS
# =============================================================================
chosen_acq = "LogEI"
res = results[chosen_acq]
final_params = res["best_param_history"][-1]  # best parameters in original scale
print(f"\nBest parameters found by {chosen_acq}: {final_params}")
print(f"Final best speed: {res['best_speed_history'][-1]:.4f} m/s")
def run_simulation_with_logging(params, sim_duration=30.0, seed=42, warmup_duration=4.0):
    """
    Runs the simulation with the given parameter vector:
      params = [stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip]
    Logs time-series data and returns a dictionary of metrics.
    Also logs instantaneous frequency for each joint and prints the average frequency per joint
    and each joint's output range.
    """
    np.random.seed(seed)
    
    # Add CPG state history tracking
    cpg_history_x = {name: [] for name in actuator_names}
    np.random.seed(seed)
    mujoco.mj_resetData(model, data)
    data.time = 0.0

    stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip = params

    # Initialize oscillators for each actuator.
    oscillators = {}
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name] = {
            "x": np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn(),
            "y": np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()
        }

    amplitude_factors = {
        "pos_frontleftflipper":  A_front,
        "pos_frontrightflipper": A_front,
        "pos_backleft":          A_back,
        "pos_backright":         A_back,
        "pos_frontlefthip":      A_hip,
        "pos_frontrighthip":     A_hip
    }

    dt_cpg = 0.001
    time_data = []
    ctrl_data = {name: [] for name in actuator_names}
    power_consumption = []
    com_positions = []
    actuator_torque_history = {name: [] for name in actuator_names}
    joint_velocity_history  = {name: [] for name in actuator_names}

    # Dictionary to log instantaneous frequency for each joint.
    freq_history = {name: [] for name in actuator_names}

    ######################
    # Warm-up Phase
    ######################
    start_time_overall = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        warmup_start = time.time()
        last_loop_time = warmup_start
        print(f"[INFO] Starting warm-up for {warmup_duration:.1f}s...")
        while viewer.is_running():
            now = time.time()
            sim_time = now - warmup_start
            loop_dt = now - last_loop_time
            last_loop_time = now
            if sim_time >= warmup_duration:
                print("[INFO] Warm-up complete. Starting main logging phase.")
                break
            steps = int(np.floor(loop_dt / dt_cpg))
            for _ in range(steps):
                x_all = [oscillators[name]["x"] for name in actuator_names]
                y_all = [oscillators[name]["y"] for name in actuator_names]
                for i, name in enumerate(actuator_names):
                    x_i = oscillators[name]["x"]
                    y_i = oscillators[name]["y"]
                    freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                           (swing_freq  / (1.0 + np.exp(a_param * y_i)))
                    x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                            lambda_cpl, x_all, y_all, i, phase_offsets)
                    oscillators[name]["x"] = x_new
                    oscillators[name]["y"] = y_new
            mujoco.mj_step(model, data)
            viewer.sync()

        ######################
        # Main Simulation Phase
        ######################
        start_time = time.time()
        last_loop_time = start_time
        while viewer.is_running():
            now = time.time()
            sim_time = now - start_time
            if sim_time >= sim_duration:
                print(f"[INFO] Reached {sim_duration:.1f}s of simulation. Stopping.")
                break

            loop_dt = now - last_loop_time
            last_loop_time = now
            steps = int(np.floor(loop_dt / dt_cpg))
            for _ in range(steps):
                x_all = [oscillators[name]["x"] for name in actuator_names]
                y_all = [oscillators[name]["y"] for name in actuator_names]
                for i, name in enumerate(actuator_names):
                    x_i = oscillators[name]["x"]
                    y_i = oscillators[name]["y"]
                    freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                           (swing_freq  / (1.0 + np.exp(a_param * y_i)))
                    freq_history[name].append(freq)
                    x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                            lambda_cpl, x_all, y_all, i, phase_offsets)
                    oscillators[name]["x"] = x_new
                    oscillators[name]["y"] = y_new
                    cpg_history_x[name].append(x_new)


            # Map oscillator outputs to joint controls
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                off = joint_output_map[name]["offset"]
                gain = joint_output_map[name]["gain"]
                amp_factor = amplitude_factors[name]
                desired_angle = off + gain * amp_factor * np.tanh(oscillators[name]["x"])
                desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
                data.ctrl[actuator_indices[name]] = desired_angle_clamped

            mujoco.mj_step(model, data)
            viewer.sync()

            time_data.append(sim_time)
            for name in actuator_names:
                ctrl_data[name].append(data.ctrl[actuator_indices[name]])
            com_positions.append(data.xpos[main_body_id].copy())

            qvel = data.qvel[:model.nu]
            torque = data.actuator_force[:model.nu]
            instant_power = np.sum(np.abs(torque) * np.abs(qvel))
            power_consumption.append(instant_power)

            for i, name in enumerate(actuator_names):
                actuator_torque_history[name].append(data.actuator_force[actuator_indices[name]])
                joint_velocity_history[name].append(qvel[actuator_indices[name]])

    # After the simulation, compute average frequency and control output range per joint.
    avg_freq = {}
    output_range = {}
    for name in actuator_names:
        freq_arr = np.array(freq_history[name])
        avg_freq[name] = np.mean(freq_arr) if freq_arr.size > 0 else 0.0

    return {
        "time_data": time_data,
        "ctrl_data": ctrl_data,
        "power_consumption": power_consumption,
        "com_positions": com_positions,
        "actuator_torque_history": actuator_torque_history,
        "joint_velocity_history": joint_velocity_history,
        "frequency_history": freq_history,
        "avg_frequency": avg_freq,
        "output_range": output_range,
        "cpg_history_x": cpg_history_x
    }

# Run final simulation with logging using the best parameters from LogEI.
log_data = run_simulation_with_logging(final_params, sim_duration=30.0)

time_data = log_data["time_data"]
ctrl_data = log_data["ctrl_data"]
power_consumption = log_data["power_consumption"]
com_positions = np.array(log_data["com_positions"])
actuator_torque_history = log_data["actuator_torque_history"]
joint_velocity_history = log_data["joint_velocity_history"]
cpg_history_x = log_data["cpg_history_x"]

# =============================================================================
#               PLOTTING TIME-SERIES DATA FROM FINAL SIMULATION
# =============================================================================
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# (0,0): Actuator control signals.
for name in actuator_names:
    axs[0, 0].plot(time_data, ctrl_data[name], label=name)
axs[0, 0].set_title("Actuator Control Signals")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Joint Angle (rad)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# (0,1): COM position over time.
if com_positions.size > 0:
    axs[0, 1].plot(time_data, com_positions[:, 0], label="COM X")
    axs[0, 1].plot(time_data, com_positions[:, 1], label="COM Y")
    axs[0, 1].plot(time_data, com_positions[:, 2], label="COM Z")
axs[0, 1].set_title("COM Position vs Time")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Position (m)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# (1,0): Instantaneous power consumption.
axs[1, 0].plot(time_data, power_consumption, label="Instant Power")
axs[1, 0].set_title("Instantaneous Power Consumption")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Power (W)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# (1,1): Trajectory (X vs. Y).
if com_positions.size > 0:
    axs[1, 1].plot(com_positions[:, 0], com_positions[:, 1], 'b-', label="Trajectory")
axs[1, 1].set_title("Trajectory (X vs Y)")
axs[1, 1].set_xlabel("X (m)")
axs[1, 1].set_ylabel("Y (m)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# (2,0): Actuator torque history.
for name in actuator_names:
    axs[2, 0].plot(time_data, actuator_torque_history[name], label=name)
axs[2, 0].set_title("Actuator Torque Over Time")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Torque (Nm)")
axs[2, 0].legend()
axs[2, 0].grid(True)

# (2,1): Joint velocity history.
for name in actuator_names:
    axs[2, 1].plot(time_data, joint_velocity_history[name], label=name)
axs[2, 1].set_title("Joint Velocity Over Time")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Velocity (rad/s)")
axs[2, 1].legend()
axs[2, 1].grid(True)

# (0,1): CPG Oscillator Outputs Over Time
for name in actuator_names:
    axs[2, 1].plot(range(len(cpg_history_x[name])), cpg_history_x[name], label=name)
axs[2, 1].set_title("CPG Oscillator Outputs Over Time")
axs[2, 1].set_xlabel("Time Step")
axs[2, 1].set_ylabel("Oscillator x Output")
axs[2, 1].legend()
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()
