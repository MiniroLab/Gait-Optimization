import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# PyTorch / BoTorch imports
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import (
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound
)
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize

###############################################################################
#                          1. MUJOCO & SIMULATION SETUP
###############################################################################
# ---------------------------------------------------------------------------
# Adjust these to your own MuJoCo model and actuator names
# ---------------------------------------------------------------------------
model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/turtlev1/testrobot1.xml"
)

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

actuator_names = [
    "pos_frontleftflipper",
    "pos_frontrightflipper",
    "pos_backleft",
    "pos_backright",
    "pos_frontlefthip",
    "pos_frontrighthip"
]
def get_actuator_index(model, name):
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if act_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

# Joint limits from the XML 
joint_limits = {}
for i, name in enumerate(actuator_names):
    ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
    joint_limits[name] = (ctrl_min, ctrl_max)

# Body to track for COM
main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
base_mass = model.body_mass[main_body_id]
total_mass = np.sum(base_mass)

# Phase offsets for each actuator
phase_offsets = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.0,
    "pos_backleft":          0.0,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.75
}

# Hopf oscillator step 
def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
    r_sq = x * x + y * y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    # Coupling term
    delta = 0.0
    phase_list = [phase_offsets[name] for name in actuator_names]
    for j in range(len(x_all)):
        if j == index:
            continue
        theta_ji = 2.0 * np.pi * (phase_list[index] - phase_list[j])
        delta += y_all[j] * np.cos(theta_ji) - x_all[j] * np.sin(theta_ji)
    dy += coupling * delta

    # Euler integration
    x_new = x + dx * dt
    y_new = y + dy * dt
    return x_new, y_new

# Function to map oscillator output to desired joint angle
joint_output_map = {
    "pos_frontleftflipper":  {"offset": 0.435, "gain": 3.0},
    "pos_frontrightflipper": {"offset": 0.435, "gain": 3.0},
    "pos_backleft":          {"offset": 0.435, "gain": 1.5},
    "pos_backright":         {"offset": 0.435, "gain": 1.5},
    "pos_frontlefthip":      {"offset": 0.435, "gain": 3.0},
    "pos_frontrighthip":     {"offset": 0.435, "gain": 3.0}
}

# For Hopf oscillator dynamics
alpha = 10.0
mu = 0.04

###############################################################################
#                2. SINGLE-OBJECTIVE SIMULATION & EVALUATION
###############################################################################
def run_simulation(params, sim_duration=30.0, seed=0):
    """
    Runs a simulation with the given parameter vector:
      params = [stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip]
    Returns the forward speed (m/s) as the main performance metric (only the x-direction).
    """
    np.random.seed(seed)  # ensure consistent initial oscillator states if desired
    mujoco.mj_resetData(model, data)  # reset the environment
    data.time = 0.0

    # Unpack parameters
    stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip = params

    # Initialize oscillator states
    oscillators = {}
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name] = {
            "x": np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn(),
            "y": np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()
        }

    # Amplitude factors per actuator
    amplitude_factors = {
        "pos_frontleftflipper":  A_front,
        "pos_frontrightflipper": A_front,
        "pos_backleft":          A_back,
        "pos_backright":         A_back,
        "pos_frontlefthip":      A_hip,
        "pos_frontrighthip":     A_hip
    }

    dt_cpg = 0.001
    power_record = []
    com_positions = []
    time_log = []

    start_time = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            now = time.time()
            sim_time = now - start_time
            if sim_time >= sim_duration:
                break

            # Determine how many CPG steps to integrate
            steps = int(np.floor((sim_time - (time_log[-1] if time_log else 0)) / dt_cpg))
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

            # Map oscillator outputs to joint controls
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                off = joint_output_map[name]["offset"]
                g   = joint_output_map[name]["gain"]
                amp_factor = amplitude_factors[name]
                # Use tanh of oscillator x-value as an example output
                desired_angle = off + g * amp_factor * np.tanh(oscillators[name]["x"])
                desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
                data.ctrl[actuator_indices[name]] = desired_angle_clamped

            mujoco.mj_step(model, data)
            viewer.sync()

            # Record data at this simulation step
            time_log.append(sim_time)
            com_positions.append(data.xpos[main_body_id].copy())
            # Record instantaneous power consumption
            qvel = data.qvel[:model.nu]
            torque = data.actuator_force[:model.nu]
            instant_power = np.sum(np.abs(torque) * np.abs(qvel))
            power_record.append(instant_power)

    # Compute forward velocity using only the x-direction displacement
    if len(com_positions) > 1:
        x_disp = com_positions[-1][0] - com_positions[0][0]
        avg_speed = x_disp / sim_duration
    else:
        avg_speed = 0.0

    return avg_speed

def objective_function(params, orig_bounds):
    """
    Single-objective function for BoTorch: we want to maximize forward speed.
    Because BoTorch minimizes by default, we return -speed.
    """
    # Unnormalize the parameters from [0,1] to the original bounds
    params_orig = unnormalize(params, orig_bounds)
    params_np = params_orig.detach().numpy()
    n = params_np.shape[0]

    results = []
    for i in range(n):
        speed = run_simulation(params_np[i, :], sim_duration=30.0, seed=0)
        results.append([-speed])  # negative for minimization
    return torch.tensor(results, dtype=torch.double)

###############################################################################
#              3. COMPARISON OF ACQUISITION FUNCTIONS (PI, LogEI, UCB)
###############################################################################
# Parameter bounds:
# params = [stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip]
orig_bounds = torch.tensor([
    [1.0, 1.0, 5.0, 0.1, 0.5, 0.5, 0.5],  # Lower bounds
    [5.0, 5.0, 15.0, 1.0, 5.0, 5.0, 5.0]   # Upper bounds
], dtype=torch.double)

# Normalized bounds (for optimization in [0,1])
norm_bounds = torch.stack([
    torch.zeros(orig_bounds.shape[1], dtype=torch.double),
    torch.ones(orig_bounds.shape[1], dtype=torch.double)
])

# Initial guess in original space
initial_params_orig = torch.tensor([2.0, 2.0, 10.0, 0.5, 1.0, 1.0, 1.0], dtype=torch.double)
initial_params = normalize(initial_params_orig.unsqueeze(0), orig_bounds)  # shape [1,7]

def run_bayesian_opt(acq_name, num_iterations=30):
    """
    Run Bayesian Optimization for forward speed maximization using the specified
    acquisition function (acq_name: "PI", "LogEI", or "UCB").
    At each iteration, the candidate parameters and resulting speed are printed,
    and the best speed so far is tracked.
    
    Returns:
      - best_speed_history: list of best speed (in m/s) so far at each iteration
      - param_history: list of best parameter vectors (in original space)
      - actual_speed_history: list of the candidate speed obtained at each iteration
    """
    train_x = initial_params.clone()
    train_obj = objective_function(train_x, orig_bounds)  # shape [1,1]

    best_speed_history = []
    param_history = []
    actual_speed_history = []

    current_speed = -train_obj[0, 0].item()  # convert back to positive speed
    best_speed = current_speed
    best_speed_history.append(best_speed)
    best_index = torch.argmin(train_obj)  # lower objective is better
    best_params = unnormalize(train_x[best_index], orig_bounds).detach().numpy()
    param_history.append(best_params)
    actual_speed_history.append(current_speed)
    
    print(f"\n--- {acq_name} Optimization ---")
    print(f"Iteration 0: params = {best_params}, speed = {current_speed:.4f} m/s, best_speed = {best_speed:.4f} m/s")

    for i in range(1, num_iterations + 1):
        # Fit a GP model to current data
        model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=1)).double()
        model = model.to(dtype=torch.double)

        # Choose acquisition function based on acq_name
        if acq_name == "PI":
            acq_func = ProbabilityOfImprovement(model=model, best_f=train_obj.min(), maximize=False)
        elif acq_name == "LogEI":
            acq_func = LogExpectedImprovement(model=model, best_f=train_obj.min(), maximize=False)
        elif acq_name == "UCB":
            acq_func = UpperConfidenceBound(model=model, beta=2.0, maximize=False)
        else:
            raise ValueError(f"Unknown acquisition function: {acq_name}")

        # Optimize acquisition function in normalized space
        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=norm_bounds,
            q=1,
            num_restarts=5,
            raw_samples=64
        )

        # Evaluate candidate
        new_obj = objective_function(candidate, orig_bounds)
        new_speed = -new_obj[0, 0].item()  # convert to positive speed

        # Update training data
        train_x = torch.cat([train_x, candidate], dim=0)
        train_obj = torch.cat([train_obj, new_obj], dim=0)

        # Update logs
        current_params = unnormalize(candidate[0], orig_bounds).detach().numpy()
        actual_speed_history.append(new_speed)
        if new_speed > best_speed:
            best_speed = new_speed
        best_speed_history.append(best_speed)
        best_index = torch.argmin(train_obj)
        best_params = unnormalize(train_x[best_index], orig_bounds).detach().numpy()
        param_history.append(best_params)

        print(f"Iteration {i}: Candidate params = {current_params}, candidate speed = {new_speed:.4f} m/s, best_speed = {best_speed:.4f} m/s")

    print(f"Final best speed for {acq_name}: {best_speed:.4f} m/s,  Params = {best_params}")
    return best_speed_history, param_history, actual_speed_history

###############################################################################
#       4. RUN EXPERIMENTS FOR PI, LogEI, AND UCB & PLOT COMPARISON
###############################################################################
acq_functions = ["PI", "LogEI", "UCB"]
results = {}
num_iterations = 30  # Number of iterations for each acquisition function

for acqf in acq_functions:
    print(f"\n===== Running Bayesian Optimization with {acqf} =====")
    best_speed_hist, param_hist, actual_speed_hist = run_bayesian_opt(acqf, num_iterations=num_iterations)
    results[acqf] = {
        "best_speed_history": np.array(best_speed_hist),
        "param_history": np.array(param_hist),  # shape [num_iterations+1, 7]
        "actual_speed_history": np.array(actual_speed_hist)
    }

iters = np.arange(num_iterations + 1)

# A) Plot parameter evolution with candidate speed over iterations for each ACQ
# Define the actual parameter names
param_names = ["stance_freq", "swing_freq", "a_param", "lambda_cpl", "A_front", "A_back", "A_hip"]

for acq in acq_functions:
    res = results[acq]
    param_hist = res["param_history"]  # shape [num_iterations+1, 7]
    actual_speed_hist = res["actual_speed_history"]
    dim = param_hist.shape[1]
    # Create subplots: one for each parameter
    nrows = (dim + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3 * nrows))
    axes = axes.flatten()
    for d in range(dim):
        ax = axes[d]
        ax.plot(iters, param_hist[:, d], marker='o', color='tab:blue', label=param_names[d])
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{param_names[d]} Value", color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.set_title(f"{acq}: {param_names[d]} vs. Iteration")
        ax.grid(True)
        # Optionally annotate the final parameter value on the plot
        final_val = param_hist[-1, d]
        ax.annotate(f"{final_val:.2f}", xy=(iters[-1], final_val),
                    xytext=(5, 0), textcoords="offset points", color='tab:blue')
        # Overlay actual candidate speed on a secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(iters, actual_speed_hist, marker='x', linestyle='--', color='tab:red', label="Candidate Speed")
        ax2.set_ylabel("Candidate Speed (m/s)", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()
    plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Left subplot: Best Speed vs. Iteration (all ACQs on same plot)
for acq in acq_functions:
    res = results[acq]
    axs[0].plot(iters, res["best_speed_history"], marker='o', label=acq)
axs[0].set_title("Best Speed vs. Iteration")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Best Speed (m/s)")
axs[0].grid(True)
axs[0].legend()

# Right subplot: Actual Candidate Speed vs. Iteration (all ACQs on same plot)
for acq in acq_functions:
    res = results[acq]
    axs[1].plot(iters, res["actual_speed_history"], marker='o', label=acq)
axs[1].set_title("Actual Candidate Speed vs. Iteration")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Candidate Speed (m/s)")
axs[1].grid(True)
axs[1].legend()

fig.tight_layout()
plt.show()


###############################################################################
#       5. RUN THE ROBOT WITH THE BEST PARAMETERS & PLOT TIME-BASED DATA
###############################################################################
# Suppose we choose "LogEI" as our preferred acquisition function.
chosen_acq = "LogEI"
res = results[chosen_acq]
final_params = res["param_history"][-1]  # best parameters from final iteration
print(f"\nBest parameters found by {chosen_acq}: {final_params}")
print(f"Final best speed: {res['best_speed_history'][-1]:.4f} m/s")


def run_simulation_with_logging(params, sim_duration=30.0, seed=42):
    """
    Similar to run_simulation, but logs time-series data for plotting.
    Logs:
      - time_data: simulation time stamps
      - ctrl_data: control signals for each actuator
      - power_consumption: instantaneous power consumption over time
      - com_positions: center-of-mass positions over time
      - actuator_torque_history: actuator torque for each actuator over time
      - joint_velocity_history: joint velocities over time
    """
    np.random.seed(seed)
    mujoco.mj_resetData(model, data)
    data.time = 0.0

    stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip = params

    # Initialize oscillators
    oscillators = {}
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name] = {
            "x": np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn(),
            "y": np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()
        }

    # Define amplitude factors for each actuator
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

    start_time = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            now = time.time()
            sim_time = now - start_time
            if sim_time >= sim_duration:
                break

            steps = int(np.floor((sim_time - (time_data[-1] if time_data else 0)) / dt_cpg))
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

            # Apply controls based on oscillator outputs
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                off = joint_output_map[name]["offset"]
                g = joint_output_map[name]["gain"]
                amp_factor = amplitude_factors[name]
                desired_angle = off + g * amp_factor * np.tanh(oscillators[name]["x"])
                desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
                data.ctrl[actuator_indices[name]] = desired_angle_clamped

            mujoco.mj_step(model, data)
            viewer.sync()

            # Log time and control data
            time_data.append(sim_time)
            for name in actuator_names:
                ctrl_data[name].append(data.ctrl[actuator_indices[name]])
            com_positions.append(data.xpos[main_body_id].copy())

            # Log instantaneous power consumption
            qvel = data.qvel[:model.nu]
            torque = data.actuator_force[:model.nu]
            instant_power = np.sum(np.abs(torque) * np.abs(qvel))
            power_consumption.append(instant_power)

            # Log actuator torque and joint velocity per actuator
            for i, name in enumerate(actuator_names):
                actuator_torque_history[name].append(data.actuator_force[actuator_indices[name]])
                joint_velocity_history[name].append(qvel[actuator_indices[name]])

    return {
        "time_data": time_data,
        "ctrl_data": ctrl_data,
        "power_consumption": power_consumption,
        "com_positions": com_positions,
        "actuator_torque_history": actuator_torque_history,
        "joint_velocity_history": joint_velocity_history
    }

# Run final simulation with logging using the best parameters found from LogEI
log_data = run_simulation_with_logging(final_params, sim_duration=30.0)

time_data = log_data["time_data"]
ctrl_data = log_data["ctrl_data"]
power_consumption = log_data["power_consumption"]
com_positions = np.array(log_data["com_positions"])
actuator_torque_history = log_data["actuator_torque_history"]
joint_velocity_history = log_data["joint_velocity_history"]

# Plotting time-series data in a 3x2 grid
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# (0,0) : Actuator Control Signals
for name in actuator_names:
    axs[0, 0].plot(time_data, ctrl_data[name], label=name)
axs[0, 0].set_title("Actuator Control Signals")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Joint Angle (rad)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# (0,1) : COM Position vs Time
if com_positions.size > 0:
    axs[0, 1].plot(time_data, com_positions[:, 0], label="COM X")
    axs[0, 1].plot(time_data, com_positions[:, 1], label="COM Y")
    axs[0, 1].plot(time_data, com_positions[:, 2], label="COM Z")
axs[0, 1].set_title("COM Position vs Time")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Position (m)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# (1,0) : Instantaneous Power Consumption vs Time
axs[1, 0].plot(time_data, power_consumption, label="Instant Power")
axs[1, 0].set_title("Instantaneous Power Consumption")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Power (W)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# (1,1) : Trajectory (X vs Y)
if com_positions.size > 0:
    axs[1, 1].plot(com_positions[:, 0], com_positions[:, 1], 'b-', label="Trajectory")
axs[1, 1].set_title("Trajectory (X vs Y)")
axs[1, 1].set_xlabel("X (m)")
axs[1, 1].set_ylabel("Y (m)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# (2,0) : Actuator Torque Over Time
for name in actuator_names:
    axs[2, 0].plot(time_data, actuator_torque_history[name], label=name)
axs[2, 0].set_title("Actuator Torque Over Time")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Torque (Nm)")
axs[2, 0].legend()
axs[2, 0].grid(True)

# (2,1) : Joint Velocity Over Time
for name in actuator_names:
    axs[2, 1].plot(time_data, joint_velocity_history[name], label=name)
axs[2, 1].set_title("Joint Velocity Over Time")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Velocity (rad/s)")
axs[2, 1].legend()
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()
