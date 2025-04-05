import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# BoTorch and PyTorch imports
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement)
from botorch.acquisition.multi_objective import logei
from botorch.optim import optimize_acqf
from botorch.models import transforms
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.sampling.normal import SobolQMCNormalSampler

# =============================================================================
#                               HOPF OSCILLATOR DYNAMICS
# =============================================================================
def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
    """
    Integrate one step of the Hopf oscillator (with coupling) using Euler's method.

    The base dynamics:
      dx/dt = α (μ - (x² + y²)) x - ω y
      dy/dt = α (μ - (x² + y²)) y + ω x

    The coupling term is added to dy:
      dy += λ * Δ_i,
    where
      Δ_i = Σ_{j≠i} [ y_j cos(θ_{ji}) - x_j sin(θ_{ji}) ],
      and θ_{ji} = 2π (φ_i - φ_j) with φ's defined in phase_offsets.
    """
    r_sq = x * x + y * y
    # Base Hopf oscillator dynamics (limit cycle)
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    # Compute coupling term using the desired phase offsets:
    delta = 0.0
    # Construct a list of phase offsets corresponding to the order in actuator_names
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

# =============================================================================
#                               LOAD MUJOCO MODEL
# =============================================================================
model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/turtlev1/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# =============================================================================
#                           SENSOR & ACTUATOR LOOKUP
# =============================================================================
sensor_name2id = {}
for i in range(model.nsensor):
    name_adr = model.sensor_adr[i]
    name_chars = []
    for c in model.names[name_adr:]:
        if c == 0:
            break
        name_chars.append(chr(c))
    sensor_name = "".join(name_chars)
    sensor_name2id[sensor_name] = i

def get_sensor_data(data, model, sensor_name2id, sname):
    if sname not in sensor_name2id:
        return None
    sid = sensor_name2id[sname]
    dim = model.sensor_dim[sid]
    start_idx = model.sensor_adr[sid]
    return data.sensordata[start_idx : start_idx + dim].copy()

def get_actuator_index(model, name):
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

# =============================================================================
#               DEFINE ACTUATOR NAMES, INDICES, AND JOINT LIMITS
# =============================================================================
actuator_names = [
    "pos_frontleftflipper",
    "pos_frontrightflipper",
    "pos_backleft",
    "pos_backright",
    "pos_frontlefthip",
    "pos_frontrighthip"
]
actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

# For this example, we use the actuator control range from the model:
joint_limits = {}
for i, name in enumerate(actuator_names):
    ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
    joint_limits[name] = (ctrl_min, ctrl_max)
    # print(f"{name}: ctrl range = [{ctrl_min:.3f}, {ctrl_max:.3f}]")

# =============================================================================
#           SELECT BODY FOR COM/ORIENTATION & TOTAL MASS
# =============================================================================
main_body_name = "base"  # Adjust as needed
try:
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)
except Exception as e:
    print("Could not find body named", main_body_name, ":", e)
    main_body_id = 0

base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
base_mass = model.body_mass[base_body_id]
total_mass = np.sum(base_mass)

alpha = 10.0       # Convergence speed
mu = 0.04          # Radius^2 (amplitude ~ sqrt(0.04)=0.2)

# =============================================================================
#                       JOINT OUTPUT MAPPING FUNCTION
# =============================================================================
def map_oscillator_to_joint(x_val, min_angle, max_angle):
    """
    Linearly map x_val from [-sqrt(mu), sqrt(mu)] to [min_angle, max_angle] using:
      offset = (min_angle + max_angle)/2,
      gain   = (max_angle - min_angle)/2.
    """
    amplitude = np.sqrt(mu)
    offset = (min_angle + max_angle) / 2.0
    gain = (max_angle - min_angle) / 2.0
    desired_angle = offset + gain * (x_val / amplitude)
    return desired_angle

# Optionally, you can define a joint_output_map dictionary if you want custom gains/offsets:
joint_output_map = {
    "pos_frontleftflipper":  {"offset": 0.435, "gain": 3},
    "pos_frontrightflipper": {"offset": 0.435, "gain": 3},
    "pos_backleft":          {"offset": 0.435, "gain": 1.5},
    "pos_backright":         {"offset": 0.435, "gain": 1.5},
    "pos_frontlefthip":      {"offset": 0.435, "gain": 3},
    "pos_frontrighthip":     {"offset": 0.435, "gain": 3}
}

def matrix_to_euler(mat):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)  # Convert matrix to quaternion
    r = R.from_quat(quat)  # Convert quaternion to Rotation object
    euler_angles = r.as_euler('xyz', degrees=False)  # Convert to Euler angles
    return euler_angles

# =============================================================================
#                           SIMULATION FUNCTION
# =============================================================================
def run_simulation(params, phase_offsets):
    """
    Runs a single simulation episode with the given parameters.  This function
    encapsulates the core simulation logic, making it suitable for use within
    an optimization loop.

    Args:
        params (list): A list of CPG parameters:
            [stance_freq, swing_freq, a_param, lambda_cpl, A_frontleftflipper,
             A_frontrightflipper, A_backleft, A_backright, A_frontlefthip, A_frontrighthip]
        phase_offsets (dict): Dictionary defining the phase offsets.

    Returns:
        tuple: (vx, dy, CoT, stability, sim_data)  A tuple containing the
               performance metrics and simulation data.
    """
    # Unpack parameters
    (stance_freq, swing_freq, a_param, lambda_cpl,
     A_frontflipper, A_back, A_fronthip) = params

    # Amplitude scaling for each actuator
    amplitude_factors = {
        "pos_frontleftflipper":  A_frontflipper,
        "pos_frontrightflipper": A_frontflipper,
        "pos_backleft":          A_back,
        "pos_backright":         A_back,
        "pos_frontlefthip":      A_fronthip,
        "pos_frontrighthip":     A_fronthip
    }

    # Initialize oscillator states
    oscillators = {}
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name] = {
            "x": np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn(),
            "y": np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()
        }

    # Set up time parameters
    dt_cpg = 0.001
    time_duration = 30.0

    # Data logging
    time_data = []
    ctrl_data = {name: [] for name in actuator_names}
    cpg_outputs = {name: [] for name in actuator_names}
    com_positions = []
    body_orientations = []
    power_consumption = []

    sensor_data_history = {
        "torque_backright": [],
        "torque_backleft": [],
        "torque_frontrighthip": [],
        "torque_frontrightflipper": [],
        "torque_frontlefthip": [],
        "torque_frontleftflipper": [],
        "base_acc": [],
        "base_gyro": []
    }
    jointact_sensor_map = {
        "torque_backright":           "sens_jointactfrc_backright",
        "torque_backleft":            "sens_jointactfrc_backleft",
        "torque_frontrighthip":     "sens_jointactfrc_frontrighthip",
        "torque_frontrightflipper": "sens_jointactfrc_frontrightflipper",
        "torque_frontlefthip":      "sens_jointactfrc_frontlefthip",
        "torque_frontleftflipper":  "sens_jointactfrc_frontleftflipper"
    }
    base_imu_map = {
        "base_gyro": "sens_base_gyro",
        "base_acc":   "sens_base_acc"
    }

    actuator_torque_history = {name: [] for name in actuator_names}
    joint_velocity_history  = {name: [] for name in actuator_names}

    # Simulation loop using the mujoco API
    start_time = time.time()
    sim_time = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and sim_time < time_duration:
            now = time.time()
            sim_time = now - start_time
            loop_dt = sim_time - (time_data[-1] if time_data else 0.0)

            # Integrate Hopf oscillators
            steps = int(np.floor(loop_dt / dt_cpg))
            for _ in range(steps):
                x_all = [oscillators[name]["x"] for name in actuator_names]
                y_all = [oscillators[name]["y"] for name in actuator_names]
                for i, name in enumerate(actuator_names):
                    x_i = oscillators[name]["x"]
                    y_i = oscillators[name]["y"]
                    # Check for NaN values in oscillator states
                    if np.isnan(x_i) or np.isnan(y_i):
                        return 0.0, 0.0, 1e6, {}  # Penalize if unstable
                    # Compute instantaneous frequency
                    freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                           (swing_freq  / (1.0 + np.exp(a_param * y_i)))
                    x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                            lambda_cpl, x_all, y_all, i, phase_offsets)
                    oscillators[name]["x"] = x_new
                    oscillators[name]["y"] = y_new

            # Map oscillator outputs to joint controls
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                amplitude_factor = amplitude_factors[name]  # Get amplitude for this actuator.
                offset = joint_output_map[name]["offset"]
                gain = joint_output_map[name]["gain"]
                desired_angle = offset + gain * np.tanh(oscillators[name]["x"])
                desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
                data.ctrl[actuator_indices[name]] = desired_angle_clamped
                cpg_outputs[name].append(desired_angle_clamped)

            time_data.append(sim_time)

            # Step the simulation
            mujoco.mj_step(model, data)

            # Record data
            for name in actuator_names:
                ctrl_data[name].append(data.ctrl[actuator_indices[name]])
            com_positions.append(data.xpos[main_body_id].copy())
            body_orientations.append(data.xmat[main_body_id].copy())
            qvel = data.qvel[:model.nu]
            torque = data.actuator_force[:model.nu]
            instant_power = np.sum(np.abs(torque) * np.abs(qvel))
            power_consumption.append(instant_power)

            for varname, sname in jointact_sensor_map.items():
                val = get_sensor_data(data, model, sensor_name2id, sname)
                if val is not None:
                    sensor_data_history[varname].append(val[0])
            for varname, sname in base_imu_map.items():
                val = get_sensor_data(data, model, sensor_name2id, sname)
                if val is not None:
                    sensor_data_history[varname].append(val.copy())
            for name in actuator_names:
                idx = actuator_indices[name]
                actuator_torque_history[name].append(data.actuator_force[idx])
            for name in actuator_names:
                idx = actuator_indices[name]
                joint_velocity_history[name].append(qvel[idx])
            viewer.sync()

    # --------------------- Performance Metrics -----------------------
    final_time = time_data[-1] if len(time_data) > 0 else 0.0
    displacement = com_positions[-1] - com_positions[0] if len(com_positions) > 1 else np.zeros(3)
    distance_traveled = np.linalg.norm(displacement)
    avg_velocity = distance_traveled / final_time if final_time > 0 else 0.0
    dt_integration = dt_cpg  # Use the CPG timestep for integration
    total_energy = np.sum(power_consumption) * dt_integration if power_consumption else 0.0
    weight = total_mass * 9.81
    cost_of_transport = total_energy / (weight * distance_traveled) if distance_traveled > 0.01 else 1e6  # Avoid division by zero

    # Calculate lateral displacement (dy)
    y_positions = [com[1] for com in com_positions]  # Extract Y coordinates
    y_mean = np.mean(y_positions)
    dy = np.sqrt(np.mean((np.array(y_positions) - y_mean)**2)) if y_positions else 0.0

    # Calculate stability (S)
    roll_angles = []
    pitch_angles = []
    for mat in body_orientations:
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        # r = mujoco.Mjr(data.qpos)  # Create a Mjr object
        # mujoco.mjr_mat2euler(roll_angles, pitch_angles, None, mat)
        roll_angles, pitch_angles, _ = matrix_to_euler(mat)
    if roll_angles and pitch_angles:
        roll_var   = np.var(roll_angles)
        pitch_var  = np.var(pitch_angles)
        stability  = -(roll_var + pitch_var)
        if stability == 0:
            stability += 1e-6
    else:
        stability = 0.0
        stability += 1e-6  # Small positive value to avoid division by zero
    # if roll_angles and pitch_angles:
    #   roll_var   = np.var(roll_angles)
    #   pitch_var  = np.var(pitch_angles)
    #   stability  = -(roll_var + pitch_var)
    # else:
    #   stability = 0.0

    sim_data = {
        "time_data": time_data,
        "ctrl_data": ctrl_data,
        "cpg_outputs": cpg_outputs,
        "com_positions": com_positions,
        "body_orientations": body_orientations,
        "power_consumption": power_consumption,
        "sensor_data_history": sensor_data_history,
        "actuator_torque_history": actuator_torque_history,
        "joint_velocity_history": joint_velocity_history,
    }
    return avg_velocity, dy, cost_of_transport, sim_data


# def normalize(x, orig_bounds):
#     """
#     Normalize x (tensor of shape [..., d]) to [0, 1] given orig_bounds.
#     orig_bounds is a 2 x d tensor where:
#       orig_bounds[0] = lower bounds,
#       orig_bounds[1] = upper bounds.
#     """
#     lb = orig_bounds[0]
#     ub = orig_bounds[1]
#     return (x - lb) / (ub - lb)

# def unnormalize(x_norm, orig_bounds):
#     """
#     Reverse the normalization: map x_norm in [0, 1] back to original space.
#     """
#     lb = orig_bounds[0]
#     ub = orig_bounds[1]
#     return x_norm * (ub - lb) + lb


# =============================================================================
#                           OBJECTIVE FUNCTION
# =============================================================================
def evaluate_objective(params, phase_offsets, orig_bounds):
    """
    Evaluate the objectives for a given set of design variables.  This
    function is a wrapper around run_simulation to make it compatible with
    BoTorch.

    Args:
        params (torch.Tensor): A 2D tensor where each row represents a solution
            and each column a design variable.
        phase_offsets (dict): The phase offsets.

    Returns:
        torch.Tensor: A 2D tensor of objective values.
    """
    params_orig = unnormalize(params, orig_bounds)
    params_np = params_orig.detach().numpy()  # Convert to numpy for use with run_simulation
    num_points = params_np.shape[0]
    objectives = np.zeros((num_points, 3))  # 3 objectives: vx, dy, CoT

    for i, p in enumerate(params_np):
        vx, dy, CoT, _ = run_simulation(p, phase_offsets)

        # Log candidate and objective
        print(f"Candidate {i}: params = {p}, vx = {vx}, dy = {dy}, CoT = {CoT}")

        obj0 = -vx  # For velocity, we minimize negative velocity
        obj1 = dy
        obj2 = CoT

        # Replace NaNs/infs with a penalty (e.g., a large finite number)
        if np.isnan(obj0) or np.isinf(obj0):
            obj0 = 1e6
        if np.isnan(obj1) or np.isinf(obj1):
            obj1 = 1e6
        if np.isnan(obj2) or np.isinf(obj2):
            obj2 = 1e6

        objectives[i, 0] = obj0
        objectives[i, 1] = obj1
        objectives[i, 2] = obj2

    return torch.tensor(objectives, dtype=torch.double)

# =============================================================================
#                           PARAMETER BOUNDS
# =============================================================================
# # Define the parameter bounds as a PyTorch tensor
# bounds = torch.tensor([
#     [1.0, 1.0, 5.0, 0.1, 0.5, 0.5, 0.5],  # Lower bounds
#     [3.0, 3.0, 15.0, 1.0, 5.0, 5.0, 5.0]   # Upper bounds
# ], dtype=torch.double)

# Original bounds (for example, using 7 parameters)
orig_bounds = torch.tensor([
    [1.0, 1.0, 5.0, 0.1, 0.5, 0.5, 0.5],  # Lower bounds
    [5.0, 5.0, 15.0, 1.0, 5.0, 5.0, 5.0]   # Upper bounds
], dtype=torch.double)

# Normalized bounds: each dimension becomes [0, 1]
norm_bounds = torch.tensor([
    [0.0] * orig_bounds.shape[1],
    [1.0] * orig_bounds.shape[1]
], dtype=torch.double)


# =============================================================================
#                           INITIAL GUESS
# =============================================================================
# Provide an initial guess for the parameters.
initial_params_orig = torch.tensor([
    2.0,    # stance_freq
    2.0,    # swing_freq
    10.0,   # a_param
    0.5,    # lambda_cpl
    1.0,    # A_frontflipper
    1.0,    # A_back
    1.0,    # A_fronthip
], dtype=torch.double).unsqueeze(0)  # Make it a 2D tensor

# Normalize the initial parameters
initial_params = normalize(initial_params_orig, orig_bounds)

# =============================================================================
#                           PHASE OFFSETS
# =============================================================================
# Define the phase offsets - crucial for coordinated movement
phase_offsets_db = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.5,  # 180 degrees out of phase with left
    "pos_backleft":          0.25, # 90 degrees
    "pos_backright":         0.75, # 270 degrees
    "pos_frontlefthip":      0.0,
    "pos_frontrighthip":     0.5
}

phase_offsets = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.5,
    "pos_backleft":          0.5,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.25
}
# =============================================================================
#                           OPTIMIZATION
# =============================================================================
def main(num_iterations=3):
    """
    Main optimization loop using BoTorch.
    """
    # Initialize training data with the initial guess
    train_x = initial_params
    train_obj = evaluate_objective(train_x, phase_offsets, orig_bounds)
    train_x = train_x.double()
    train_obj = train_obj.double()

    # Define the Pareto front approximation
    pareto_front = train_obj
    hv_history = []

    # Lists to track the best objective values at each iteration.
    best_velocity_history = []  # in m/s
    avg_velocity_history = []

    best_dy_history = []        # lateral displacement (m)
    avg_dy_history = []

    best_cot_history = []       # cost of transport
    avg_cot_history = []


    # Initialize model
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=3))

    # Define the acquisition function
    acq_func = logei.qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        X_baseline=train_x,  # Baseline for the acquisition function
        ref_point=torch.tensor([-0.01, 0.15, 20.0], dtype=torch.double), # Reference point for HV (IMPORTANT:  Sign matches objectives)
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])), # Increased sampler shape
    )

    # Optimization loop
    for i in range(num_iterations):
        print(f"Iteration {i + 1}/{num_iterations}")

        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=norm_bounds,
            q=1,  # Number of candidates to generate per iteration
            num_restarts=20, # Increased num_restarts
            raw_samples=512, # Increased raw_samples
        )

        # Evaluate candidates
        new_obj = evaluate_objective(candidates, phase_offsets, orig_bounds)

        # Update training data
        train_x = torch.cat([train_x, candidates], dim=0)
        train_obj = torch.cat([train_obj, new_obj], dim=0)

        # Update Pareto front approximation
        is_non_dominated_mask = is_non_dominated(train_obj)
        pareto_front = train_obj[is_non_dominated_mask]

        # Calculate Hypervolume
        hv = Hypervolume(ref_point=torch.tensor([-0.01, 0.15, 20.0], dtype=torch.double)) # Consistent ref_point
        hv_value = hv.compute(pareto_front)
        hv_history.append(hv_value)
        print(f"Hypervolume: {hv_value:.4f}")

        # Record best objective values from the current Pareto front.
        # For velocity, our objective is -vx, so best velocity = -min(objective[0])
        best_velocity = -pareto_front[:, 0].min().item()  # actual velocity in m/s
        avg_velocity  = -pareto_front[:, 0].mean().item()

        best_dy = pareto_front[:, 1].min().item()          # best (lowest) lateral displacement
        avg_dy  = pareto_front[:, 1].mean().item()

        best_cot = pareto_front[:, 2].min().item()         # best (lowest) cost of transport
        avg_cot  = pareto_front[:, 2].mean().item()


        best_velocity_history.append(best_velocity)
        avg_velocity_history.append(avg_velocity)

        best_dy_history.append(best_dy)
        avg_dy_history.append(avg_dy)

        best_cot_history.append(best_cot)
        avg_cot_history.append(avg_cot)


        # Update model
        model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=3))
        acq_func.model = model  # Update the model in the acquisition function

    print("Optimization finished!")
    return train_x, train_obj, pareto_front, hv_history, best_velocity_history, best_dy_history, best_cot_history, avg_velocity_history, avg_dy_history, avg_cot_history

if __name__ == "__main__":
    (train_x, train_obj, pareto_front, hv_history,
     best_velocity_history, best_dy_history, best_cot_history, avg_velocity_history, avg_dy_history, avg_cot_history) = main(num_iterations=3)

    # =============================================================================
    #                           PARETO FRONT PLOTTING
    # =============================================================================
    # # Plot the Pareto front
    pareto_front_np = pareto_front.detach().numpy()

    iterations = list(range(1, len(best_velocity_history) + 1))



    # Create a single figure with 3 subplots in one row
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # --- Subplot 1: Velocity vs. Lateral Displacement ---
    axes[0].scatter(-pareto_front_np[:, 0], pareto_front_np[:, 1], c="blue", s=30)
    axes[0].set_xlabel("Velocity (m/s)")
    axes[0].set_ylabel("Lateral Displacement (m)")
    axes[0].set_title("Pareto Front:\nVelocity vs. Displacement")
    axes[0].grid(True)

    # --- Subplot 2: Velocity vs. Cost of Transport ---
    axes[1].scatter(-pareto_front_np[:, 0], pareto_front_np[:, 2], c="green", s=30)
    axes[1].set_xlabel("Velocity (m/s)")
    axes[1].set_ylabel("Cost of Transport")
    axes[1].set_title("Pareto Front:\nVelocity vs. CoT")
    axes[1].grid(True)

    # --- Subplot 3: Hypervolume over Iterations ---
    axes[2].plot(hv_history, marker='o')
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Hypervolume")
    axes[2].set_title("Hypervolume over Iterations")
    axes[2].grid(True)

    # Adjust layout so titles/labels don't overlap
    plt.tight_layout()
    plt.show()


    # plt.figure(figsize=(8, 6))
    # plt.scatter(-pareto_front_np[:, 0], pareto_front_np[:, 1], c="blue", s=30)
    # plt.xlabel("Velocity (m/s)")
    # plt.ylabel("Lateral Displacement (m)")
    # plt.title("Pareto Front")
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.scatter(-pareto_front_np[:, 0], pareto_front_np[:, 2], c="green", s=30)
    # plt.xlabel("Velocity (m/s)")
    # plt.ylabel("Cost of Transport")
    # plt.title("Pareto Front")
    # plt.grid(True)
    # plt.show()

    # # Plot Hypervolume
    # plt.figure(figsize=(8, 6))
    # plt.plot(hv_history, marker='o')
    # plt.xlabel("Iteration")
    # plt.ylabel("Hypervolume")
    # plt.title("Hypervolume over Iterations")
    # plt.grid(True)
    # plt.show()


    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # --- Subplot 1: Velocity over Iterations ---
    axes[0].plot(iterations, best_velocity_history, label="Best", marker='o', color="black")
    axes[0].plot(iterations, avg_velocity_history, label="Average", color="black", linestyle="--")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Velocity over Iterations")
    axes[0].grid(True)
    axes[0].legend()

    # --- Subplot 2: Lateral Displacement over Iterations ---
    axes[1].plot(iterations, best_dy_history, label="Best", marker='o', color="blue")
    axes[1].plot(iterations, avg_dy_history, label="Average", color="blue", linestyle="--")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Lateral Displacement (m)")
    axes[1].set_title("Lateral Displacement over Iterations")
    axes[1].grid(True)
    axes[1].legend()

    # --- Subplot 3: Cost of Transport over Iterations ---
    axes[2].plot(iterations, best_cot_history, label="Best", marker='o', color="green")
    axes[2].plot(iterations, avg_cot_history, label="Average", color="green", linestyle="--")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Cost of Transport")
    axes[2].set_title("Cost of Transport over Iterations")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # # =============================================================================
    # #                PLOTTING CHANGE IN EACH OBJECTIVE OVER ITERATIONS
    # # =============================================================================

    # plt.figure(figsize=(8, 6))
    # plt.plot(iterations, best_velocity_history, label="Best", marker='o', color="black")
    # plt.plot(iterations, avg_velocity_history, label="Average", color="black", linestyle="--")
    # plt.xlabel("Iteration")
    # plt.ylabel("Velocity (m/s)")
    # plt.title("Velocity over Iterations")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.plot(iterations, best_dy_history, label="Best", marker='o', color="blue")
    # plt.plot(iterations, avg_dy_history, label="Average", color="blue", linestyle="--")
    # plt.xlabel("Iteration")
    # plt.ylabel("Lateral Displacement (m)")
    # plt.title("Lateral Displacement over Iterations")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.plot(iterations, best_cot_history, label="Minimum", marker='o', color="green")
    # plt.plot(iterations, avg_cot_history, label="Average", color="green", linestyle="--")
    # plt.xlabel("Iteration")
    # plt.ylabel("Cost of Transport")
    # plt.title("Cost of Transport over Iterations")
    # plt.grid(True)
    # plt.legend()
    # plt.show()




    # Print the optimal parameters from the Pareto front
    print("Optimal Solutions (Pareto Front):")
    for i, x in enumerate(pareto_front):
        print(f"Solution {i + 1}:")
        params = train_x[is_non_dominated(train_obj)][i].tolist()
        print(f"  Stance Freq: {params[0]:.3f}, Swing Freq: {params[1]:.3f}, a_param: {params[2]:.3f}, lambda: {params[3]:.3f}")
        print(f"  A_frontleftflipper: {params[4]:.3f}, A_frontrightflipper: {params[4]:.3f}, A_backleft: {params[5]:.3f}, A_backright: {params[5]:.3f}")
        print(f"  A_frontlefthip: {params[6]:.3f}, A_frontrighthip: {params[6]:.3f}")
        print(f"  Velocity: {-pareto_front[i, 0]:.4f} m/s, Cost of Transport: {pareto_front[i, 2]:.4f}, Lateral Displacement: {pareto_front[i, 1]:.4f} m")
