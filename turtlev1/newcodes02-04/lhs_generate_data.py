import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc
from pathlib import Path


# =============================================================================
#                           HOPF OSCILLATOR DYNAMICS
# =============================================================================
def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
    """
    Integrate one step of the Hopf oscillator (with coupling) using Euler's method.
    
    Base dynamics:
      dx/dt = α (μ - (x² + y²)) x - ω y
      dy/dt = α (μ - (x² + y²)) y + ω x
      
    Coupling term:
      dy += λ * Δ_i, where Δ_i = Σ_{j≠i}[ y_j cos(θ_ji) - x_j sin(θ_ji) ]
      with θ_ji = 2π (φ_i - φ_j), φ's defined in phase_offsets.
    """
    r_sq = x * x + y * y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    delta = 0.0
    phase_list = [phase_offsets[name] for name in actuator_names]
    for j in range(len(x_all)):
        if j == index:
            continue
        theta_ji = 2.0 * np.pi * (phase_list[index] - phase_list[j])
        delta += y_all[j] * np.cos(theta_ji) - x_all[j] * np.sin(theta_ji)
    dy += coupling * delta

    x_new = x + dx * dt
    y_new = y + dy * dt
    return x_new, y_new

# =============================================================================
#                         LOAD MUJOCO MODEL
# =============================================================================
model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/Gait-Optimization/turtlev1/xmls/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)


data_path = Path(
    "C:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/"
    "NnamdiFiles/mujocotest1/assets/Gait-Optimization/data"
)

# =============================================================================
#         SENSOR & ACTUATOR LOOKUP
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
#             DEFINE ACTUATOR NAMES, INDICES, AND JOINT LIMITS
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

joint_limits = {}
for i, name in enumerate(actuator_names):
    ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
    joint_limits[name] = (ctrl_min, ctrl_max)
    print(f"{name}: ctrl range = [{ctrl_min:.3f}, {ctrl_max:.3f}]")

# =============================================================================
#   SELECT BODY FOR COM/ORIENTATION & TOTAL MASS
# =============================================================================
main_body_name = "base"
try:
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)
except Exception as e:
    print("Could not find body named", main_body_name, ":", e)
    main_body_id = 0
base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
base_mass = model.body_mass[base_body_id]
total_mass = np.sum(base_mass)

# =============================================================================
#            CPG PARAMETERS & INITIAL OSCILLATOR STATES
# =============================================================================
alpha = 20.0       # Convergence speed
mu = 1.0          # Radius^2
a_param = 10.0     # Logistic steepness for frequency blending

# Define stance and swing frequencies (rad/s)
stance_freq = 2.0
swing_freq  = 2.0

# Coupling constant (λ)
lambda_cpl = 0.8

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

phase_offsets = phase_offsets_d  # Choose sync or diagonal phase offsets

# Initialize oscillator states
oscillators = {}
for name in actuator_names:
    phase0 = phase_offsets[name] * 2.0 * np.pi
    oscillators[name] = {
        "x": np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn(),
        "y": np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()
    }

# =============================================================================
#            JOINT OUTPUT MAPPING FUNCTION
# =============================================================================
def map_oscillator_to_joint(x_val, min_angle, max_angle):
    amplitude = np.sqrt(mu)
    offset = (min_angle + max_angle) / 2.0
    gain = (max_angle - min_angle) / 2.0
    desired_angle = offset + gain * (x_val / amplitude)
    return desired_angle

joint_output_map = {
    "pos_frontleftflipper":  {"offset": 0, "gain": 1},
    "pos_frontrightflipper": {"offset": 0, "gain": 1},
    "pos_backleft":          {"offset": 0, "gain": 1},
    "pos_backright":         {"offset": 0, "gain": 1},
    "pos_frontlefthip":      {"offset": 0, "gain": 1},
    "pos_frontrighthip":     {"offset": 0, "gain": 1}
}

# =============================================================================
#             SET UP TIME PARAMETERS AND DATA LOGGING
# =============================================================================
dt_cpg = 0.001         # CPG time step
time_duration = 20.0   # Simulation duration (seconds)

# =============================================================================
#       SIMULATION FUNCTION WITH LOGGING
# =============================================================================
def run_simulation_with_logging(params, sim_duration=20.0, seed=42):
    """
    Runs the simulation with the given parameter vector:
      params = [stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip]
    Logs time-series data and returns a dictionary of metrics.
    Also logs instantaneous frequency for each joint and prints the average frequency per joint
    and each joint's output range.
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
    
    # Dictionary to log instantaneous frequency for each joint
    freq_history = {name: [] for name in actuator_names}

    start_time = time.time()

    warmup_duration = 4.0  # seconds to stabilize oscillators

    with mujoco.viewer.launch_passive(model, data) as viewer:
        warmup_start = time.time()
        last_loop_time = warmup_start

        # ------------------ WARM-UP PHASE ------------------
        print(f"[INFO] Starting warm-up for {warmup_duration}s...")
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

        # ------------------ MAIN SIMULATION ------------------
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
            steps = int(np.floor((sim_time - (time_data[-1] if time_data else 0)) / dt_cpg))
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

            # Map oscillator outputs to joint controls
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                off = joint_output_map[name]["offset"]
                gain = joint_output_map[name]["gain"]
                amp_factor = amplitude_factors[name]
                desired_angle = off + amp_factor * np.tanh(oscillators[name]["x"])
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


    # After simulation, print frequency statistics for each joint.
    print("\n=== Frequency Statistics per Joint ===")
    for name in actuator_names:
        if freq_history[name]:
            avg_freq = np.mean(freq_history[name])
            print(f"Joint {name}: Average frequency = {avg_freq:.4f} rad/s")
    
    # Print joint output ranges
    print("\n=== Joint Output Ranges ===")
    for name in actuator_names:
        min_angle, max_angle = joint_limits[name]
        print(f"Joint {name}: Output range = [{min_angle:.3f}, {max_angle:.3f}]")
    
    return {
        "time_data": time_data,
        "ctrl_data": ctrl_data,
        "power_consumption": power_consumption,
        "com_positions": com_positions,
        "actuator_torque_history": actuator_torque_history,
        "joint_velocity_history": joint_velocity_history,
        "freq_history": freq_history
    }


# =============================================================================
#           LHS FOR INITIAL TRAINING DATA
# =============================================================================
# We use Latin Hypercube Sampling (LHS) to generate N experiments in a 7D space.
d = 7  # number of parameters
N = 30 # number of experiments (heuristic: 10x the number of dimensions)

# sampler = qmc.LatinHypercube(d=d)
# lhs_samples = sampler.random(n=N)  # shape (N, 7) in [0,1]^7

# Define the bounds for each parameter:
# [stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip]
bounds = np.array([
    [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],  # lower bounds
    [6.0, 6.0, 20.0, 1.0, 2.0, 2.0, 2.0]   # upper bounds
])

num_datasets = 10   # Generate 10 data sets

# =============================================================================
#           RUN TRIALS OVER LHS SAMPLES AND SAVE RESULTS TO CSV
# =============================================================================

for seed_val in range(1, num_datasets + 1):
    print(f"\n--- Generating data set with LHS seed {seed_val} ---")
    # Create an LHS sampler with current seed.
    sampler = qmc.LatinHypercube(d=d, seed=seed_val)
    lhs_samples = sampler.random(n=N)  # samples in [0,1]^7
    # Scale samples to desired bounds.
    param_samples = qmc.scale(lhs_samples, bounds[0, :], bounds[1, :])
    
    results_list = []
    dt_integration = dt_cpg

    for i in range(N):
        params = param_samples[i]
        print(f"Dataset seed {seed_val}: Running trial {i+1}/{N} with parameters: {params}")
        # Run simulation with the same seed as the dataset seed.
        log_data = run_simulation_with_logging(params, sim_duration=20.0, seed=seed_val)
        
        # Compute metrics:
        # Time Duration
        time_duration_trial = 20.0
        # COM positions:
        com_positions = np.array(log_data["com_positions"])
        if len(com_positions) > 1:
            displacement = com_positions[-1] - com_positions[0]
            straight_line_distance = np.linalg.norm(displacement)
            avg_forward_speed = (com_positions[-1][0] - com_positions[0][0]) / time_duration_trial
            # Lateral deviation in Y (standard deviation)
            lateral_deviation = np.std(com_positions[:, 1])
        else:
            straight_line_distance = 0.0
            avg_forward_speed = 0.0
            lateral_deviation = 0.0
        
        total_energy = np.sum(log_data["power_consumption"]) * dt_integration
        weight = total_mass * 9.81
        if straight_line_distance > 0.01:
            cost_of_transport = total_energy / (weight * straight_line_distance)
        else:
            cost_of_transport = np.nan
        
        total_COM_displacement = straight_line_distance
        
        # Average frequency: average over all joints' instantaneous frequencies
        freq_history = log_data["freq_history"]
        joint_freqs = []
        for name in actuator_names:
            if freq_history[name]:
                joint_freqs.append(np.mean(freq_history[name]))
        if joint_freqs:
            avg_frequency = np.mean(joint_freqs)
        else:
            avg_frequency = 0.0

        results_list.append({
            "Trial": i+1,
            "stance_freq": params[0],
            "swing_freq": params[1],
            "a_param": params[2],
            "lambda_cpl": params[3],
            "A_front": params[4],
            "A_back": params[5],
            "A_hip": params[6],
            "Time_Duration": time_duration_trial,
            "Straight_line_distance": straight_line_distance,
            "Average_Forward_Speed": avg_forward_speed,
            "Average_Lateral_Deviation": lateral_deviation,
            "Total_Energy_Consumed": total_energy,
            "Cost_of_Transport": cost_of_transport,
            "Total_COM_Displacement": total_COM_displacement,
            "Average_Frequency": avg_frequency
        })

    df = pd.DataFrame(results_list)
    file_name = f"lhs_simulation_results_diag_set_{seed_val}.csv"
    df.to_csv(data_path / file_name, index=False)
    print(f"Results saved to {file_name}")
