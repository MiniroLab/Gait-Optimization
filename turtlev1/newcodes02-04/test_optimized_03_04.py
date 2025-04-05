import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# =============================================================================
#                           HOPF OSCILLATOR DYNAMICS
# =============================================================================
def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
    """
    Integrate one step of the Hopf oscillator (with coupling) using Euler's method.
    
    Base dynamics:
      dx/dt = α (μ - (x² + y²)) x - ω y
      dy/dt = α (μ - (x² + y²)) y + ω x
      
    Coupling term added to dy:
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
    "assets/turtlev1/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

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
alpha = 10.0       # Convergence speed
mu = 1         # Radius^2
a_param = 10.0     # Logistic steepness for frequency blending

# Define stance and swing frequencies (rad/s)
stance_freq = 2.0
swing_freq  = 2.0

# Coupling constant (λ)
lambda_cpl = 0.5

# Choose phase offsets (diagonal phase offsets in this example)
phase_offsets_d = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.5,
    "pos_backleft":          0.5,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.25
}

# Phase offsets for each actuator
phase_offsets = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.0,
    "pos_backleft":          0.0,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.75
}

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
    offset = joint_output_map[name]["offset"]
    gain =  joint_output_map[name]["gain"]

    desired_angle = offset + gain * (x_val / amplitude)
    desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
    return desired_angle


joint_output_map = {
    "pos_frontleftflipper":  {"offset": 0, "gain": 3.5},
    "pos_frontrightflipper": {"offset": 0, "gain": 3.5},
    "pos_backleft":          {"offset": 0, "gain": 2},
    "pos_backright":         {"offset": 0, "gain": 2},
    "pos_frontlefthip":      {"offset": 0, "gain": 3.5},
    "pos_frontrighthip":     {"offset": 0, "gain": 3.5}
}

# =============================================================================
#             SET UP TIME PARAMETERS AND DATA LOGGING
# =============================================================================
dt_cpg = 0.001         # CPG time step
time_duration = 30.0   # Simulation duration (seconds)

# =============================================================================
#       SIMULATION FUNCTION WITH LOGGING
# =============================================================================
def run_simulation_with_logging(params, sim_duration=30.0, seed=42):
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
                    # Compute instantaneous frequency for this joint
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
#           DICTIONARY OF OPTIMIZED PARAMETERS
# =============================================================================
# These arrays are your best-found parameters for each acquisition function.
best_params = {
    # "PI": np.array ([4.,          4.,         15.,          1.,          4,  4, 4]),

    "PI2": np.array([4.01999459, 1.0, 8.61925497, 0.47604849, 2.88507555, 1.94553649, 1.88408733]),
    "UCB": np.array([3.04999059, 5.0, 15.0, 1.0, 5.0, 5.0, 5.0]),
     "PI": np.array([3.07,	2.59,	18.33,	0.91,	3.28,	2.1, 1.28])

}

# =============================================================================
#           RUN SIMULATION WITH A SELECTED SET OF OPTIMIZED PARAMETERS
# =============================================================================
# To test a given set, simply change acq_to_use to "PI", "LogEI", or "UCB".
acq_to_use = "PI"  # Choose acquisition function to test
params_to_use = best_params[acq_to_use]
print(f"\nRunning simulation using {acq_to_use} optimized parameters:")
print(params_to_use)

# Run simulation (with logging) for a shorter test duration (e.g., 10 s)
test_log_data = run_simulation_with_logging(params_to_use, sim_duration=30.0)

# Compute performance metrics: forward speed (x-direction)
test_com_positions = np.array(test_log_data["com_positions"])
if len(test_com_positions) > 1:
    test_x_disp = test_com_positions[-1][0] - test_com_positions[0][0]
    test_avg_speed = test_x_disp / 30
else:
    test_avg_speed = 0.0

# =============================================================================
#           PRINT PERFORMANCE METRICS
# =============================================================================
print("\n=== Test Performance Metrics ===")
print(f"Coupling strength (λ): {lambda_cpl}")
if len(test_com_positions) > 1:
    displacement = test_com_positions[-1] - test_com_positions[0]
    distance_traveled = np.linalg.norm(displacement)
    displacement_cm = displacement * 100  # Convert displacement to cm
    print(f"Total COM displacement: {displacement} m, {displacement_cm} cm")
    print(f"Straight-line distance traveled: {distance_traveled:.3f} m, {distance_traveled * 100:.2f} cm")
else:
    print("Not enough COM data to compute displacement.")
dt_integration = dt_cpg
if len(test_log_data["power_consumption"]) > 1:
    total_energy = np.sum(test_log_data["power_consumption"]) * dt_integration
    print(f"Approx total energy consumed: {total_energy:.3f} J (assuming torque*velocity)")
    if len(test_com_positions) > 1 and distance_traveled > 0.01:
        weight = total_mass * 9.81
        cost_of_transport = total_energy / (weight * distance_traveled)
        print(f"Approx cost of transport: {cost_of_transport:.3f}")
test_avg_speed_cm_s = test_avg_speed * 100  # Convert m/s to cm/s
print(f"Test Average Forward Speed (x-direction): {test_avg_speed:.3f} m/s, {test_avg_speed_cm_s:.2f} cm/s")

# =============================================================================
#           PLOT PERFORMANCE METRICS FROM TEST SIMULATION
# =============================================================================
test_time_data = test_log_data["time_data"]
test_ctrl_data = test_log_data["ctrl_data"]
test_power = test_log_data["power_consumption"]
test_com = test_com_positions

fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# (0,0): Actuator Control Signals
for name in actuator_names:
    axs[0, 0].plot(test_time_data, test_ctrl_data[name], label=name)
axs[0, 0].set_title("Test: Actuator Control Signals")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Joint Angle (rad)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# (0,1): COM Position vs. Time (X, Y, Z)
if test_com.size > 0:
    axs[0, 1].plot(test_time_data, test_com[:, 0], label="COM X")
    axs[0, 1].plot(test_time_data, test_com[:, 1], label="COM Y")
    axs[0, 1].plot(test_time_data, test_com[:, 2], label="COM Z")
axs[0, 1].set_title("Test: COM Position vs. Time")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Position (m)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# (1,0): Instantaneous Power Consumption vs. Time
axs[1, 0].plot(test_time_data, test_power, label="Instant Power")
axs[1, 0].set_title("Test: Instantaneous Power Consumption")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Power (W)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# (1,1): Trajectory (X vs. Y)
if test_com.size > 0:
    axs[1, 1].plot(test_com[:, 0], test_com[:, 1], 'b-', label="Trajectory")
axs[1, 1].set_title("Test: Trajectory (X vs. Y)")
axs[1, 1].set_xlabel("X (m)")
axs[1, 1].set_ylabel("Y (m)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# (2,0): Actuator Torque Over Time
for name in actuator_names:
    axs[2, 0].plot(test_time_data, test_log_data["actuator_torque_history"][name], label=name)
axs[2, 0].set_title("Test: Actuator Torque Over Time")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Torque (Nm)")
axs[2, 0].legend()
axs[2, 0].grid(True)

# (2,1): Joint Velocity Over Time
for name in actuator_names:
    axs[2, 1].plot(test_time_data, test_log_data["joint_velocity_history"][name], label=name)
axs[2, 1].set_title("Test: Joint Velocity Over Time")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Velocity (rad/s)")
axs[2, 1].legend()
axs[2, 1].grid(True)

fig.tight_layout()
plt.show()
# Save the figure as a PNG file
# fig.savefig("test_simulation_results.png", dpi=300, bbox_inches='tight')    