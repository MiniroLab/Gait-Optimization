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
      dx/dt = α (μ - (x²+y²)) x - ω y
      dy/dt = α (μ - (x²+y²)) y + ω x
      
    Coupling term (added to dy):
      dy += λ * Δ_i, where
      Δ_i = Σ_{j≠i} [ y_j cos(θ_{ji}) - x_j sin(θ_{ji}) ],
      with θ_{ji} = 2π (φ_i - φ_j) and φ_i defined in phase_offsets.
    """
    r_sq = x * x + y * y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    # Compute coupling term using desired phase offsets:
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
#   SELECT BODY FOR COM/ORIENTATION & TOTAL MASS (base only)
# =============================================================================
main_body_name = "base"
try:
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)
except Exception as e:
    print("Could not find body named", main_body_name, ":", e)
    main_body_id = 0

base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
base_mass = model.body_mass[base_body_id]
total_mass = base_mass  # Only the base's mass

# =============================================================================
#            CPG PARAMETERS & INITIAL OSCILLATOR STATES
# =============================================================================
alpha = 10.0       # Convergence speed
mu = 0.04          # Radius^2 (amplitude ~ sqrt(0.04)=0.2)
a_param = 10.0     # Logistic steepness for frequency blending

# Frequency parameters will be varied. (Stance and swing set equal here.)
lambda_cpl = 0.5   # Coupling constant

# Define phase offsets (as fractions; 1 corresponds to 2π)
# Using diagonal gait as an example:
phase_offsets = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.5,
    "pos_backleft":          0.5,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.25
}

# Initialize oscillator states using phase offsets.
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
    """
    Map x_val from [-sqrt(mu), sqrt(mu)] to [min_angle, max_angle] using:
      offset = (min_angle+max_angle)/2,
      gain   = (max_angle-min_angle)/2.
    """
    amplitude = np.sqrt(mu)
    offset = (min_angle + max_angle) / 2.0
    gain = (max_angle - min_angle) / 2.0
    desired_angle = offset + gain * (x_val / amplitude)
    return desired_angle

# Custom mapping using joint_output_map (optional)
joint_output_map = {
    "pos_frontleftflipper":  {"offset": 0.435, "gain": 3},
    "pos_frontrightflipper": {"offset": 0.435, "gain": 3},
    "pos_backleft":          {"offset": 0.435, "gain": 1.5},
    "pos_backright":         {"offset": 0.435, "gain": 1.5},
    "pos_frontlefthip":      {"offset": 0.435, "gain": 3},
    "pos_frontrighthip":     {"offset": 0.435, "gain": 3}
}

# =============================================================================
#             SIMULATION & DATA LOGGING PARAMETERS
# =============================================================================
dt_cpg = 0.001         # Integration time step
sim_duration = 30.0    # Simulation duration per run (seconds)

# Data logging containers (for each simulation run):
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
    "torque_backright":       "sens_jointactfrc_backright",
    "torque_backleft":        "sens_jointactfrc_backleft",
    "torque_frontrighthip":   "sens_jointactfrc_frontrighthip",
    "torque_frontrightflipper":"sens_jointactfrc_frontrightflipper",
    "torque_frontlefthip":    "sens_jointactfrc_frontlefthip",
    "torque_frontleftflipper":"sens_jointactfrc_frontleftflipper"
}
base_imu_map = {
    "base_gyro": "sens_base_gyro",
    "base_acc":  "sens_base_acc"
}

actuator_torque_history = {name: [] for name in actuator_names}
joint_velocity_history  = {name: [] for name in actuator_names}

# =============================================================================
#       SIMULATION FUNCTION: Run simulation at a given frequency (Hz)
# =============================================================================
def run_simulation(freq_hz):
    """
    Run simulation for a given frequency (Hz) applied to both stance and swing.
    Returns:
      - avg_speed: Average forward speed (x-displacement / sim_duration)
      - avg_freq: Average computed local frequency (rad/s) across all integration steps and joints.
    """

    # Reset simulation state to initial conditions.
    mujoco.mj_resetData(model, data)
    data.time = 0
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    
    # Convert frequency (Hz) to rad/s:
    freq_rad = freq_hz * 2.0 * np.pi
    stance_freq_local = freq_rad
    swing_freq_local  = freq_rad

    # Reset oscillator states to initial values using fixed phase offsets:
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name]["x"] = np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn()
        oscillators[name]["y"] = np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()
    
    # Local logging containers:
    sim_time = 0.0
    time_data_local = []
    com_positions_local = []
    freq_record = []  # Record computed local frequency values

    start_time = time.time()
    last_loop_time = start_time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            now = time.time()
            sim_time = now - start_time
            loop_dt = now - last_loop_time
            last_loop_time = now

            if sim_time >= sim_duration:
                break

            steps = int(np.floor(loop_dt / dt_cpg))
            for _ in range(steps):
                x_all = [oscillators[name]["x"] for name in actuator_names]
                y_all = [oscillators[name]["y"] for name in actuator_names]
                for i, name in enumerate(actuator_names):
                    x_i = oscillators[name]["x"]
                    y_i = oscillators[name]["y"]
                    # Compute local frequency via logistic blending
                    freq_local = (stance_freq_local / (1.0 + np.exp(-a_param * y_i))) + \
                                 (swing_freq_local  / (1.0 + np.exp(a_param * y_i)))
                    freq_record.append(freq_local)
                    x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq_local, dt_cpg,
                                             lambda_cpl, x_all, y_all, i, phase_offsets)
                    oscillators[name]["x"] = x_new
                    oscillators[name]["y"] = y_new

            # Map oscillator outputs to joint commands:
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                offset = joint_output_map[name]["offset"]
                gain = joint_output_map[name]["gain"]
                desired_angle = offset + gain * oscillators[name]["x"]
                desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
                data.ctrl[actuator_indices[name]] = desired_angle_clamped

            mujoco.mj_step(model, data)
            time_data_local.append(sim_time)
            com_positions_local.append(data.xpos[main_body_id].copy())
            viewer.sync()

    # Calculate average forward speed (x-displacement / duration)
    if len(com_positions_local) >= 2:
        displacement = com_positions_local[-1] - com_positions_local[0]
        avg_speed = displacement[0] / sim_duration
    else:
        avg_speed = 0.0

    # Calculate average computed local frequency (mean of all recorded values)
    avg_freq = np.mean(freq_record) if len(freq_record) > 0 else 0.0

    return avg_speed, avg_freq

# =============================================================================
#                   RUN SIMULATIONS OVER FREQUENCY INCREMENTS
# =============================================================================
# Define 10 frequency values in Hz (e.g., from 0.5 Hz to 5.0 Hz)
freq_values_Hz = np.linspace(0, 1.2, 15)
freq_values_rad_s = freq_values_Hz * 2.0 * np.pi  # Convert to rad/s
avg_speeds = []
avg_freqs_rad_s = []
avg_freqs_Hz = []

for freq in freq_values_Hz:
    print(f"Running simulation at frequency {freq:.2f} Hz...")
    avg_speed, avg_freq = run_simulation(freq)
    print(f"At {freq:.2f} Hz, {((freq * 2.0 * np.pi)):.2f} rad/s: Avg Forward Speed = {avg_speed:.2f} m/s, Avg Computed Frequency = {avg_freq:.2f} rad/s, {((avg_freq)/(2.0 * np.pi)):.2f} Hz")
    avg_speeds.append(avg_speed)
    avg_freqs_rad_s.append(avg_freq)
    avg_freqs_Hz.append((avg_freq)/(2.0 * np.pi))

# Save data to a text file: columns: Input Frequency, Avg Forward Speed, Avg Computed Frequency.
data_to_save = np.column_stack((freq_values_rad_s, freq_values_Hz, avg_speeds, avg_freqs_rad_s, avg_freqs_Hz))
np.savetxt("frequency_vs_velocity3.txt", data_to_save,
           header="Frequency_rad/s   Frequency_Hz   Avg_Forward_Speed_m/s   Avg_Computed_Frequency_rad/s   Avg_Computed_Frequency_Hz",)


# =============================================================================
#                              PLOT RESULTS WITH SUBPLOTS
# =============================================================================
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# Subplot (0,0): Forward Speed vs. Input Frequency (Hz)
axs[0, 0].plot(freq_values_Hz, avg_speeds, marker='o', linestyle='-')
axs[0, 0].set_title("Forward Speed vs. Input Frequency (Hz)")
axs[0, 0].set_xlabel("Input Frequency (Hz)")
axs[0, 0].set_ylabel("Avg Forward Speed (m/s)")
axs[0, 0].grid(True)

# Subplot (0,1): Forward Speed vs. Input Frequency (rad/s)
axs[0, 1].plot(freq_values_rad_s, avg_speeds, marker='o', linestyle='-')
axs[0, 1].set_title("Forward Speed vs. Input Frequency (rad/s)")
axs[0, 1].set_xlabel("Input Frequency (rad/s)")
axs[0, 1].set_ylabel("Avg Forward Speed (m/s)")
axs[0, 1].grid(True)

# Subplot (1,0): Avg Computed Local Frequency (Hz) vs. Forward Speed
axs[1, 0].plot(avg_freqs_Hz, avg_speeds, marker='^', linestyle='-.', color='g')
axs[1, 0].set_title("Avg Computed Local Frequency (Hz) vs. Forward Speed")
axs[1, 0].set_xlabel("Avg Computed Frequency (Hz)")
axs[1, 0].set_ylabel("Avg Forward Speed (m/s)")
axs[1, 0].grid(True)

# Subplot (1,1): Avg Computed Local Frequency (rad/s) vs. Forward Speed
axs[1, 1].plot(avg_freqs_rad_s, avg_speeds, marker='^', linestyle='-.', color='g')
axs[1, 1].set_title("Avg Computed Local Frequency (rad/s) vs. Forward Speed")
axs[1, 1].set_xlabel("Avg Computed Frequency (rad/s)")
axs[1, 1].set_ylabel("Avg Forward Speed (m/s)")
axs[1, 1].grid(True)

# Subplot (2,0): Avg Computed Local Frequency vs. Input Frequency (Hz)
axs[2, 0].plot(freq_values_Hz, avg_freqs_Hz, marker='s', linestyle='--', color='r')
axs[2, 0].set_title("Avg Computed Frequency vs. Input Frequency (Hz)")
axs[2, 0].set_xlabel("Input Frequency (Hz)")
axs[2, 0].set_ylabel("Avg Computed Frequency (Hz)")
axs[2, 0].grid(True)

# Subplot (2,1): Avg Computed Local Frequency vs. Input Frequency (rad/s)
axs[2, 1].plot(freq_values_rad_s, avg_freqs_rad_s, marker='s', linestyle='--', color='r')
axs[2, 1].set_title("Avg Computed Frequency vs. Input Frequency (rad/s)")
axs[2, 1].set_xlabel("Input Frequency (rad/s)")
axs[2, 1].set_ylabel("Avg Computed Frequency (rad/s)")
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()

# # =============================================================================
# #                              PLOT RESULTS
# # =============================================================================
# plt.figure(figsize=(8, 6))
# plt.plot(freq_values_Hz, avg_speeds, marker='o', linestyle='-')
# plt.title("Forward Speed vs. Input Frequency (Hz)")
# plt.xlabel("Input Frequency (Hz)")
# plt.ylabel("Average Forward Speed (m/s)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(freq_values_rad_s, avg_speeds, marker='o', linestyle='-')
# plt.title("Forward Speed vs. Input Frequency (Hz)")
# plt.xlabel("Input Frequency (rad/s)")
# plt.ylabel("Average Forward Speed (m/s)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(avg_freqs_Hz, avg_speeds, marker='^', linestyle='-.', color='g')
# plt.title("Avg Computed Local Frequency (Hz) vs. Forward Speed")
# plt.xlabel("Avg Computed Frequency (Hz)")
# plt.ylabel("Average Forward Speed (m/s)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(avg_freqs_rad_s, avg_speeds, marker='^', linestyle='-.', color='g')
# plt.title("Avg Computed Local Frequency (rad/s) vs. Forward Speed")
# plt.xlabel("Avg Computed Frequency (rad/s)")
# plt.ylabel("Average Forward Speed (m/s)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(freq_values_Hz, avg_freqs_Hz, marker='s', linestyle='--', color='r')
# plt.title("Average Computed Local Frequency vs. Input Frequency (Hz)")
# plt.xlabel("Input Frequency (Hz)")
# plt.ylabel("Avg Computed Frequency (Hz)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(freq_values_rad_s, avg_freqs_rad_s, marker='s', linestyle='--', color='r')
# plt.title("Average Computed Local Frequency vs. Input Frequency (rad/s)")
# plt.xlabel("Input Frequency (rad/s)")
# plt.ylabel("Avg Computed Frequency (rad/s)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
