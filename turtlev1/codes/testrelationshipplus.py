import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl

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
      with θ_{ji} = 2π (φ_i - φ_j) and φ defined in phase_offsets.
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

def quaternion_to_yaw(q):
    """
    Convert a quaternion (assumed in [w, x, y, z] order) to a yaw angle in radians.
    """
    w, x, y, z = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

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
total_mass = base_mass  # Use only the base's mass

# =============================================================================
#            CPG PARAMETERS & INITIAL OSCILLATOR STATES
# =============================================================================
alpha = 10.0       # Convergence speed
mu = 0.04          # Radius^2 (amplitude ~ sqrt(0.04)=0.2)
a_param = 10.0     # Logistic steepness for frequency blending

# Frequency parameters will be varied (both stance and swing set equal).
lambda_cpl = 0.5   # Coupling constant

# Define phase offsets (as fractions; 1 corresponds to 2π)
phase_offsets = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.5,
    "pos_backleft":          0.5,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.25
}

# Initialize oscillator states for each joint based on phase offsets.
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
    Linearly map x_val from [-sqrt(mu), sqrt(mu)] to [min_angle, max_angle] using:
      offset = (min_angle+max_angle)/2,
      gain   = (max_angle-min_angle)/2.
    """
    amplitude = np.sqrt(mu)
    offset = (min_angle + max_angle) / 2.0
    gain = (max_angle - min_angle) / 2.0
    desired_angle = offset + gain * (x_val / amplitude)
    return desired_angle

# Optionally, a custom joint mapping using joint_output_map:
joint_output_map = {
    "pos_frontleftflipper":  {"offset": 0.435, "gain": 3.5},
    "pos_frontrightflipper": {"offset": 0.435, "gain": 3.5},
    "pos_backleft":          {"offset": 0.435, "gain": 2},
    "pos_backright":         {"offset": 0.435, "gain": 2},
    "pos_frontlefthip":      {"offset": 0.435, "gain": 3.5},
    "pos_frontrighthip":     {"offset": 0.435, "gain": 3.5}
}

# =============================================================================
#             SIMULATION & DATA LOGGING PARAMETERS
# =============================================================================
dt_cpg = 0.001         # Time step for CPG integration
sim_duration = 30.0    # Duration of each simulation run (seconds)

# Data logging containers for each simulation run:
# We'll log time, COM positions, and also record energy consumption.
# (Other signals, such as joint control, remain available from the main simulation.)
# For the additional performance metrics, we record:
# - Total energy consumed (via integration of instantaneous power)
# - Cost of transport (CoT)
# - X and Y displacements (forward and lateral)
def run_simulation(freq_hz):
    """
    Run simulation for a given frequency (Hz) applied to both stance and swing.
    Returns:
      - avg_speed: Average forward speed (x-displacement / sim_duration)
      - avg_freq: Average computed local frequency (rad/s)
      - total_energy: Total energy consumed (J)
      - cost_of_transport: Energy per weight*distance (J/(N*m))
      - x_disp: X-displacement (m)
      - y_disp: Absolute Y-displacement (m)
    """
    # Reset simulation state:
    mujoco.mj_resetData(model, data)
    data.time = 0
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    # Convert frequency (Hz) to rad/s:
    freq_rad = freq_hz * 2.0 * np.pi
    stance_freq_local = freq_rad
    swing_freq_local  = freq_rad

    # Reset oscillator states:
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name]["x"] = np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn()
        oscillators[name]["y"] = np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()

    sim_time = 0.0
    time_data_local = []
    com_positions_local = []
    yaw_data_local = []      # To record yaw orientation over time
    power_record = []
    freq_record = []

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

            # Record yaw orientation from the base body:
            quat = data.xquat[main_body_id].copy()  # Quaternion in [w, x, y, z]
            yaw = quaternion_to_yaw(quat)
            yaw_data_local.append(yaw)

            # Record instantaneous power consumption:
            qvel = data.qvel[:model.nu]
            torque = data.actuator_force[:model.nu]
            instant_power = np.sum(np.abs(torque) * np.abs(qvel))
            power_record.append(instant_power)

            viewer.sync()

    # Compute metrics:
    if len(com_positions_local) >= 2:
        displacement = com_positions_local[-1] - com_positions_local[0]
        x_disp = displacement[0]
        y_disp = np.abs(displacement[1])
        avg_speed = x_disp / sim_duration
    else:
        avg_speed = 0.0
        x_disp = 0.0
        y_disp = 0.0

     # Compute change in yaw orientation (final - initial):
    if len(yaw_data_local) >= 2:
        yaw_change = yaw_data_local[-1] - yaw_data_local[0]
    else:
        yaw_change = 0.0

    total_energy = np.sum(power_record) * dt_cpg
    weight = total_mass * 9.81
    cost_of_transport = total_energy / (weight * np.abs(x_disp + 1e-3))  # Avoid division by zero
    avg_freq = np.mean(freq_record) if len(freq_record) > 0 else 0.0

    return avg_speed, avg_freq, total_energy, cost_of_transport, x_disp, y_disp, yaw_change

# =============================================================================
#                   RUN SIMULATIONS OVER FREQUENCY INCREMENTS
# =============================================================================
# Define 15 frequency values in Hz, e.g., from 0 Hz to 1.2 Hz.
freq_values_Hz = np.linspace(0, 1.2, 15)
freq_values_rad_s = freq_values_Hz * 2.0 * np.pi
avg_speeds = []
avg_freqs_rad_s = []
avg_freqs_Hz = []
total_energy_list = []
cost_of_transport_list = []
x_disp_list = []
y_disp_list = []
yaw_change_list = []  # To record yaw changes


for freq in freq_values_Hz:
    print(f"Running simulation at frequency {freq:.2f} Hz...")
    avg_speed, avg_freq, total_energy, cost_transport, x_disp, y_disp, yaw_change = run_simulation(freq)
    print(f"At {freq:.2f} Hz ({freq * 2.0 * np.pi:.2f} rad/s):")
    print(f"  Avg Forward Speed = {avg_speed:.4f} m/s")
    print(f"  Avg Computed Frequency = {avg_freq:.2f} rad/s, {(avg_freq)/(2.0 * np.pi):.2f} Hz")
    print(f"  Total Energy = {total_energy:.2f} J")
    print(f"  Cost of Transport = {cost_transport:.2f}")
    print(f"  X displacement = {x_disp:.4f} m, Y displacement = {y_disp:.4f} m")
    print(f"  Yaw Orientation Change = {yaw_change:.2f} rad\n")
    avg_speeds.append(avg_speed)
    avg_freqs_rad_s.append(avg_freq)
    avg_freqs_Hz.append(avg_freq/(2.0 * np.pi))
    total_energy_list.append(total_energy)
    cost_of_transport_list.append(cost_transport)
    x_disp_list.append(x_disp)
    y_disp_list.append(y_disp)
    yaw_change_list.append(yaw_change)

# Save data to a file.
data_to_save = np.column_stack((freq_values_rad_s, freq_values_Hz, avg_speeds, avg_freqs_rad_s,
                                 avg_freqs_Hz, total_energy_list, cost_of_transport_list,
                                 x_disp_list, y_disp_list, yaw_change_list))
# np.savetxt("frequency_vs_metrics_v1_flat.txt", data_to_save,
#            header="Frequency_rad/s   Frequency_Hz   Avg_Forward_Speed_m/s   Avg_Computed_Frequency_rad/s   Avg_Computed_Frequency_Hz   Total_Energy_J   Cost_of_Transport   X_Displacement_m   Y_Displacement_m   Yaw_Change_rad")


# Create a DataFrame with appropriate column names
columns = ["Frequency_rad/s", "Frequency_Hz", "Avg_Forward_Speed_m/s", 
           "Avg_Computed_Frequency_rad/s", "Avg_Computed_Frequency_Hz", 
           "Total_Energy_J", "Cost_of_Transport", "X_Displacement_m", 
           "Y_Displacement_m", "Yaw_Change_rad"]
df = pd.DataFrame(data_to_save, columns=columns)

# Save to an Excel file
df.to_excel("frequency_vs_metrics_v4_flat.xlsx", index=False)

print("Data saved to frequency_vs_metrics_v4_flat.xlsx")


# =============================================================================
#                              PLOT RESULTS WITH SUBPLOTS
# =============================================================================
# Updated to 6 rows x 2 columns to include yaw change plot.
fig, axs = plt.subplots(6, 2, figsize=(15, 30))

# Row 0: Forward Speed vs. Input Frequency
axs[0, 0].plot(freq_values_Hz, avg_speeds, marker='o', linestyle='-')
axs[0, 0].set_title("Forward Speed vs. Input Frequency (Hz)")
axs[0, 0].set_xlabel("Input Frequency (Hz)")
axs[0, 0].set_ylabel("Avg Forward Speed (m/s)")
axs[0, 0].grid(True)

axs[0, 1].plot(freq_values_rad_s, avg_speeds, marker='o', linestyle='-')
axs[0, 1].set_title("Forward Speed vs. Input Frequency (rad/s)")
axs[0, 1].set_xlabel("Input Frequency (rad/s)")
axs[0, 1].set_ylabel("Avg Forward Speed (m/s)")
axs[0, 1].grid(True)

# Row 1: Avg Computed Frequency vs. Forward Speed
axs[1, 0].plot(avg_freqs_Hz, avg_speeds, marker='^', linestyle='-.', color='g')
axs[1, 0].set_title("Avg Computed Frequency (Hz) vs. Forward Speed")
axs[1, 0].set_xlabel("Avg Computed Frequency (Hz)")
axs[1, 0].set_ylabel("Avg Forward Speed (m/s)")
axs[1, 0].grid(True)

axs[1, 1].plot(avg_freqs_rad_s, avg_speeds, marker='^', linestyle='-.', color='g')
axs[1, 1].set_title("Avg Computed Frequency (rad/s) vs. Forward Speed")
axs[1, 1].set_xlabel("Avg Computed Frequency (rad/s)")
axs[1, 1].set_ylabel("Avg Forward Speed (m/s)")
axs[1, 1].grid(True)

# Row 2: Avg Computed Frequency vs. Input Frequency
axs[2, 0].plot(freq_values_Hz, avg_freqs_Hz, marker='s', linestyle='--', color='r')
axs[2, 0].set_title("Avg Computed Frequency vs. Input Frequency (Hz)")
axs[2, 0].set_xlabel("Input Frequency (Hz)")
axs[2, 0].set_ylabel("Avg Computed Frequency (Hz)")
axs[2, 0].grid(True)

axs[2, 1].plot(freq_values_rad_s, avg_freqs_rad_s, marker='s', linestyle='--', color='r')
axs[2, 1].set_title("Avg Computed Frequency vs. Input Frequency (rad/s)")
axs[2, 1].set_xlabel("Input Frequency (rad/s)")
axs[2, 1].set_ylabel("Avg Computed Frequency (rad/s)")
axs[2, 1].grid(True)

# Row 3: Cost of Transport and Total Energy vs. Input Frequency (Hz)
axs[3, 0].plot(freq_values_Hz, cost_of_transport_list, marker='d', linestyle='-', color='m')
axs[3, 0].set_title("Cost of Transport vs. Input Frequency (Hz)")
axs[3, 0].set_xlabel("Input Frequency (Hz)")
axs[3, 0].set_ylabel("Cost of Transport (J/(N*m))")
axs[3, 0].grid(True)

axs[3, 1].plot(freq_values_Hz, total_energy_list, marker='d', linestyle='-', color='c')
axs[3, 1].set_title("Total Energy vs. Input Frequency (Hz)")
axs[3, 1].set_xlabel("Input Frequency (Hz)")
axs[3, 1].set_ylabel("Total Energy (J)")
axs[3, 1].grid(True)

# Row 4: X and Y Displacement vs. Input Frequency (Hz)
axs[4, 0].plot(freq_values_Hz, x_disp_list, marker='o', linestyle='-', color='k')
axs[4, 0].set_title("X Displacement (Forward) vs. Input Frequency (Hz)")
axs[4, 0].set_xlabel("Input Frequency (Hz)")
axs[4, 0].set_ylabel("X Displacement (m)")
axs[4, 0].grid(True)

axs[4, 1].plot(freq_values_Hz, y_disp_list, marker='o', linestyle='-', color='orange')
axs[4, 1].set_title("Y Displacement (Lateral) vs. Input Frequency (Hz)")
axs[4, 1].set_xlabel("Input Frequency (Hz)")
axs[4, 1].set_ylabel("Y Displacement (m)")
axs[4, 1].grid(True)

# Row 5: Yaw Change vs. Input Frequency (Hz)
axs[5, 0].plot(freq_values_Hz, yaw_change_list, marker='o', linestyle='-', color='b')
axs[5, 0].set_title("Yaw Orientation Change vs. Input Frequency (Hz)")
axs[5, 0].set_xlabel("Input Frequency (Hz)")
axs[5, 0].set_ylabel("Yaw Change (rad)")
axs[5, 0].grid(True)

# Hide the empty subplot (Row 5, Column 1)
axs[5, 1].axis('off')

plt.tight_layout()
plt.show()
