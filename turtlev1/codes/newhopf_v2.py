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

# For this example, we use the actuator control range from the model:
joint_limits = {}
for i, name in enumerate(actuator_names):
    ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
    joint_limits[name] = (ctrl_min, ctrl_max)
    print(f"{name}: ctrl range = [{ctrl_min:.3f}, {ctrl_max:.3f}]")

# =============================================================================
#   SELECT BODY FOR COM/ORIENTATION & TOTAL MASS
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



# =============================================================================
#            CPG PARAMETERS & INITIAL OSCILLATOR STATES
# =============================================================================
alpha = 10.0       # Convergence speed
mu = 1         # Radius^2 (amplitude ~ sqrt(0.04)=0.2)
a_param = 10.0     # Logistic steepness for frequency blending

# Define stance and swing frequencies (rad/s)
stance_freq = 2.0
swing_freq  = 2.0

# Single coupling constant for all joints (λ)
lambda_cpl = 0.5

# Define phase offsets (as fractions; 1 corresponds to 2π)

# phase_offsets = [0.0, 0.0, 0.0, 0.0, 0.75, 0.75] # sync phase offsets
phase_offsets_d = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.0,
    "pos_backleft":          0.0,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.75
}

# phase_offsets = [0.0, 0.5, 0.5, 0.0, 0.25, 0.75] # diag phase offsets
phase_offsets_s = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.5,
    "pos_backleft":          0.5,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.25
}

phase_offsets = phase_offsets_d

# Initialize oscillator states for each joint with the desired phase offsets.
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
dt_cpg = 0.001         # Time step for CPG integration
time_duration = 30.0   # Simulation duration (seconds)

# Data logging for simulation and performance metrics:
time_data = []
ctrl_data = {name: [] for name in actuator_names}
cpg_outputs = {name: [] for name in actuator_names}
com_positions = []
body_orientations = []
power_consumption = []

# Additional sensor and actuator histories:
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
#                        MAIN SIMULATION LOOP
# =============================================================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    last_loop_time = start_time

    while viewer.is_running():
        now = time.time()
        sim_time = now - start_time
        loop_dt = now - last_loop_time
        last_loop_time = now

        if sim_time >= time_duration:
            print(f"Reached {time_duration:.1f} s of simulation. Stopping.")
            break

        # ------------------------------------------------------------
        # 1) Integrate the Hopf oscillators using multiple sub-steps.
        # ------------------------------------------------------------
        steps = int(np.floor(loop_dt / dt_cpg))
        for _ in range(steps):
            x_all = [oscillators[name]["x"] for name in actuator_names]
            y_all = [oscillators[name]["y"] for name in actuator_names]
            for i, name in enumerate(actuator_names):
                x_i = oscillators[name]["x"]
                y_i = oscillators[name]["y"]
                # Compute instantaneous frequency based on y_i (logistic blending)
                freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                       (swing_freq  / (1.0 + np.exp(a_param * y_i)))
                x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                         lambda_cpl, x_all, y_all, i, phase_offsets)
                oscillators[name]["x"] = x_new
                oscillators[name]["y"] = y_new

        # ------------------------------------------------------------
        # 2) Map oscillator outputs to joint control commands (full-range mapping)
        # ------------------------------------------------------------
        # Here we use the inherent phase relationships from the CPG network.
        for name in actuator_names:
            min_angle, max_angle = joint_limits[name]
            # Option 1: Use direct mapping (if you want to use the oscillator x output)
            # desired_angle = map_oscillator_to_joint(oscillators[name]["x"], min_angle, max_angle)
            # Option 2: Alternatively, use a custom mapping (e.g., with joint_output_map) if desired:
            amplitude = np.sqrt(mu)

            offset = joint_output_map[name]["offset"]
            gain = joint_output_map[name]["gain"]
            desired_angle = offset + gain * np.tanh(oscillators[name]["x"])
    
            # desired_angle = offset + gain * oscillators[name]["x"]
            # desired_angle = offset + gain * (oscillators[name]["x"] / amplitude)
            desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
            data.ctrl[actuator_indices[name]] = desired_angle_clamped
            cpg_outputs[name].append(desired_angle_clamped)

        time_data.append(sim_time)

        # ------------------------------------------------------------
        # 3) Step the MuJoCo simulation and collect performance data.
        # ------------------------------------------------------------
        mujoco.mj_step(model, data)

        # Record control signals:
        for name in actuator_names:
            ctrl_data[name].append(data.ctrl[actuator_indices[name]])
        # (a) COM position:
        com_positions.append(data.xpos[main_body_id].copy())
        # (b) Orientation (rotation matrix):
        body_orientations.append(data.xmat[main_body_id].copy())
        # (c) Mechanical power (using absolute torque * absolute joint velocity):
        qvel = data.qvel[:model.nu]
        torque = data.actuator_force[:model.nu]
        instant_power = np.sum(np.abs(torque) * np.abs(qvel))
        power_consumption.append(instant_power)
        # (d) Joint torque sensors:
        for varname, sname in jointact_sensor_map.items():
            val = get_sensor_data(data, model, sensor_name2id, sname)
            if val is not None:
                sensor_data_history[varname].append(val[0])
        # (e) Base IMU data:
        for varname, sname in base_imu_map.items():
            val = get_sensor_data(data, model, sensor_name2id, sname)
            if val is not None:
                sensor_data_history[varname].append(val.copy())
        # (f) Actuator torques:
        for name in actuator_names:
            idx = actuator_indices[name]
            actuator_torque_history[name].append(data.actuator_force[idx])
        # (g) Joint velocities:
        for name in actuator_names:
            idx = actuator_indices[name]
            joint_velocity_history[name].append(qvel[idx])

        viewer.sync()

# =============================================================================
#                     PERFORMANCE ANALYSIS & PLOTS
# =============================================================================
final_time = time_data[-1] if len(time_data) > 0 else 0.0
print("\n=== Performance Analysis ===")
print(f"Simulation time recorded: {final_time:.2f} s")
print(f"Coupling strength (λ): {lambda_cpl}")
print(f"Total robot mass: {total_mass:.2f} kg")

if len(com_positions) > 1:
    displacement = com_positions[-1] - com_positions[0]
    distance_traveled = np.linalg.norm(displacement)
    print(f"Total COM displacement: {displacement}")
    print(f"Straight-line distance traveled: {distance_traveled:.4f} m")
    avg_velocity = distance_traveled / final_time if final_time > 0 else 0
    print(f"Approx average speed: {avg_velocity:.4f} m/s")

dt_integration = dt_cpg
if len(power_consumption) > 1:
    total_energy = np.sum(power_consumption) * dt_integration
    print(f"Approx total energy consumed: {total_energy:.3f} J (assuming torque*velocity)")
    if distance_traveled > 0.01:
        weight = total_mass * 9.81
        cost_of_transport = total_energy / (weight * distance_traveled)
        print(f"Approx cost of transport: {cost_of_transport:.3f}")

if len(body_orientations) > 0:
    final_orientation = body_orientations[-1]
    print(f"Final orientation matrix of main body: \n{final_orientation}")

for varname in jointact_sensor_map:
    if sensor_data_history[varname]:
        last_val = sensor_data_history[varname][-1]
        print(f"Final torque sensor reading for {varname}: {last_val:.3f}")

print("\n=== Final Actuator Torque Values (from data.actuator_force) ===")
for name in actuator_names:
    idx = actuator_indices[name]
    torque_value = data.actuator_force[idx]
    print(f"{name}: {torque_value:.3f} Nm")

total_actuator_torques = {}
for name in actuator_names:
    total_actuator_torques[name] = np.sum(np.abs(actuator_torque_history[name])) * dt_integration
print("\n=== Total Integrated Actuator Torques (abs) ===")
for name, tot in total_actuator_torques.items():
    print(f"{name}: {tot:.3f} Nm·s")

# --------------------------- PLOTS ---------------------------------
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# (0,0): Actuator Control Signals
for name in actuator_names:
    axs[0, 0].plot(time_data, ctrl_data[name], label=name)
axs[0, 0].set_title("Actuator Control Signals")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Joint Angle (rad)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# (0,1): COM Position vs Time (X, Y, Z)
com_arr = np.array(com_positions)
if len(com_arr) > 0:
    axs[0, 1].plot(time_data, com_arr[:, 0], label="COM X")
    axs[0, 1].plot(time_data, com_arr[:, 1], label="COM Y")
    axs[0, 1].plot(time_data, com_arr[:, 2], label="COM Z")
axs[0, 1].set_title("COM Position vs Time")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Position (m)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# (1,0): Instantaneous Power Consumption
axs[1, 0].plot(time_data, power_consumption, label="Instant Power")
axs[1, 0].set_title("Instantaneous Power Consumption")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Power (W)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# (1,1): Trajectory (X vs Y)
if len(com_arr) > 0:
    axs[1, 1].plot(com_arr[:, 0], com_arr[:, 1], 'b-', label="Trajectory")
axs[1, 1].set_title("Trajectory (X vs Y)")
axs[1, 1].set_xlabel("X (m)")
axs[1, 1].set_ylabel("Y (m)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# (2,0): Actuator Torque Over Time
for name in actuator_names:
    axs[2, 0].plot(time_data, actuator_torque_history[name], label=name)
axs[2, 0].set_title("Actuator Torque Over Time")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Torque (Nm)")
axs[2, 0].legend()
axs[2, 0].grid(True)

# (2,1): Joint Velocity Over Time
for name in actuator_names:
    axs[2, 1].plot(time_data, joint_velocity_history[name], label=name)
axs[2, 1].set_title("Joint Velocity Over Time")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Velocity (rad/s)")
axs[2, 1].legend()
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()
# =============================================================================
#                         END OF SIMULATION