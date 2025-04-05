import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# =============================================================================
#           LOAD MUJOCO MODEL AND SET UP SENSOR/ACTUATOR LOOKUP
# =============================================================================

model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/turtlev1/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Build sensor lookup (if sensors are defined in the XML)
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
    """Helper function to retrieve sensor reading for the given sensor name."""
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

# Note: Adjusted joint limits for this example.
joint_limits = {
    "pos_frontleftflipper":  (-0.3, 1.571),
    "pos_frontrightflipper": (-0.3, 1.571),
    "pos_backleft":          (-0.3, 1.571),
    "pos_backright":         (-0.3, 1.571),
    "pos_frontlefthip":      (-0.6, 1.571),
    "pos_frontrighthip":     (-0.6, 1.571)
}

# =============================================================================
#            SELECT BODY FOR COM / ORIENTATION & TOTAL MASS
# =============================================================================

main_body_name = "base"
try:
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)
except Exception as e:
    print("Could not find body named", main_body_name, ":", e)
    main_body_id = 0  # fallback

total_mass = np.sum(model.body_mass)

# =============================================================================
#            HOPF OSCILLATOR PARAMETERS & INITIALIZATION
# =============================================================================

# Number of actuated joints (each driven by a Hopf oscillator)
N = 6

# Hopf oscillator constants
alpha = 10.0      # Convergence rate to the limit cycle
mu = 1.0          # amplitude^2 => amplitude = sqrt(mu) = 1.0

# Coupling strength coefficient (inter-oscillator sync)
K_cp = 0.5

# Frequency modulation parameters for gait phases
omega_swing = 0.5 * np.pi   # Swing phase frequency (e.g., 1 Hz -> π rad/s)
beta = 0.5                # Proportion of the gait cycle in the support phase
omega_stance = (1.0 - beta) / beta * omega_swing

# Exponential slope for the smooth transition between ω_stance and ω_swing
a_exp = 1000.0

# Phase offsets to generate a desired gait pattern.
# (For example, sync phase offsets: all zeros for flippers or set hips differently.)
# phase_offsets = [0.0, 0.0, 0.0, 0.0, 0.25, 0.25]
# Alternative (diagonal gait) example:
phase_offsets = [0.0, 0.5, 0.5, 0.0, 0.25, 0.75]

# Initialize oscillator states (x_i, y_i) for each joint
x = np.zeros(N)
y = np.zeros(N)
for i in range(N):
    phase0 = phase_offsets[i] * 2.0 * np.pi
    x[i] = np.sqrt(mu) * np.cos(phase0)
    y[i] = np.sqrt(mu) * np.sin(phase0)

# =============================================================================
#            DEFINE FUNCTIONS FOR OSCILLATOR DYNAMICS & COUPLING
# =============================================================================

def compute_omega(yi):
    """
    Compute the instantaneous oscillator frequency ω based on y.
    Smoothly blend between ω_stance and ω_swing.
    """
    return (
        (omega_stance / (1.0 + np.exp(-a_exp * yi))) +
        (omega_swing  / (1.0 + np.exp( a_exp * yi)))
    )

def update_cpgs(x, y, dt):
    """
    Update the state of N coupled Hopf oscillators for one time step dt.
    Returns updated (x_next, y_next).
    """
    x_next = np.zeros_like(x)
    y_next = np.zeros_like(y)
    
    # Current amplitude and instantaneous frequency
    r = np.sqrt(x**2 + y**2)
    omega_vals = [compute_omega(y[i]) for i in range(N)]
    
    # Precompute phase differences φ_ij = 2π(phase_offsets[i] - phase_offsets[j])
    phi_matrix = np.zeros((N, N))
    for i_osc in range(N):
        for j_osc in range(N):
            if i_osc != j_osc:
                phi_matrix[i_osc, j_osc] = 2.0 * np.pi * (
                    phase_offsets[i_osc] - phase_offsets[j_osc]
                )
    
    # Hopf oscillator update with coupling
    for i_osc in range(N):
        w_i = omega_vals[i_osc]
        cx, cy = 0.0, 0.0
        for j_osc in range(N):
            if i_osc == j_osc:
                continue
            phi_ij = phi_matrix[i_osc, j_osc]
            cx += x[j_osc] * np.cos(phi_ij) - y[j_osc] * np.sin(phi_ij)
            cy += x[j_osc] * np.sin(phi_ij) + y[j_osc] * np.cos(phi_ij)
        
        dx = alpha * (mu - r[i_osc]**2) * x[i_osc] - w_i * y[i_osc] + K_cp * cx
        dy = alpha * (mu - r[i_osc]**2) * y[i_osc] + w_i * x[i_osc] + K_cp * cy
        
        x_next[i_osc] = x[i_osc] + dx * dt
        y_next[i_osc] = y[i_osc] + dy * dt
    
    return x_next, y_next

def map_oscillator_to_joint(x_val, min_angle, max_angle):
    """
    Linearly map x_val from [-sqrt(mu), +sqrt(mu)] to [min_angle, max_angle].
    Uses:
      - offset = (min_angle + max_angle) / 2
      - gain   = (max_angle - min_angle) / 2
      - Normalize x_val by sqrt(mu) so that it is in [-1, 1]
      - desired_angle = offset + gain * (x_val / sqrt(mu))
    This ensures that when x_val reaches ±sqrt(mu), the joint command reaches the corresponding min or max.
    """
    amplitude = np.sqrt(mu)
    offset = (min_angle + max_angle) / 2.0
    gain = (max_angle - min_angle) / 2.0
    desired_angle = offset + gain * (x_val / amplitude)
    return desired_angle

# =============================================================================
#             SET UP TIME PARAMETERS AND DATA LOGGING
# =============================================================================

dt_cpg = 0.001      # Time step for oscillator updates
time_ds = 60.0      # Total simulation duration (seconds)

# Data logging for simulation & performance metrics:
time_data = []
cpg_outputs = {name: [] for name in actuator_names}  # Joint control commands

# Performance data:
ctrl_data = {name: [] for name in actuator_names}
com_positions = []
body_orientations = []
power_consumption = []
sensor_data_history = {
    "torque_backright": [],
    "torque_backleft": [],
    "torque_frontrighthip": [],
    "torque_frontrightflipper": [],
    "torque_frontlefthip": [],
    "torque_frontleftflipper": []
}
jointact_sensor_map = {
    "torque_backright":        "sens_jointactfrc_backright",
    "torque_backleft":         "sens_jointactfrc_backleft",
    "torque_frontrighthip":    "sens_jointactfrc_frontrighthip",
    "torque_frontrightflipper":"sens_jointactfrc_frontrightflipper",
    "torque_frontlefthip":     "sens_jointactfrc_frontlefthip",
    "torque_frontleftflipper": "sens_jointactfrc_frontleftflipper"
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
        
        # End simulation after time_ds seconds
        if sim_time >= time_ds:
            print(f"Reached {time_ds:.1f} s of simulation.")
            break

        # ------------------------------------------------------------
        #  Multiple oscillator sub-steps based on elapsed real time
        # ------------------------------------------------------------
        steps = int(np.floor(loop_dt / dt_cpg))
        for _ in range(steps):
            x, y = update_cpgs(x, y, dt_cpg)

        # ------------------------------------------------------------
        #  Record instantaneous frequencies (for plotting later)
        # ------------------------------------------------------------
        inst_freq = [compute_omega(y[i]) for i in range(N)]
        for i in range(N):
            # Append frequency for oscillator i
            # (We could also log x and y if desired.)
            pass  # (Logging inst_freq is optional; we log reference values below.)
        # Log reference stance and swing frequencies:
        omega_st_ref = omega_stance
        omega_sw_ref = omega_swing

        # ------------------------------------------------------------
        #  Map oscillator outputs to joint control commands using full range mapping
        # ------------------------------------------------------------
        for i, name in enumerate(actuator_names):
            min_angle, max_angle = joint_limits[name]
            desired_angle = map_oscillator_to_joint(x[i], min_angle, max_angle)
            desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
            data.ctrl[actuator_indices[name]] = desired_angle_clamped
            cpg_outputs[name].append(desired_angle_clamped)

        # ------------------------------------------------------------
        #  Step the MuJoCo simulation and collect performance metrics
        # ------------------------------------------------------------
        mujoco.mj_step(model, data)

        # Store simulation time and control data:
        time_data.append(sim_time)
        for name in actuator_names:
            ctrl_data[name].append(data.ctrl[actuator_indices[name]])

        # (a) Center of Mass (COM) position
        com_pos = data.xpos[main_body_id].copy()
        com_positions.append(com_pos)

        # (b) Orientation (rotation matrix of the main body)
        orientation_mat = data.xmat[main_body_id].copy()
        body_orientations.append(orientation_mat)

        # (c) Instantaneous mechanical power consumption (using actuator force and joint velocity)
        qvel = data.qvel[:model.nu]
        torque = data.actuator_force[:model.nu]
        instant_power = np.sum(np.abs(torque) * np.abs(qvel))
        power_consumption.append(instant_power)

        # (d) Joint torque sensor data
        for varname, sname in jointact_sensor_map.items():
            val = get_sensor_data(data, model, sensor_name2id, sname)
            if val is not None:
                sensor_data_history[varname].append(val[0])

        # (e) Actuator torques (from data.actuator_force)
        for name in actuator_names:
            idx = actuator_indices[name]
            actuator_torque_history[name].append(data.actuator_force[idx])

        # (f) Joint velocities
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
print(f"Coupling strength (K_cp): {K_cp}")
print(f"Total robot mass: {total_mass:.2f} kg")

# COM displacement and trajectory
if len(com_positions) > 1:
    displacement = com_positions[-1] - com_positions[0]
    distance_traveled = np.linalg.norm(displacement)
    print(f"Total displacement of COM: {displacement}")
    print(f"Straight-line distance traveled: {distance_traveled:.3f} m")
    avg_velocity = distance_traveled / final_time if final_time > 0 else 0
    print(f"Approx average speed: {avg_velocity:.3f} m/s")

# Energy consumption and cost of transport
dt_integration = dt_cpg
if len(power_consumption) > 1:
    total_energy = np.sum(power_consumption) * dt_integration
    print(f"Approx total energy consumed: {total_energy:.3f} J")
    if distance_traveled > 0.01:
        weight = total_mass * 9.81
        cost_of_transport = total_energy / (weight * distance_traveled)
        print(f"Approx cost of transport: {cost_of_transport:.3f}")

# Orientation analysis
if len(body_orientations) > 0:
    final_orientation = body_orientations[-1]
    print(f"Final orientation matrix: \n{final_orientation}")

# Final sensor readings from joint torque sensors
for varname in jointact_sensor_map:
    if sensor_data_history[varname]:
        last_val = sensor_data_history[varname][-1]
        print(f"Final torque sensor reading for {varname}: {last_val:.3f}")

# Final actuator torques
print("\n=== Final Actuator Torque Values ===")
for name in actuator_names:
    idx = actuator_indices[name]
    torque_value = data.actuator_force[idx]
    print(f"{name}: {torque_value:.3f} Nm")

# Total integrated actuator torques (absolute sum over time)
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
