import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#                           HOPF OSCILLATOR DYNAMICS
# -----------------------------------------------------------------------------
def hopf_step(x, y, alpha, mu, omega, dt, coupling, xall, yall, index):
    """
    Integrate one step of the Hopf oscillator (with coupling) using Euler's method.
    """
    r_sq = x * x + y * y

    # Base Hopf dynamics (limit cycle)
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    # Add linear coupling from other oscillators
    for j in range(len(xall)):
        if j == index:
            continue
        K_ij = coupling[index, j]
        dx += K_ij * (xall[j] - x)
        dy += K_ij * (yall[j] - y)

    # Euler integration
    x_new = x + dx * dt
    y_new = y + dy * dt
    return x_new, y_new

# -----------------------------------------------------------------------------
#                         LOAD MUJOCO MODEL
# -----------------------------------------------------------------------------
model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/turtlev1/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# -----------------------------------------------------------------------------
#  Build a lookup from sensor name to sensor ID (if sensors are defined in the XML)
# -----------------------------------------------------------------------------
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
    """Helper function to retrieve the sensor reading for the given sensor name."""
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

# -----------------------------------------------------------------------------
#            DEFINE ACTUATOR NAMES & GET INDICES
# -----------------------------------------------------------------------------
actuator_names = [
    "pos_frontleftflipper",
    "pos_frontrightflipper",
    "pos_backleft",
    "pos_backright",
    "pos_frontlefthip",
    "pos_frontrighthip"
]
actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

# -----------------------------------------------------------------------------
#          JOINT LIMITS (ctrlrange) PER ACTUATOR (from your MJCF)
# -----------------------------------------------------------------------------
joint_limits = {
    # "pos_frontleftflipper":  (-1.57,  0.64),
    # "pos_frontrightflipper": (-0.64,  1.571),
    # "pos_backleft":          (-1.571, 0.524),
    # "pos_backright":         (-0.524, 1.571),
    # "pos_frontlefthip":      (-1.571, 1.22),
    # "pos_frontrighthip":     (-1.22,  1.571)
    "pos_frontleftflipper":  (-0.7, 1.571),
    "pos_frontrightflipper": (-0.7, 1.571),
    "pos_backleft":          (-0.7, 1.571),
    "pos_backright":         (-0.7, 1.571),
    "pos_frontlefthip":      (-1.22, 1.571),
    "pos_frontrighthip":     (-1.22, 1.571)
}

# -----------------------------------------------------------------------------
#        CPG PARAMETERS & INITIAL OSCILLATOR STATES
# -----------------------------------------------------------------------------
alpha    = 10.0   # Gain, convergence speed
mu       = 0.04   # radius^2 => amplitude ~ sqrt(0.04) = 0.2
a_param  = 10.0   # logistic steepness for stance/swing frequency blending

# Define separate stance & swing frequencies (rad/s)
stance_freq = 4.0
swing_freq  = 8.0

# Random initial states so they don't start in exact sync
oscillators = {}
for name in actuator_names:
    oscillators[name] = {
        "x": 0.02 * np.random.randn(),
        "y": 0.02 * np.random.randn()
    }

# -----------------------------------------------------------------------------
#        COUPLING MATRIX (K) FOR IN-PHASE VS. OUT-OF-PHASE
# -----------------------------------------------------------------------------
num_joints = len(actuator_names)
K = np.zeros((num_joints, num_joints))

in_phase_coupling  = 0.8
out_phase_coupling = -0.8

left_indices  = [0, 2, 4]  # frontleftflipper, backleft, frontlefthip
right_indices = [1, 3, 5]  # frontrightflipper, backright, frontrighthip

for i in range(num_joints):
    for j in range(num_joints):
        if i == j:
            continue
        both_left  = (i in left_indices) and (j in left_indices)
        both_right = (i in right_indices) and (j in right_indices)
        cross_side = (i in left_indices and j in right_indices) or (i in right_indices and j in left_indices)
        
        if both_left or both_right:
            K[i, j] = in_phase_coupling
        elif cross_side:
            K[i, j] = out_phase_coupling

# -----------------------------------------------------------------------------
#  PHASE OFFSETS & LINEAR MAPPING FROM (x,y) -> JOINT ANGLE
# -----------------------------------------------------------------------------
# Using the diagonal forward configuration as an example:
phase_offsets_diagforward = {
    "pos_frontleftflipper":  -np.pi,
    "pos_frontrightflipper":  0.0,
    "pos_backleft":           0.0,
    "pos_backright":          np.pi,
    "pos_frontlefthip":      -np.pi/2,
    "pos_frontrighthip":      np.pi/2
}

joint_output_map = {
    "pos_frontleftflipper":  {"offset": 0.8, "gain": 3.0},
    "pos_frontrightflipper": {"offset":  0.8, "gain": 3.0},
    "pos_backleft":          {"offset": 0.5, "gain": 1.0},
    "pos_backright":         {"offset":  0.5, "gain": 1.0},
    "pos_frontlefthip":      {"offset":  0.3, "gain": -1.0},
    "pos_frontrighthip":     {"offset":  0.3, "gain":  1.0}
}

# -----------------------------------------------------------------------------
#  SELECT WHICH BODY TO TRACK FOR COM, ORIENTATION, ETC.
# -----------------------------------------------------------------------------
main_body_name = "base"  # main body for CoM & orientation measurements
try:
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)
except Exception as e:
    print("Could not find body named", main_body_name, ":", e)
    main_body_id = 0  # fallback to root body

# Total mass (approx. sum of all body masses)
total_mass = np.sum(model.body_mass)

# -----------------------------------------------------------------------------
#                       DATA COLLECTION FOR PLOTTING
# -----------------------------------------------------------------------------
time_data = []
ctrl_data = {name: [] for name in actuator_names}

# Performance metrics storage:
com_positions = []        # store main body positions (COM)
body_orientations = []    # store rotation matrices
power_consumption = []    # store instantaneous power (torque*velocity)
run_time = 30.0           # simulation duration in seconds

# Sensor data history for joint torque sensors (from <jointactuatorfrc>)
# (We removed the base IMU sensor data collection as requested.)
sensor_data_history = {
    "torque_backright": [],
    "torque_backleft": [],
    "torque_frontrighthip": [],
    "torque_frontrightflipper": [],
    "torque_frontlefthip": [],
    "torque_frontleftflipper": []
}
# We'll also record the actuator torques from data.actuator_force:
actuator_torque_history = {name: [] for name in actuator_names}

# We'll also record each joint's velocity (from data.qvel[:model.nu])
joint_velocity_history = {name: [] for name in actuator_names}

# The sensor names in the XML for joint torque sensors:
jointact_sensor_map = {
    "torque_backright":       "sens_jointactfrc_backright",
    "torque_backleft":        "sens_jointactfrc_backleft",
    "torque_frontrighthip":   "sens_jointactfrc_frontrighthip",
    "torque_frontrightflipper":"sens_jointactfrc_frontrightflipper",
    "torque_frontlefthip":    "sens_jointactfrc_frontlefthip",
    "torque_frontleftflipper":"sens_jointactfrc_frontleftflipper"
}

# -----------------------------------------------------------------------------
#                       MAIN SIMULATION LOOP
# -----------------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    dt_cpg = 0.001
    last_loop_time = time.time()

    while viewer.is_running():
        now = time.time()
        loop_dt = now - last_loop_time
        last_loop_time = now

        sim_time = now - start_time

        # ---------------------------------------------------------------------
        # Stop after run_time seconds to gather final metrics
        # ---------------------------------------------------------------------
        if sim_time >= run_time:
            print(f"Reached {run_time:.0f} seconds of simulation. Stopping now for analysis.")
            break

        # ---------------------------------------------------------------------
        # 1) Integrate the Hopf oscillators
        # ---------------------------------------------------------------------
        steps = int(np.floor(loop_dt / dt_cpg))
        for _ in range(steps):
            x_all = [oscillators[name]["x"] for name in actuator_names]
            y_all = [oscillators[name]["y"] for name in actuator_names]

            for i, name in enumerate(actuator_names):
                x_i = oscillators[name]["x"]
                y_i = oscillators[name]["y"]

                # Frequency blending using logistic function
                freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                       (swing_freq  / (1.0 + np.exp( a_param * y_i)))
                x_new, y_new = hopf_step(
                    x_i, y_i, alpha, mu, freq, dt_cpg,
                    K, x_all, y_all, i
                )
                oscillators[name]["x"] = x_new
                oscillators[name]["y"] = y_new

        # ---------------------------------------------------------------------
        # 2) Convert oscillator states to joint angles & apply control
        # ---------------------------------------------------------------------
        for name in actuator_names:
            x_i = oscillators[name]["x"]
            y_i = oscillators[name]["y"]

            delta_phi = phase_offsets_diagforward[name]
            # Compute phase-projected value
            x_phase = x_i * np.cos(delta_phi) - y_i * np.sin(delta_phi)

            offset = joint_output_map[name]["offset"]
            gain   = joint_output_map[name]["gain"]
            angle_raw = offset + gain * x_phase

            min_angle, max_angle = joint_limits[name]
            angle_clamped = np.clip(angle_raw, min_angle, max_angle)
            data.ctrl[actuator_indices[name]] = angle_clamped

        # ---------------------------------------------------------------------
        # 3) Step MuJoCo simulation
        # ---------------------------------------------------------------------
        mujoco.mj_step(model, data)

        # Record simulation time and control commands
        time_data.append(sim_time)
        for name in actuator_names:
            ctrl_data[name].append(data.ctrl[actuator_indices[name]])

        # ---------------------------------------------------------------------
        # 4) Collect kinematics and performance metrics
        # ---------------------------------------------------------------------
        # (a) Main body (COM) position
        com_pos = data.xpos[main_body_id].copy()
        com_positions.append(com_pos)

        # (b) Main body orientation (rotation matrix)
        orientation_mat = data.xmat[main_body_id].copy()
        body_orientations.append(orientation_mat)

        # (c) Mechanical power usage: torque * joint velocity (for actuators)
        qvel = data.qvel[:model.nu]
        torque = data.actuator_force[:model.nu]
        instant_power = np.sum(np.abs(torque) * np.abs(qvel))
        power_consumption.append(instant_power)

        # ---------------------------------------------------------------------
        # 5) Collect sensor data for joint torque sensors (from <jointactuatorfrc>)
        for varname, sname in jointact_sensor_map.items():
            val = get_sensor_data(data, model, sensor_name2id, sname)
            if val is not None:
                sensor_data_history[varname].append(val[0])  # assume 1D reading

        # ---------------------------------------------------------------------
        # 6) Collect actuator torque values from data.actuator_force
        for name in actuator_names:
            idx = actuator_indices[name]
            actuator_torque_history[name].append(data.actuator_force[idx])


        # 7) Collect joint velocities for each actuator (from data.qvel)
        for name in actuator_names:
            idx = actuator_indices[name]
            joint_velocity_history[name].append(qvel[idx])

        viewer.sync()

# -----------------------------------------------------------------------------
#  AFTER SIMULATION: Print Final Metrics
# -----------------------------------------------------------------------------
final_time = time_data[-1] if len(time_data) > 0 else 0.0
print("\n=== Performance Analysis ===")
print(f"Simulation time recorded: {final_time:.2f} s")
print(f"In Phase coupling value: {in_phase_coupling}")
print(f"Out of Phase coupling value: {out_phase_coupling}")
print(f"Total mass of robot: {total_mass:.2f} kg")

# COM analysis
if len(com_positions) > 1:
    displacement = com_positions[-1] - com_positions[0]
    distance_traveled = np.linalg.norm(displacement)
    print(f"Total displacement of COM: {displacement}")
    print(f"Straight-line distance traveled: {distance_traveled:.3f} m")
    avg_velocity = distance_traveled / final_time if final_time > 0 else 0
    print(f"Approx average speed: {avg_velocity:.3f} m/s")

# Power and energy consumption
dt_integration = dt_cpg
if len(power_consumption) > 1:
    total_energy = np.sum(power_consumption) * dt_integration
    print(f"Approx total energy consumed: {total_energy:.3f} J (torque*velocity)")
    if distance_traveled > 0.01:
        weight = total_mass * 9.81
        cost_of_transport = total_energy / (weight * distance_traveled)
        print(f"Approx cost of transport: {cost_of_transport:.3f}")

# Orientation analysis
if len(body_orientations) > 0:
    final_orientation = body_orientations[-1]
    print(f"Final orientation matrix of main body: \n{final_orientation}")

# Joint torque sensor final readings
for varname in jointact_sensor_map:
    if sensor_data_history[varname]:
        last_val = sensor_data_history[varname][-1]
        print(f"Final torque sensor reading for {varname}: {last_val:.3f}")

# Actuator torque values from data.actuator_force (final values)
print("\n=== Final Actuator Torque Values (from data.actuator_force) ===")
for name in actuator_names:
    idx = actuator_indices[name]
    torque_value = data.actuator_force[idx]
    print(f"{name}: {torque_value:.3f} Nm")

# Compute total (absolute) actuator torques over time
total_actuator_torques = {}
for name in actuator_names:
    total_actuator_torques[name] = np.sum(np.abs(actuator_torque_history[name])) * dt_cpg
print("\n=== Total Actuator Torques Over Time (integrated absolute torque) ===")
for name, tot in total_actuator_torques.items():
    print(f"{name}: {tot:.3f} Nm·s")

# -----------------------------------------------------------------------------
#                PLOTTING: Create Subplots for Collected Metrics
# -----------------------------------------------------------------------------
# We'll use a 3 x 2 grid of subplots.
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# Subplot (0,0): Actuator Control Signals
for name in actuator_names:
    axs[0, 0].plot(time_data, ctrl_data[name], label=name)
axs[0, 0].set_title("Actuator Control Signals")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Joint Angle (rad)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Subplot (0,1): COM Position vs Time (each coordinate)
com_positions_arr = np.array(com_positions)  # shape (n, 3)
axs[0, 1].plot(time_data, com_positions_arr[:, 0], label="COM X")
axs[0, 1].plot(time_data, com_positions_arr[:, 1], label="COM Y")
axs[0, 1].plot(time_data, com_positions_arr[:, 2], label="COM Z")
axs[0, 1].set_title("COM Position vs Time")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Position (m)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Subplot (1,0): Trajectory: COM X vs COM Y
axs[1, 0].plot(time_data, power_consumption, label="Instant Power")
axs[1, 0].set_title("Instantaneous Power Consumption")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Power (W)")
axs[1, 0].legend()
axs[1, 0].grid(True)


# Subplot (1,1): Instantaneous Power Consumption
axs[1, 1].plot(com_positions_arr[:, 0], com_positions_arr[:, 1], 'b-', label="Trajectory")
axs[1, 1].set_title("Trajectory (X vs Y)")
axs[1, 1].set_xlabel("X (m)")
axs[1, 1].set_ylabel("Y (m)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Subplot (2,0): Actuator Torque Over Time (from data.actuator_force)
for name in actuator_names:
    axs[2, 0].plot(time_data, actuator_torque_history[name], label=name)
axs[2, 0].set_title("Actuator Torque Over Time")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Torque (Nm)")
axs[2, 0].legend()
axs[2, 0].grid(True)

# Subplot (2,1): Joint Velocity Over Time
for name in actuator_names:
    axs[2, 1].plot(time_data, joint_velocity_history[name], label=name)
axs[2, 1].set_title("Joint Velocity Over Time")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Velocity (rad/s)")
axs[2, 1].legend()
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()
