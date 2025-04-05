import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# =============================================================================
#                 HOPF OSCILLATOR WITH PHASE OFFSETS (2D MATRIX)
# =============================================================================
def hopf_step_with_phase(
    x, y, alpha, mu, omega, dt,
    xall, yall, index,
    coupling_matrix,
    phase_matrix
):
    """
    Integrate one step of the Hopf oscillator (with linear + phase offsets in coupling)
    using a simple Euler method.

    x, y : state of the oscillator i
    xall, yall : states of all oscillators
    coupling_matrix : K[i, j], the coupling gains
    phase_matrix : φ[i, j], the desired phase offset between oscillator i and j
                   (i.e., offset for i minus offset for j)
    """
    r_sq = x*x + y*y

    # Base Hopf dynamics (limit cycle)
    dx = alpha*(mu - r_sq)*x - omega*y
    dy = alpha*(mu - r_sq)*y + omega*x

    # Add coupling from other oscillators i <- j
    #   dx += K[i, j]*((x_j*cos φ[i,j] - y_j*sin φ[i,j]) - (x*cos φ[i,j] - y*sin φ[i,j]))
    #   dy += K[i, j]*((x_j*sin φ[i,j] + y_j*cos φ[i,j]) - (x*sin φ[i,j] + y*cos φ[i,j]))
    for j in range(len(xall)):
        if j == index:
            continue

        K_ij  = coupling_matrix[index, j]
        phi_ij = phase_matrix[index, j]  # phase offset for i relative to j

        xj = xall[j]
        yj = yall[j]
        cos_phi = np.cos(phi_ij)
        sin_phi = np.sin(phi_ij)

        # Coupling terms
        dx_coupling = K_ij * ((xj*cos_phi - yj*sin_phi) - (x*cos_phi - y*sin_phi))
        dy_coupling = K_ij * ((xj*sin_phi + yj*cos_phi) - (x*sin_phi + y*cos_phi))

        dx += dx_coupling
        dy += dy_coupling

    # Euler step
    x_new = x + dx*dt
    y_new = y + dy*dt
    return x_new, y_new

# =============================================================================
#                   LOAD MUJOCO MODEL
# =============================================================================
model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/turtlev1/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# ------------------------------------------------------------------------------
#  Build a lookup from sensor name to sensor ID (if needed)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
#            DEFINE ACTUATOR NAMES & GET INDICES
# ------------------------------------------------------------------------------
actuator_names = [
    "pos_frontleftflipper",
    "pos_frontrightflipper",
    "pos_backleft",
    "pos_backright",
    "pos_frontlefthip",
    "pos_frontrighthip"
]
actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

# ------------------------------------------------------------------------------
#          JOINT LIMITS (ctrlrange) PER ACTUATOR (from your MJCF)
# ------------------------------------------------------------------------------
joint_limits = {
    "pos_frontleftflipper":  (-0.7, 1.571),
    "pos_frontrightflipper": (-0.7, 1.571),
    "pos_backleft":          (-0.7, 1.571),
    "pos_backright":         (-0.7, 1.571),
    "pos_frontlefthip":      (-0.7, 1.571),
    "pos_frontrighthip":     (-0.7, 1.571)
}

# ------------------------------------------------------------------------------
#        CPG PARAMETERS & INITIAL OSCILLATOR STATES
# ------------------------------------------------------------------------------
alpha    = 10.0   # convergence speed
mu       = 0.04   # radius^2 => amplitude ~ sqrt(0.04) = 0.2
a_param  = 10.0   # logistic steepness for stance/swing freq blending
stance_freq = 4.0
swing_freq  = 4.0

# Random initial x,y
oscillators = {}
for name in actuator_names:
    oscillators[name] = {
        "x": 0.02*np.random.randn(),
        "y": 0.02*np.random.randn()
    }

num_joints = len(actuator_names)

# ------------------------------------------------------------------------------
#   COUPLING GAINS (K) & PHASE OFFSETS (phiMatrix) BUILD
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
#   YOUR DESIRED PHASE OFFSETS PER JOINT (RELATIVE TO A GLOBAL REF)
# ------------------------------------------------------------------------------
phase_offsets_diagforward = {
     # Diagonal pairing: front left and back right are in-phase,
    # while front right is shifted by -pi and back left is shifted by pi.
    "pos_frontleftflipper":  -np.pi,
    "pos_frontrightflipper":  0.0,      # opposite phase relative to front left
    "pos_backleft":           np.pi,      # in-phase with front right
    "pos_backright":          0.0,    # opposite phase relative to front right
    # Hips: both are shifted so as to maintain a downward orientation.
    "pos_frontlefthip":      -np.pi/2,
    "pos_frontrighthip":      np.pi/2
}

# We now build the 2D matrix of phase differences: φ[i,j] = offset_i - offset_j
phase_matrix = np.zeros((num_joints, num_joints))
for i, name_i in enumerate(actuator_names):
    for j, name_j in enumerate(actuator_names):
        phase_matrix[i, j] = (
            phase_offsets_diagforward[name_i] -
            phase_offsets_diagforward[name_j]
        )

# ------------------------------------------------------------------------------
#     MAPPING FUNCTION WITH DIRECTIONAL CONTROL
# ------------------------------------------------------------------------------
direction_param = 0  # set this at runtime for direction change, etc.

joint_output_map = {
    "pos_frontleftflipper":  {"offset": 0.8, "gain": 3.0},
    "pos_frontrightflipper": {"offset":  0.8, "gain": 3.0},
    "pos_backleft":          {"offset": 0.5, "gain": 1.0},
    "pos_backright":         {"offset":  0.5, "gain": 1.0},
    "pos_frontlefthip":      {"offset":  0.3, "gain": 1.0},
    "pos_frontrighthip":     {"offset":  0.3, "gain":  1.0}
}

def map_cpg_to_joint(x_i, y_i, direction, offset, gain):
    """
    A simple directional mapping:
      x_dir = x_i*cos(direction) - y_i*sin(direction)
      angle = offset + gain * x_dir
    """
    cosd = np.cos(direction)
    sind = np.sin(direction)
    x_dir = x_i*cosd - y_i*sind
    angle_raw = offset + gain*x_dir
    return angle_raw

# ------------------------------------------------------------------------------
#   SELECT BODY FOR COM / ORIENTATION
# ------------------------------------------------------------------------------
main_body_name = "base"
try:
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)
except Exception as e:
    print("Could not find body named", main_body_name, ":", e)
    main_body_id = 0  # fallback

total_mass = np.sum(model.body_mass)

# ------------------------------------------------------------------------------
#   DATA COLLECTION
# ------------------------------------------------------------------------------
time_data = []
ctrl_data = {name: [] for name in actuator_names}
com_positions = []
body_orientations = []
power_consumption = []
run_time = 30.0

# We'll store sensor data
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
#                       MAIN SIMULATION LOOP
# =============================================================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    dt_cpg = 0.001
    last_loop_time = time.time()

    while viewer.is_running():
        now = time.time()
        loop_dt = now - last_loop_time
        last_loop_time = now

        sim_time = now - start_time
        if sim_time >= run_time:
            print(f"Reached {run_time:.1f} seconds of simulation. Stopping now for analysis.")
            break

        # 1) Integrate Hopf with phase coupling
        steps = int(np.floor(loop_dt / dt_cpg))
        for _ in range(steps):
            x_all = [oscillators[name]["x"] for name in actuator_names]
            y_all = [oscillators[name]["y"] for name in actuator_names]

            for i, name in enumerate(actuator_names):
                x_i = oscillators[name]["x"]
                y_i = oscillators[name]["y"]

                # frequency blending
                freq = (stance_freq / (1.0 + np.exp(-a_param*y_i))) + \
                       (swing_freq  / (1.0 + np.exp( a_param*y_i)))

                x_new, y_new = hopf_step_with_phase(
                    x_i, y_i,
                    alpha, mu, freq, dt_cpg,
                    x_all, y_all, i,
                    K, phase_matrix
                )
                oscillators[name]["x"] = x_new
                oscillators[name]["y"] = y_new

        # 2) Convert oscillator states to joint angles
        for name in actuator_names:
            x_i = oscillators[name]["x"]
            y_i = oscillators[name]["y"]

            cfg = joint_output_map[name]
            offset = cfg["offset"]
            gain   = cfg["gain"]

            angle_raw = map_cpg_to_joint(x_i, y_i, direction_param, offset, gain)
            min_angle, max_angle = joint_limits[name]
            angle_clamped = np.clip(angle_raw, min_angle, max_angle)

            data.ctrl[actuator_indices[name]] = angle_clamped

        # 3) Step simulation
        mujoco.mj_step(model, data)

        # Store data
        time_data.append(sim_time)
        for name in actuator_names:
            ctrl_data[name].append(data.ctrl[actuator_indices[name]])

        # (a) COM
        com_pos = data.xpos[main_body_id].copy()
        com_positions.append(com_pos)

        # (b) Orientation
        orientation_mat = data.xmat[main_body_id].copy()
        body_orientations.append(orientation_mat)

        # (c) Power usage
        qvel = data.qvel[:model.nu]
        torque = data.actuator_force[:model.nu]
        instant_power = np.sum(np.abs(torque) * np.abs(qvel))
        power_consumption.append(instant_power)

        # (d) Joint torque sensors
        for varname, sname in jointact_sensor_map.items():
            val = get_sensor_data(data, model, sensor_name2id, sname)
            if val is not None:
                sensor_data_history[varname].append(val[0])

        # (e) Actuator torques
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
print(f"In-Phase coupling: {in_phase_coupling}")
print(f"Out-of-Phase coupling: {out_phase_coupling}")
print(f"Total robot mass: {total_mass:.2f} kg")

# COM displacement
if len(com_positions) > 1:
    displacement = com_positions[-1] - com_positions[0]
    distance_traveled = np.linalg.norm(displacement)
    print(f"Total displacement of COM: {displacement}")
    print(f"Straight-line distance traveled: {distance_traveled:.3f} m")
    avg_velocity = distance_traveled / final_time if final_time > 0 else 0
    print(f"Approx average speed: {avg_velocity:.3f} m/s")

# Energy
dt_integration = dt_cpg
if len(power_consumption) > 1:
    total_energy = np.sum(power_consumption)*dt_integration
    print(f"Approx total energy consumed: {total_energy:.3f} J")
    if distance_traveled > 0.01:
        weight = total_mass*9.81
        cost_of_transport = total_energy / (weight * distance_traveled)
        print(f"Approx cost of transport: {cost_of_transport:.3f}")

# Orientation
if len(body_orientations) > 0:
    final_orientation = body_orientations[-1]
    print(f"Final orientation matrix: \n{final_orientation}")

# Sensor data (final)
for varname in jointact_sensor_map:
    if sensor_data_history[varname]:
        last_val = sensor_data_history[varname][-1]
        print(f"Final torque sensor reading for {varname}: {last_val:.3f}")

# Actuator torques
print("\n=== Final Actuator Torque Values ===")
for name in actuator_names:
    idx = actuator_indices[name]
    torque_value = data.actuator_force[idx]
    print(f"{name}: {torque_value:.3f} Nm")

total_actuator_torques = {}
for name in actuator_names:
    total_actuator_torques[name] = np.sum(np.abs(actuator_torque_history[name]))*dt_integration
print("\n=== Total Integrated Actuator Torques (abs) ===")
for name, tot in total_actuator_torques.items():
    print(f"{name}: {tot:.3f} Nm·s")

# --------------------------- PLOTS ---------------------------------
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# (0,0) : Control Signals
for name in actuator_names:
    axs[0, 0].plot(time_data, ctrl_data[name], label=name)
axs[0, 0].set_title("Actuator Control Signals")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Joint Angle (rad)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# (0,1) : COM Position vs Time
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

# (1,0) : Instantaneous Power
axs[1, 0].plot(time_data, power_consumption, label="Instant Power")
axs[1, 0].set_title("Instantaneous Power Consumption")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Power (W)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# (1,1) : Trajectory (X vs Y)
if len(com_arr) > 0:
    axs[1, 1].plot(com_arr[:, 0], com_arr[:, 1], 'b-', label="Trajectory")
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
