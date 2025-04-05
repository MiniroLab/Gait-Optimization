import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient  # updated multi-objective BO package
from ax.utils.measurement.synthetic_functions import branin  # for reference; not used here

# -----------------------------------------------------------------------------
#                           HOPF OSCILLATOR DYNAMICS
# -----------------------------------------------------------------------------
def hopf_step(x, y, alpha, mu, omega, dt, coupling, xall, yall, index):
    """
    Integrate one step of the Hopf oscillator (with coupling) using Euler's method.
    """
    r_sq = x * x + y * y
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
#                         MUJOCO HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_sensor_data(data, model, sensor_name2id, sname):
    """Retrieve sensor reading for the given sensor name."""
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
#                         LOAD MUJOCO MODEL
# -----------------------------------------------------------------------------
model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/turtlev1/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)

# Build sensor lookup from XML names
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
    "pos_frontleftflipper":  (-1.57,  0.64),
    "pos_frontrightflipper": (-0.64,  1.571),
    "pos_backleft":          (-1.571, 0.524),
    "pos_backright":         (-0.524, 1.571),
    "pos_frontlefthip":      (-1.571, 1.22),
    "pos_frontrighthip":     (-1.22,  1.571)
}

# -----------------------------------------------------------------------------
#        DEFAULT CPG PARAMETERS (to be tuned by optimization)
# -----------------------------------------------------------------------------
alpha_default = 10.0  # Convergence speed of oscillator
mu_default    = 0.04  # Amplitude factor (radius^2)

# -----------------------------------------------------------------------------
#  PHASE OFFSETS & LINEAR MAPPING FROM (x,y) -> JOINT ANGLE
# -----------------------------------------------------------------------------
phase_offsets_diagforward = {
    "pos_frontleftflipper":  -np.pi,
    "pos_frontrightflipper":  0.0,
    "pos_backleft":           0.0,
    "pos_backright":          np.pi,
    "pos_frontlefthip":      -np.pi/2,
    "pos_frontrighthip":      np.pi/2
}

joint_output_map = {
    "pos_frontleftflipper":  {"offset": -0.8, "gain": 3.0},
    "pos_frontrightflipper": {"offset":  0.8, "gain": 3.0},
    "pos_backleft":          {"offset": -0.5, "gain": 1.0},
    "pos_backright":         {"offset":  0.5, "gain": 1.0},
    "pos_frontlefthip":      {"offset":  0.3, "gain": -1.0},
    "pos_frontrighthip":     {"offset":  0.3, "gain":  1.0}
}

# -----------------------------------------------------------------------------
#        BODY & MASS PARAMETERS
# -----------------------------------------------------------------------------
main_body_name = "base"  # Body for center-of-mass (COM) and orientation measurements
try:
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)
except Exception as e:
    print("Could not find body named", main_body_name, ":", e)
    main_body_id = 0

total_mass = np.sum(model.body_mass)

# -----------------------------------------------------------------------------
#                  SIMULATION SETTINGS (for headless evaluation)
# -----------------------------------------------------------------------------
run_time = 30.0     # Duration (in seconds) for each simulation evaluation
dt_cpg   = 0.001    # Integration timestep for oscillator dynamics

# -----------------------------------------------------------------------------
#          NEW OBJECTIVE FUNCTION FOR MULTI-OBJECTIVE OPTIMIZATION
# -----------------------------------------------------------------------------
def simulate_objectives(a_param_opt, stance_freq_opt, swing_freq_opt, in_phase_coupling_opt, out_phase_coupling_opt):
    """
    Runs a headless simulation using provided CPG parameters and returns two objectives:
      1. A velocity reward = (forward speed) - (penalty * lateral drift speed)
         where forward speed is computed from COM x-displacement and lateral drift
         is computed from the absolute COM y-displacement.
      2. The cost of transport (CoT) computed as total energy consumption divided by (weight * distance traveled).
    
    Both objectives are computed over a run_time period.
    """
    # Map optimizer parameters to simulation parameters
    a_param = a_param_opt
    stance_freq = stance_freq_opt
    swing_freq  = swing_freq_opt

    # Recompute coupling matrix K based on optimized coupling strengths
    num_joints = len(actuator_names)
    left_indices  = [0, 2, 4]
    right_indices = [1, 3, 5]
    K = np.zeros((num_joints, num_joints))
    for i in range(num_joints):
        for j in range(num_joints):
            if i == j:
                continue
            both_left  = (i in left_indices) and (j in left_indices)
            both_right = (i in right_indices) and (j in right_indices)
            cross_side = (i in left_indices and j in right_indices) or (i in right_indices and j in left_indices)
            if both_left or both_right:
                K[i, j] = in_phase_coupling_opt
            elif cross_side:
                K[i, j] = out_phase_coupling_opt

    # Initialize simulation data and oscillator states for a new evaluation
    data = mujoco.MjData(model)
    oscillators = {}
    for name in actuator_names:
        oscillators[name] = {"x": 0.02 * np.random.randn(), "y": 0.02 * np.random.randn()}

    # Record COM positions (we use x and y for forward and lateral displacement)
    com_positions = []
    com_positions.append(data.xpos[main_body_id].copy())

    # For energy consumption
    power_consumption = []
    sim_steps = int(run_time / dt_cpg)
    
    for step in range(sim_steps):
        # 1) Integrate the Hopf oscillators
        x_all = [oscillators[name]["x"] for name in actuator_names]
        y_all = [oscillators[name]["y"] for name in actuator_names]
        for i, name in enumerate(actuator_names):
            x_i = oscillators[name]["x"]
            y_i = oscillators[name]["y"]
            # Logistic blending of stance and swing frequencies
            freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                   (swing_freq  / (1.0 + np.exp(a_param * y_i)))
            x_new, y_new = hopf_step(x_i, y_i, alpha_default, mu_default, freq, dt_cpg, K, x_all, y_all, i)
            oscillators[name]["x"] = x_new
            oscillators[name]["y"] = y_new

        # 2) Map oscillator states to joint commands and apply to the robot
        for name in actuator_names:
            x_i = oscillators[name]["x"]
            y_i = oscillators[name]["y"]
            delta_phi = phase_offsets_diagforward[name]
            x_phase = x_i * np.cos(delta_phi) - y_i * np.sin(delta_phi)
            offset = joint_output_map[name]["offset"]
            gain   = joint_output_map[name]["gain"]
            angle_raw = offset + gain * x_phase
            min_angle, max_angle = joint_limits[name]
            angle_clamped = np.clip(angle_raw, min_angle, max_angle)
            data.ctrl[actuator_indices[name]] = angle_clamped

        # 3) Advance the simulation
        mujoco.mj_step(model, data)

        # 4) Record the COM position
        com_positions.append(data.xpos[main_body_id].copy())

        # 5) Record instantaneous power consumption (torque*velocity)
        qvel = data.qvel[:model.nu]
        torque = data.actuator_force[:model.nu]
        power = np.sum(np.abs(torque) * np.abs(qvel))
        power_consumption.append(power)

    # Compute forward and lateral displacement (using first and last COM positions)
    initial_pos = com_positions[0]
    final_pos = com_positions[-1]
    forward_disp = final_pos[0] - initial_pos[0]  # x-axis displacement
    lateral_disp = np.abs(final_pos[1] - initial_pos[1])  # absolute y-axis displacement
    avg_forward_speed = forward_disp / run_time
    avg_lateral_speed = lateral_disp / run_time

    # Define a penalty weight for lateral drift (this value can be tuned)
    lateral_penalty_weight = 0.5
    velocity_reward = avg_forward_speed - lateral_penalty_weight * avg_lateral_speed

    # Energy consumption & cost of transport (CoT)
    total_energy = np.sum(power_consumption) * dt_cpg  # approximate total energy in Joules
    distance_traveled = np.linalg.norm(final_pos - initial_pos)
    weight_force = total_mass * 9.81
    cot = total_energy / (weight_force * distance_traveled) if distance_traveled > 0 else np.inf

    # In multi-objective optimization we wish to maximize velocity_reward and minimize CoT.
    # (Ax allows you to specify objective directions.)
    return {"velocity_reward": velocity_reward, "cot": cot}

# -----------------------------------------------------------------------------
#         MULTI-OBJECTIVE BAYESIAN OPTIMIZATION WITH AX
# -----------------------------------------------------------------------------
# Define parameter bounds for optimization
parameters = [
    {"name": "a_param_opt", "type": "range", "bounds": [5, 25]},
    {"name": "stance_freq_opt", "type": "range", "bounds": [1.0, 8.0]},
    {"name": "swing_freq_opt", "type": "range", "bounds": [1.0, 8.0]},
    {"name": "in_phase_coupling_opt", "type": "range", "bounds": [0.2, 1.5]},
    {"name": "out_phase_coupling_opt", "type": "range", "bounds": [-1.5, -0.2]}
]

# Create an AxClient for multi-objective optimization
ax_client = AxClient(enforce_sequential_optimization=False)
ax_client.create_experiment(
    name="turtle_robot_multiobj",
    parameters=parameters,
    objective_name="velocity_reward",  # dummy placeholder (will be overridden below)
    minimize=False,  # We want to maximize velocity_reward
    outcome_constraints=[{"name": "cot", "op": "<=", "bound": 1.0}],  # example constraint if desired
    objectives={
        "velocity_reward": {"minimize": False},
        "cot": {"minimize": True}
    },
    total_trials=30,
)

# Run optimization loop
for i in range(30):
    parameters, trial_index = ax_client.get_next_trial()
    # Evaluate simulation with the current parameters
    res = simulate_objectives(**parameters)
    # Report both objectives to Ax.
    ax_client.complete_trial(trial_index=trial_index, raw_data=res)
    print(f"Trial {i}: params: {parameters}, results: {res}")

# Get Pareto frontier (the AxClient provides a data object for further analysis)
# Here we simply pick one candidate that gives a balanced trade-off.
# For demonstration, we choose the candidate with the highest velocity_reward among those with cot below median.
data = ax_client.experiment.fetch_data()
# For simplicity, extract all trial parameters and outcomes.
results = ax_client.get_trials_data_frame()
print("\n=== Optimization Results ===")
print(results)

# For the final simulation, select the best candidate (here we choose the one with the maximum velocity_reward)
# You may choose another selection strategy based on your trade-off preference.
best_idx = results["velocity_reward"].idxmax()
best_params = results.loc[best_idx, ["a_param_opt", "stance_freq_opt", "swing_freq_opt",
                                       "in_phase_coupling_opt", "out_phase_coupling_opt"]].to_dict()
print("\n=== Selected Optimal Parameters for Final Simulation ===")
for key, val in best_params.items():
    print(f"{key}: {val}")

# -----------------------------------------------------------------------------
#        FINAL SIMULATION WITH OPTIMAL PARAMETERS (WITH VIEWER)
# -----------------------------------------------------------------------------
a_param = best_params['a_param_opt']
stance_freq = best_params['stance_freq_opt']
swing_freq  = best_params['swing_freq_opt']
in_phase_coupling = best_params['in_phase_coupling_opt']
out_phase_coupling = best_params['out_phase_coupling_opt']

# Recompute the coupling matrix K with optimal parameters
num_joints = len(actuator_names)
left_indices  = [0, 2, 4]
right_indices = [1, 3, 5]
K = np.zeros((num_joints, num_joints))
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

# Reset simulation data and oscillator states for final run
data = mujoco.MjData(model)
oscillators = {}
for name in actuator_names:
    oscillators[name] = {"x": 0.02 * np.random.randn(), "y": 0.02 * np.random.randn()}

# -----------------------------------------------------------------------------
#                    DATA COLLECTION FOR PLOTTING (Final Run)
# -----------------------------------------------------------------------------
time_data = []
ctrl_data = {name: [] for name in actuator_names}
com_positions = []        
body_orientations = []
power_consumption = []
actuator_torque_history = {name: [] for name in actuator_names}
joint_velocity_history = {name: [] for name in actuator_names}

# Sensor names for joint torque sensors defined in the XML
jointact_sensor_map = {
    "torque_backright":       "sens_jointactfrc_backright",
    "torque_backleft":        "sens_jointactfrc_backleft",
    "torque_frontrighthip":   "sens_jointactfrc_frontrighthip",
    "torque_frontrightflipper":"sens_jointactfrc_frontrightflipper",
    "torque_frontlefthip":    "sens_jointactfrc_frontlefthip",
    "torque_frontleftflipper":"sens_jointactfrc_frontleftflipper"
}
sensor_data_history = {key: [] for key in jointact_sensor_map.keys()}

# -----------------------------------------------------------------------------
#                       MAIN SIMULATION LOOP (with viewer)
# -----------------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    last_loop_time = time.time()
    while viewer.is_running():
        now = time.time()
        loop_dt = now - last_loop_time
        last_loop_time = now
        sim_time = now - start_time

        # Terminate simulation after 30 seconds of real-time
        if sim_time >= 30.0:
            print("Reached simulation runtime. Exiting simulation loop.")
            break

        # Integrate oscillators using fixed dt_cpg steps
        steps = int(np.floor(loop_dt / dt_cpg))
        for _ in range(steps):
            x_all = [oscillators[name]["x"] for name in actuator_names]
            y_all = [oscillators[name]["y"] for name in actuator_names]
            for i, name in enumerate(actuator_names):
                x_i = oscillators[name]["x"]
                y_i = oscillators[name]["y"]
                freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                       (swing_freq / (1.0 + np.exp(a_param * y_i)))
                x_new, y_new = hopf_step(x_i, y_i, alpha_default, mu_default, freq, dt_cpg,
                                         K, x_all, y_all, i)
                oscillators[name]["x"] = x_new
                oscillators[name]["y"] = y_new

        # Map oscillator states to joint angles and update controls
        for name in actuator_names:
            x_i = oscillators[name]["x"]
            y_i = oscillators[name]["y"]
            delta_phi = phase_offsets_diagforward[name]
            x_phase = x_i * np.cos(delta_phi) - y_i * np.sin(delta_phi)
            offset = joint_output_map[name]["offset"]
            gain   = joint_output_map[name]["gain"]
            angle_raw = offset + gain * x_phase
            min_angle, max_angle = joint_limits[name]
            angle_clamped = np.clip(angle_raw, min_angle, max_angle)
            data.ctrl[actuator_indices[name]] = angle_clamped

        # Advance the simulation
        mujoco.mj_step(model, data)

        # Record metrics for plotting
        time_data.append(sim_time)
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
        for name in actuator_names:
            idx = actuator_indices[name]
            actuator_torque_history[name].append(data.actuator_force[idx])
            joint_velocity_history[name].append(qvel[idx])
        viewer.sync()

# -----------------------------------------------------------------------------
#           PERFORMANCE ANALYSIS AFTER FINAL SIMULATION RUN
# -----------------------------------------------------------------------------
final_time = time_data[-1] if len(time_data) > 0 else 0.0

print("\n=== Performance Analysis ===")
print(f"Simulation time recorded: {final_time:.2f} s")
print(f"Total mass of robot: {total_mass:.2f} kg")
print(f"In Phase coupling value: {in_phase_coupling}")
print(f"Out of Phase coupling value: {out_phase_coupling}")
print(f"a_param value: {a_param}")
print(f"Stance Frequency: {stance_freq}")
print(f"Swing Frequency: {swing_freq}")

if len(com_positions) > 1:
    initial_pos = com_positions[0]
    final_pos = com_positions[-1]
    displacement_vector = final_pos - initial_pos
    distance_traveled = np.linalg.norm(displacement_vector)
    forward_disp = final_pos[0] - initial_pos[0]
    lateral_disp = np.abs(final_pos[1] - initial_pos[1])
    print(f"Total COM displacement: {displacement_vector}")
    print(f"Straight-line distance traveled: {distance_traveled:.3f} m")
    avg_speed = distance_traveled / final_time if final_time > 0 else 0
    print(f"Approx average speed: {avg_speed:.3f} m/s")
    print(f"Forward speed: {forward_disp/run_time:.3f} m/s")
    print(f"Lateral drift speed: {lateral_disp/run_time:.3f} m/s")

if len(power_consumption) > 1:
    dt_integration = dt_cpg
    total_energy = np.sum(power_consumption) * dt_integration
    print(f"Approx total energy consumed: {total_energy:.3f} J (torque*velocity)")
    if distance_traveled > 0.01:
        weight_force = total_mass * 9.81
        cost_of_transport = total_energy / (weight_force * distance_traveled)
        print(f"Approx cost of transport: {cost_of_transport:.3f}")

# Compute total (integrated absolute) actuator torques over time
total_actuator_torques = {}
for name in actuator_names:
    total_actuator_torques[name] = np.sum(np.abs(actuator_torque_history[name])) * dt_cpg

print("\n=== Total Actuator Torques Over Time (integrated absolute torque) ===")
for name, tot in total_actuator_torques.items():
    print(f"{name}: {tot:.3f} Nm·s")

# -----------------------------------------------------------------------------
#                PLOTTING: Create Subplots for Collected Metrics
# -----------------------------------------------------------------------------
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

# Subplot (1,0): Instantaneous Power Consumption
axs[1, 0].plot(time_data, power_consumption, label="Instant Power")
axs[1, 0].set_title("Instantaneous Power Consumption")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Power (W)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Subplot (1,1): Trajectory (COM X vs COM Y)
axs[1, 1].plot(com_positions_arr[:, 0], com_positions_arr[:, 1], 'b-', label="Trajectory")
axs[1, 1].set_title("Trajectory (X vs Y)")
axs[1, 1].set_xlabel("X (m)")
axs[1, 1].set_ylabel("Y (m)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Subplot (2,0): Actuator Torque Over Time
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
