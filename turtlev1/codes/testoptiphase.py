from bayes_opt import BayesianOptimization
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
    r_sq = x*x + y*y 
    
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
    "pos_frontleftflipper":  (-0.7, 1.571), 
    "pos_frontrightflipper": (-0.7, 1.571), 
    "pos_backleft":          (-0.7, 1.571), 
    "pos_backright":         (-0.7, 1.571), 
    "pos_frontlefthip":      (-0.7, 1.571), 
    "pos_frontrighthip":     (-0.7, 1.571) 
}

# ----------------------------------------------------------------------------- 
#        CPG PARAMETERS & INITIAL OSCILLATOR STATES 
# ----------------------------------------------------------------------------- 
alpha    = 10.0   # Gain, convergence speed 
mu       = 0.04   # radius^2 => amplitude ~ sqrt(0.04) = 0.2 
a_param  = 10.0   # logistic steepness for stance/swing freq blending

# We define separate stance & swing frequencies (rad/s) 
stance_freq = 2.0 
swing_freq  = 2.0

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
        cross_side = (i in left_indices and j in right_indices) or \
                     (i in right_indices and j in left_indices)
        
        if both_left or both_right:
            K[i, j] = in_phase_coupling
        elif cross_side:
            K[i, j] = out_phase_coupling

# ----------------------------------------------------------------------------- 
#  PHASE OFFSETS & LINEAR MAPPING FROM (x,y) -> JOINT ANGLE 
# ----------------------------------------------------------------------------- 
phase_offsets_diagforward = { 
    "pos_frontleftflipper":  0, 
    "pos_frontrightflipper":  -np.pi,      # opposite phase relative to front left 
    "pos_backleft":          0,       # in-phase with front right 
    "pos_backright":            -np.pi,  # opposite phase relative to front right 
    "pos_frontlefthip":      np.pi/2, 
    "pos_frontrighthip":      -np.pi/2 
}

joint_output_map = { 
    "pos_frontleftflipper":  {"offset": 0.436, "gain": 1.136}, 
    "pos_frontrightflipper": {"offset": 0.436, "gain": 1.136}, 
    "pos_backleft":          {"offset": 0.436, "gain": 1.136}, 
    "pos_backright":         {"offset": 0.436, "gain": 1.136}, 
    "pos_frontlefthip":      {"offset": 0.436, "gain": 1.136}, 
    "pos_frontrighthip":     {"offset": 0.436, "gain": 1.136} 
}

# Assuming actuator order is consistent with your actuator_names list: 
joint_limits = {} 
for i, name in enumerate(actuator_names): 
    # Get the min and max from the control range 
    ctrl_min, ctrl_max = model.actuator_ctrlrange[i] 
    offset = (ctrl_min + ctrl_max) / 2 
    gain = (ctrl_max - ctrl_min) / 2 
    joint_limits[name] = (ctrl_min, ctrl_max) 
    print(f"{name}: offset = {offset:.3f}, gain = {gain:.3f}")

# ----------------------------------------------------------------------------- 
#  SELECT WHICH BODY TO TRACK FOR COM, ORIENTATION, ETC. 
# ----------------------------------------------------------------------------- 
main_body_name = "base"  # or "base", "torso", etc., depending on your MJCF 
try: 
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name) 
except: 
    main_body_id = 0  # fallback to root body

# Total mass (approx. sum of all body masses) 
total_mass = np.sum(model.body_mass)

# ----------------------------------------------------------------------------- 
#                       DATA COLLECTION FOR PLOTTING 
# ----------------------------------------------------------------------------- 
time_data = [] 
ctrl_data = {name: [] for name in actuator_names}

# ----------------------------------------------------------------------------- 
#    PERFORMANCE METRICS + (OPTIONAL) SENSOR DATA STORAGE 
# ----------------------------------------------------------------------------- 
com_positions = [] 
body_orientations = [] 
power_consumption = [] 
time_30s = 30.0  # how long we run

# We'll store the joint torque sensors, plus base accelerometer/gyro data 
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

# The sensor names in the XML: 
jointact_sensor_map = { 
    "torque_backright":       "sens_jointactfrc_backright", 
    "torque_backleft":        "sens_jointactfrc_backleft", 
    "torque_frontrighthip":   "sens_jointactfrc_frontrighthip", 
    "torque_frontrightflipper":"sens_jointactfrc_frontrightflipper", 
    "torque_frontlefthip":    "sens_jointactfrc_frontlefthip", 
    "torque_frontleftflipper":"sens_jointactfrc_frontleftflipper" 
}

base_imu_map = { 
    "base_gyro": "sens_base_gyro",  # frameangvel 
    "base_acc":  "sens_base_acc"    # framelinacc 
}

# ----------------------------------------------------------------------------- 
#                       MAIN SIMULATION LOOP 
# ----------------------------------------------------------------------------- 
def run_simulation_and_get_performance(phase_offsets):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        dt_cpg = 0.001
        last_loop_time = time.time()

        while viewer.is_running():
            now = time.time()
            loop_dt = now - last_loop_time
            last_loop_time = now
            
            sim_time = now - start_time

            # -----------------------------------------------------------------
            # Stop after 30 seconds to gather final metrics
            # -----------------------------------------------------------------
            if sim_time >= time_30s:
                print(f"Reached {time_30s:.0f} seconds of simulation. Stopping now for analysis.")
                break
            
            # -----------------------------------------------------------------
            # 1) Integrate the Hopf oscillators
            # -----------------------------------------------------------------
            steps = int(np.floor(loop_dt / dt_cpg))
            
            for _ in range(steps):
                x_all = [oscillators[name]["x"] for name in actuator_names]
                y_all = [oscillators[name]["y"] for name in actuator_names]

                for i, name in enumerate(actuator_names):
                    x_i = oscillators[name]["x"]
                    y_i = oscillators[name]["y"]
                    
                    # Frequency blending
                    freq = (stance_freq / (1.0 + np.exp(-a_param*y_i))) + \
                           (swing_freq  / (1.0 + np.exp( a_param*y_i)))

                    x_new, y_new = hopf_step(
                        x_i, y_i, alpha, mu, freq, dt_cpg,
                        K, x_all, y_all, i
                    )
                    oscillators[name]["x"] = x_new
                    oscillators[name]["y"] = y_new

            # -----------------------------------------------------------------
            # 2) Convert oscillator states to joint angles & apply control
            # -----------------------------------------------------------------
            for name in actuator_names:
                x_i = oscillators[name]["x"]
                y_i = oscillators[name]["y"]

                delta_phi = phase_offsets[name]
                x_phase = x_i * np.cos(delta_phi) - y_i * np.sin(delta_phi)

                offset = joint_output_map[name]["offset"]
                gain   = joint_output_map[name]["gain"]
                angle_raw = offset + gain * x_phase

                min_angle, max_angle = joint_limits[name]
                angle_clamped = np.clip(angle_raw, min_angle, max_angle)
                
                data.ctrl[actuator_indices[name]] = angle_clamped
            
            # -----------------------------------------------------------------
            # 3) Step MuJoCo simulation
            # -----------------------------------------------------------------
            mujoco.mj_step(model, data)

            # Record time & controls
            time_data.append(sim_time)
            for name in actuator_names:
                ctrl_data[name].append(data.ctrl[actuator_indices[name]])
            
            # -----------------------------------------------------------------
            # 4) Collect data from existing fields
            # -----------------------------------------------------------------
            # (a) COM position
            com_pos = data.xpos[main_body_id].copy()
            com_positions.append(com_pos)
            
            # (b) Orientation (using rotation matrix)
            orientation_mat = data.xmat[main_body_id].reshape(3, 3).copy()
            body_orientations.append(orientation_mat)

            # (c) Approx mechanical power usage:
            # Using torque * velocity from data.actuator_force and data.qvel
            qvel = data.qvel[:model.nu]   
            torque = data.actuator_force[:model.nu]
            instant_power = np.sum(torque * qvel)
            power_consumption.append(instant_power)

            # -----------------------------------------------------------------
            # 5) (Optional) Collect sensor data from sensor_name2id
            # -----------------------------------------------------------------
            # - Retrieve joint torque sensors via <jointactuatorfrc>
            # - Retrieve base IMU (gyro & accelerometer)
            for varname, sname in jointact_sensor_map.items():
                val = get_sensor_data(data, model, sensor_name2id, sname)
                if val is not None:
                    sensor_data_history[varname].append(val[0])  # it's 1D

            # The IMU sensors output 3D arrays
            for varname, sname in base_imu_map.items():
                val = get_sensor_data(data, model, sensor_name2id, sname)
                if val is not None:
                    sensor_data_history[varname].append(val.copy())  # 3D

            viewer.sync()



    # -------------------------------------------------------------------------
    #  After the viewer closes or time_30s have elapsed, we analyze the metrics.
    # -------------------------------------------------------------------------
    final_time = time_data[-1] if len(time_data) > 0 else 0.0

    print("\n=== Performance Analysis ===")
    print(f"Simulation time recorded: {final_time:.2f} s")
    print(f"In Phase coupling value: {in_phase_coupling}")
    print(f"Out of Phase coupling value: {out_phase_coupling}")
    print(f"Total mass of robot: {total_mass:.2f} kg")

    # Center of mass (COM) positions over time
    if len(com_positions) > 1:
        # Displacement: difference between final and initial COM
        displacement = com_positions[-1] - com_positions[0]
        distance_traveled = np.linalg.norm(displacement)
        print(f"Total displacement of COM: {displacement}")
        print(f"Straight-line distance traveled: {distance_traveled:.3f} m")

        # Average velocity (approx distance / time)
        avg_velocity = distance_traveled / final_time if final_time > 0 else 0
        print(f"Approx average speed: {avg_velocity:.3f} m/s")

    dt_integration = dt_cpg  # the step we used for oscillator updates
    # Summation of mechanical power => total energy
    if len(power_consumption) > 1:
        total_energy = np.sum(power_consumption) * dt_integration
        print(f"Approx total energy consumed: {total_energy:.3f} J (assuming torque*velocity)")

        if distance_traveled > 0.01:
            weight = total_mass * 9.81
            cost_of_transport = total_energy / (weight * distance_traveled)
            print(f"Approx cost of transport: {cost_of_transport:.3f}")

    # Example orientation analysis: final orientation matrix or average?
    if len(body_orientations) > 0:
        final_orientation = body_orientations[-1]
        orientation_penalty = np.linalg.norm(final_orientation - np.eye(3))
        print(f"Final orientation matrix of main body: {final_orientation}")
        print(f"Orientation penalty: {orientation_penalty:.3f}")

    print("=== End of Analysis ===\n")

    # Return the performance metric (e.g., average speed)
    return avg_velocity, displacement, orientation_penalty

# ----------------------------------------------------------------------------- 
#                           PLOTTING AFTER SIMULATION 
# ----------------------------------------------------------------------------- 
plt.figure(figsize=(10, 6)) 
for name in actuator_names: 
    plt.plot(time_data, ctrl_data[name], label=name)

plt.title('Joint Control with Phase Offsets') 
plt.xlabel('Time (s)') 
plt.ylabel('Joint Angle (rad)') 
plt.legend() 
plt.grid(True) 
plt.tight_layout() 
plt.show()

def objective_function(pos_frontleftflipper, pos_frontrightflipper, pos_backleft, pos_backright, pos_frontlefthip, pos_frontrighthip): 
    # Update the phase offsets 
    phase_offsets_diagforward = { 
        "pos_frontleftflipper":  pos_frontleftflipper, 
        "pos_frontrightflipper": pos_frontrightflipper, 
        "pos_backleft":          pos_backleft, 
        "pos_backright":         pos_backright, 
        "pos_frontlefthip":      pos_frontlefthip, 
        "pos_frontrighthip":     pos_frontrighthip 
    }
    
    # Run the simulation and calculate the performance metric (e.g., forward speed) 
    avg_velocity, displacement, orientation_penalty = run_simulation_and_get_performance(phase_offsets_diagforward)
    
    # Penalize lateral y displacement and change in orientation
    penalty = np.abs(displacement[1]) + orientation_penalty
    performance_metric = avg_velocity - penalty
    
    return performance_metric

# Define the parameter bounds for the optimization
pbounds = {
    "pos_frontleftflipper":  (-np.pi, np.pi),
    "pos_frontrightflipper": (-np.pi, np.pi),
    "pos_backleft":          (-np.pi, np.pi),
    "pos_backright":         (-np.pi, np.pi),
    "pos_frontlefthip":      (-np.pi, np.pi),
    "pos_frontrighthip":     (-np.pi, np.pi)
}

# Initialize the Bayesian Optimizer
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=1
)

# Run the optimization
optimizer.maximize(
    init_points=5,
    n_iter=15
)

# Extract the best parameters
best_params = optimizer.max['params']
print("Optimal phase offsets found:", best_params)

# Update the phase offsets with the best parameters
phase_offsets_diagforward = {
    "pos_frontleftflipper":  best_params["pos_frontleftflipper"],
    "pos_frontrightflipper": best_params["pos_frontrightflipper"],
    "pos_backleft":          best_params["pos_backleft"],
    "pos_backright":         best_params["pos_backright"],
    "pos_frontlefthip":      best_params["pos_frontlefthip"],
    "pos_frontrighthip":     best_params["pos_frontrighthip"]
}

# Run the simulation with the optimal phase offsets and plot the results
final_performance = run_simulation_and_get_performance(phase_offsets_diagforward)
print(f"Final performance with optimal phase offsets: {final_performance:.3f} m/s")

# Plotting the results
plt.figure(figsize=(10, 6))
for name in actuator_names:
    plt.plot(time_data, ctrl_data[name], label=name)

plt.title('Joint Control with Optimal Phase Offsets')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
