import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

# =============================================================================
# 1. LOAD MUJOCO MODEL & HELPER FUNCTIONS
# =============================================================================
model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/turtlev1/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

def get_actuator_index(model, name):
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

# Use the robot's "base" body center-of-mass as our reference
try:
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
except Exception:
    main_body_id = 0
total_mass = np.sum(model.body_mass)

# =============================================================================
# 2. HOPF OSCILLATOR DYNAMICS & GAIT CONFIGURATION
# =============================================================================
def hopf_step(x, y, alpha, mu, omega, dt, coupling, xall, yall, index):
    """One Euler integration step for a Hopf oscillator with coupling."""
    r_sq = x*x + y*y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x
    # Add linear coupling from other oscillators
    for j in range(len(xall)):
        if j == index:
            continue
        K_ij = coupling[index, j]
        dx += K_ij * (xall[j] - x)
        dy += K_ij * (yall[j] - y)
    x_new = x + dx * dt
    y_new = y + dy * dt
    return x_new, y_new

# Actuator names
actuator_names = [
    "pos_frontleftflipper", "pos_frontrightflipper",
    "pos_backleft", "pos_backright",
    "pos_frontlefthip", "pos_frontrighthip"
]
actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

# Joint limits
joint_limits = {
    "pos_frontleftflipper":  (-1.57,  0.64),
    "pos_frontrightflipper": (-0.64,  1.571),
    "pos_backleft":          (-1.571, 0.524),
    "pos_backright":         (-0.524, 1.571),
    "pos_frontlefthip":      (-1.571, 1.22),
    "pos_frontrighthip":     (-1.22,  1.571)
}

# Default diagonal-forward coupling matrix
num_joints = len(actuator_names)
K_default = np.zeros((num_joints, num_joints))
in_phase_coupling  = 0.8
out_phase_coupling = -0.8
left_indices  = [0, 2, 4]  # frontleftflipper, backleft, frontlefthip
right_indices = [1, 3, 5]  # frontrightflipper, backright, frontrighthip
for i in range(num_joints):
    for j in range(num_joints):
        if i == j:
            continue
        if (i in left_indices and j in left_indices) or (i in right_indices and j in right_indices):
            K_default[i, j] = in_phase_coupling
        else:
            K_default[i, j] = out_phase_coupling

# Phase offsets for diagonal-forward gait
phase_offsets_diagforward = {
    "pos_frontleftflipper":  -np.pi,
    "pos_frontrightflipper":  0.0,
    "pos_backleft":           0.0,
    "pos_backright":          np.pi,
    "pos_frontlefthip":      -np.pi/2,
    "pos_frontrighthip":      np.pi/2
}

# Mapping from oscillator outputs (x,y) to joint angles
joint_output_map = {
    "pos_frontleftflipper":  {"offset": -0.8, "gain": 3.0},
    "pos_frontrightflipper": {"offset":  0.8, "gain": 3.0},
    "pos_backleft":          {"offset": -0.5, "gain": 1.0},
    "pos_backright":         {"offset":  0.5, "gain": 1.0},
    "pos_frontlefthip":      {"offset":  0.3, "gain": -1.0},
    "pos_frontrighthip":     {"offset":  0.3, "gain":  1.0}
}

# =============================================================================
# 3. SIMULATION FOR SPEED (Forward Locomotion)
# =============================================================================
def simulate_for_speed(params, sim_duration=30.0, dt=0.001):
    """
    Simulate the Hopf CPG with tunable frequency/coupling and return the forward speed.
    params: [alpha, mu, a_param, in_phase, out_phase, stance_freq, swing_freq]
    """
    alpha_opt, mu_opt, a_param_opt, in_phase_val, out_phase_val, stance_freq_opt, swing_freq_opt = params
    
    # Construct a new coupling matrix using the user-provided in/out-phase values
    K = np.zeros((num_joints, num_joints))
    for i in range(num_joints):
        for j in range(num_joints):
            if i == j:
                continue
            if (i in left_indices and j in left_indices) or (i in right_indices and j in right_indices):
                K[i, j] = in_phase_val
            else:
                K[i, j] = out_phase_val

    # Initialize oscillator states
    oscillators = {}
    for name in actuator_names:
        oscillators[name] = {"x": 0.02 * np.random.randn(), "y": 0.02 * np.random.randn()}
    
    mujoco.mj_resetData(model, data)
    
    # Data for measuring speed
    com_positions = []
    init_com = data.xpos[main_body_id].copy()
    steps = int(sim_duration / dt)
    
    for _ in range(steps):
        x_all = [oscillators[name]["x"] for name in actuator_names]
        y_all = [oscillators[name]["y"] for name in actuator_names]
        
        # Update oscillators
        for i, name in enumerate(actuator_names):
            x_i = oscillators[name]["x"]
            y_i = oscillators[name]["y"]
            # Frequency blending
            freq = (stance_freq_opt / (1.0 + np.exp(-a_param_opt * y_i))) + \
                   (swing_freq_opt  / (1.0 + np.exp(a_param_opt * y_i)))
            x_new, y_new = hopf_step(x_i, y_i, alpha_opt, mu_opt, freq, dt, K, x_all, y_all, i)
            oscillators[name]["x"], oscillators[name]["y"] = x_new, y_new
        
        # Convert oscillator states to joint angles
        for name in actuator_names:
            x_i = oscillators[name]["x"]
            y_i = oscillators[name]["y"]
            phi_offset = phase_offsets_diagforward[name]
            x_phase = x_i * np.cos(phi_offset) - y_i * np.sin(phi_offset)
            
            offset = joint_output_map[name]["offset"]
            gain   = joint_output_map[name]["gain"]
            angle_raw = offset + gain * x_phase
            
            min_angle, max_angle = joint_limits[name]
            angle_clamped = np.clip(angle_raw, min_angle, max_angle)
            
            data.ctrl[get_actuator_index(model, name)] = angle_clamped
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Log COM position
        com_positions.append(data.xpos[main_body_id].copy())
    
    # Compute distance traveled & average speed
    final_com = com_positions[-1]
    displacement = final_com - init_com
    distance_traveled = np.linalg.norm(displacement)
    
    if distance_traveled < 0.001:  # If minimal movement occurred
        return 0.0
    else:
        return distance_traveled / sim_duration

# =============================================================================
# 4. BAYESIAN OPT FUNCTION (Speed-Focused)
# =============================================================================
def cpg_blackbox_speed(alpha_opt, mu_opt, a_param_opt, in_phase_val, out_phase_val, stance_freq_opt, swing_freq_opt):
    # Bundle parameters
    params = [alpha_opt, mu_opt, a_param_opt, in_phase_val, out_phase_val, stance_freq_opt, swing_freq_opt]
    # We measure forward speed
    speed = simulate_for_speed(params, sim_duration=30.0, dt=0.001)
    
    print(f"Params: alpha={alpha_opt:.3f}, mu={mu_opt:.3f}, a={a_param_opt:.3f}, "
          f"in-phase={in_phase_val:.3f}, out-phase={out_phase_val:.3f}, "
          f"stance={stance_freq_opt:.3f}, swing={swing_freq_opt:.3f} -> speed: {speed:.6f} m/s")
    
    # Return negative speed so that BayesianOptimization maximizes the objective
    return speed

# =============================================================================
# 5. SETUP BAYESIAN OPTIMIZATION FOR SPEED
# =============================================================================
pbounds = {
    'alpha_opt': (5.0, 20.0),
    'mu_opt': (0.01, 0.1),
    'a_param_opt': (5.0, 20.0),
    'in_phase_val': (0.5, 1.0),
    'out_phase_val': (-1.0, -0.5),
    'stance_freq_opt': (1.0, 3.0),
    'swing_freq_opt': (3.0, 6.0)
}

optimizer = BayesianOptimization(
    f=cpg_blackbox_speed,
    pbounds=pbounds,
    verbose=2,
    random_state=42
)

# Lists to store iteration results
speed_progress = []
param_progress = []

# Callback for storing results each iteration
def store_results(params, speed):
    param_progress.append(params)
    speed_progress.append(speed)

# # Run the Bayesian optimization 
# print("Starting Bayesian Optimization to maximize forward speed...")
# optimizer.maximize(init_points=5, n_iter=15, callback=store_results)

# Run the Bayesian optimization 
print("Starting Bayesian Optimization to maximize forward speed...")
optimizer.maximize(init_points=5, n_iter=15)

# Manually store results after each iteration
for i, res in enumerate(optimizer.res):
    store_results(res['params'], res['target'])

# Extract best parameters
opt_params = optimizer.max['params']
best_speed = optimizer.max['target']

print("\nOptimal parameters for maximum forward speed found:")
print(opt_params)
print(f"Estimated forward speed: {best_speed:.6f} m/s")

# # Extract best parameters
# opt_params = optimizer.max['params']
# best_speed = -optimizer.max['target']

# print("\nOptimal parameters for maximum forward speed found:")
# print(opt_params)
# print(f"Estimated forward speed: {best_speed:.6f} m/s")

# =============================================================================
# 6. PLOTTING OPTIMIZATION PROGRESS
# =============================================================================
# 6a) Speed vs. Iterations
plt.figure(figsize=(10, 6))
iterations = range(len(speed_progress))
# Convert stored speeds from negative to positive
speeds_positive = [-val for val in speed_progress]
plt.plot(iterations, speeds_positive, marker='o', label='Forward Speed (m/s)')
plt.title("Bayesian Optimization Progress: Speed")
plt.xlabel("Iteration")
plt.ylabel("Speed (m/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 6b) Parameter Evolution
param_names = list(pbounds.keys())
param_array = []
for p in param_progress:
    # p is a dict with param_names as keys
    param_array.append([p[k] for k in param_names])
param_array = np.array(param_array)

plt.figure(figsize=(12, 8))
for i, pname in enumerate(param_names):
    plt.subplot(3, 3, i+1)
    plt.plot(iterations, param_array[:, i], marker='x', label=pname)
    plt.title(pname)
    plt.xlabel("Iteration")
    plt.ylabel(pname)
    plt.grid(True)
    plt.tight_layout()

plt.show()

# =============================================================================
# 7. FINAL SIMULATION & JOINT TRAJECTORY PLOT
# =============================================================================
# Convert the best param dict to a list in the correct order
best_param_list = [
    opt_params['alpha_opt'],
    opt_params['mu_opt'],
    opt_params['a_param_opt'],
    opt_params['in_phase_val'],
    opt_params['out_phase_val'],
    opt_params['stance_freq_opt'],
    opt_params['swing_freq_opt']
]

# We'll simulate once more and record the front-left flipper's joint angles
sim_duration = 30.0
dt_sim = 0.001
time_steps = int(sim_duration / dt_sim)
time_vec = np.linspace(0, sim_duration, time_steps)
frontleft_trajectory = []

# Reset oscillator states & simulation data
oscillators = {}
for name in actuator_names:
    oscillators[name] = {"x": 0.02 * np.random.randn(), "y": 0.02 * np.random.randn()}
mujoco.mj_resetData(model, data)

# Build new coupling matrix based on best parameters
K_opt = np.zeros((num_joints, num_joints))
for i in range(num_joints):
    for j in range(num_joints):
        if i == j:
            continue
        if (i in left_indices and j in left_indices) or (i in right_indices and j in right_indices):
            K_opt[i, j] = best_param_list[3]  # in_phase_val
        else:
            K_opt[i, j] = best_param_list[4]  # out_phase_val

for step in range(time_steps):
    x_all = [oscillators[name]["x"] for name in actuator_names]
    y_all = [oscillators[name]["y"] for name in actuator_names]
    
    # Update oscillator states
    for i, name in enumerate(actuator_names):
        x_i = oscillators[name]["x"]
        y_i = oscillators[name]["y"]
        freq = (best_param_list[5] / (1.0 + np.exp(-best_param_list[2] * y_i))) + \
               (best_param_list[6] / (1.0 + np.exp(best_param_list[2] * y_i)))
        x_new, y_new = hopf_step(x_i, y_i, best_param_list[0], best_param_list[1], freq, dt_sim, K_opt, x_all, y_all, i)
        oscillators[name]["x"], oscillators[name]["y"] = x_new, y_new
    
    # Map oscillator output to joint angles
    for name in actuator_names:
        x_i = oscillators[name]["x"]
        y_i = oscillators[name]["y"]
        phi_offset = phase_offsets_diagforward[name]
        x_phase = x_i * np.cos(phi_offset) - y_i * np.sin(phi_offset)
        
        offset = joint_output_map[name]["offset"]
        gain   = joint_output_map[name]["gain"]
        angle_raw = offset + gain * x_phase
        
        min_angle, max_angle = joint_limits[name]
        angle_clamped = np.clip(angle_raw, min_angle, max_angle)
        
        data.ctrl[get_actuator_index(model, name)] = angle_clamped
        
        if name == "pos_frontleftflipper":
            frontleft_trajectory.append(angle_clamped)
    
    mujoco.mj_step(model, data)

plt.figure(figsize=(10, 5))
plt.plot(time_vec, frontleft_trajectory, label="Front-Left Flipper")
plt.xlabel("Time (s)")
plt.ylabel("Joint Angle (rad)")
plt.title("Joint Trajectory with Best Parameters (Forward Speed)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
