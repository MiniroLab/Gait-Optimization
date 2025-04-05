import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

# =============================================================================
# 1. LOAD MUJOCO MODEL
# =============================================================================
model_path = 'c:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/assets/turtlev1/testrobot1.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Helper: get actuator index (not used in simulation-only mode but kept for reference)
def get_actuator_index(model, name):
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

# =============================================================================
# 2. FOURIER-BASED TARGET PATTERN
# =============================================================================

# Define actuator names for the Fourier pattern (order used in the Fourier code)
fourier_order = [
    "pos_frontleftflipper", "pos_frontrightflipper", 
    "pos_backleft", "pos_backright", 
    "pos_frontlefthip", "pos_frontrighthip"
]

# Define phase shifts for each joint (in radians)
sync_phase_shifts = {
    "pos_frontleftflipper": 0.0,    
    "pos_frontrightflipper": np.pi,  
    "pos_backleft": np.pi/2,  
    "pos_backright": -np.pi/2,  
    "pos_frontlefthip": -np.pi/4,  
    "pos_frontrighthip": -np.pi/4  
}

freq_tuner = 1.5  # Frequency tuning factor

def joint_angle(t_real, A, B, omega, phase):
    return A * np.sin(omega * t_real + phase) + B * np.cos(omega * t_real + phase)

def turtle_motion_pattern(t_real):
    return {
        "pos_frontleftflipper": joint_angle(t_real, 0.23, -1.15, 0.88 * freq_tuner, sync_phase_shifts["pos_frontleftflipper"]),
        "pos_frontrightflipper": joint_angle(t_real, 0.23, -1.15, 0.88 * freq_tuner, sync_phase_shifts["pos_frontrightflipper"]),
        "pos_backleft": joint_angle(t_real, -1.3307, 0.5679, 0.9 * freq_tuner, sync_phase_shifts["pos_backleft"]),
        "pos_backright": joint_angle(t_real, -1.3307, 0.5679, 0.9 * freq_tuner, sync_phase_shifts["pos_backright"]),
        "pos_frontlefthip": joint_angle(t_real, -0.8, -0.57, 0.88 * freq_tuner, sync_phase_shifts["pos_frontlefthip"]),
        "pos_frontrighthip": joint_angle(t_real, -0.8, -0.57, 0.88 * freq_tuner, sync_phase_shifts["pos_frontrighthip"])
    }

# =============================================================================
# 3. MATSUKA OSCILLATOR SIMULATION (NO VIEWER) 
# =============================================================================

# The simulation uses the following actuator order (Matsuoka order)
matsuoka_order = [
    "pos_backright", 
    "pos_backleft", 
    "pos_frontrighthip", 
    "pos_frontrightflipper",
    "pos_frontlefthip", 
    "pos_frontleftflipper"
]

# We need a mapping from the Matsuoka order to the Fourier target order.
# Mapping: index in matsuoka_order -> corresponding Fourier name index.
# From the provided definitions:
#   Matsuoka: 0: "pos_backright"   <-- Fourier index 3
#              1: "pos_backleft"    <-- Fourier index 2
#              2: "pos_frontrighthip" <-- Fourier index 5
#              3: "pos_frontrightflipper" <-- Fourier index 1
#              4: "pos_frontlefthip"  <-- Fourier index 4
#              5: "pos_frontleftflipper" <-- Fourier index 0
order_map = [3, 2, 5, 1, 4, 0]

# Define gait function (we use the default gait)
def gait_option_default(num_oscillators=6):
    s = np.array([1.0, 1.0, 1.2, 1.2, 1.2, 1.2])
    K = np.zeros((num_oscillators, num_oscillators))
    k_rear = 0.5
    K[0, 3] = k_rear; K[3, 0] = k_rear
    K[1, 5] = k_rear; K[5, 1] = k_rear
    k_front = 0.5
    K[2, 3] = k_front; K[3, 2] = k_front
    K[4, 5] = k_front; K[5, 4] = k_front
    k_lr = 0.1
    K[2, 4] = k_lr; K[4, 2] = k_lr
    return s, K

# Simulation function that returns joint angles over time given CPG parameters.
def simulate_cpg(params, gait_fn, sim_steps=2000, dt=0.005):
    """
    Run a simulation (without viewer) of the Matsuoka oscillator CPG.
    params: [tau, tau_prime, beta_param, w_inh]
    Returns: time vector and simulated joint angles of shape (sim_steps, num_oscillators)
    """
    num_oscillators = 6
    # Unpack the optimized parameters:
    tau, tau_prime, beta_param, w_inh = params
    # Retrieve gait parameters from the chosen gait function.
    s_vec, K = gait_fn(num_oscillators)
    
    # Initialize oscillator state variables.
    u1 = np.random.uniform(-0.1, 0.1, size=num_oscillators)
    u2 = np.random.uniform(-0.1, 0.1, size=num_oscillators)
    v1 = np.zeros(num_oscillators)
    v2 = np.zeros(num_oscillators)
    
    # Mapping parameters to convert oscillator output to joint angles.
    b = np.array([0.0, 0.0, 0.175, 0.463, -0.175, -0.463])
    g_map = np.array([1.5, 1.5, 1.5, 3.5, 1.5, 3.5])
    
    # Actuator control ranges (not used for cost calculation, but kept for clipping).
    ctrl_min = np.array([-0.524, -1.571, -1.22, -0.64, -1.571, -1.57])
    ctrl_max = np.array([ 1.571,  0.524,  1.571,  1.571,  1.22,  0.64])
    def clip_angle(theta, idx):
        return np.clip(theta, ctrl_min[idx], ctrl_max[idx])
    
    theta_history = []
    time_history = []
    
    for step in range(sim_steps):
        t = step * dt
        time_history.append(t)
        
        # Compute firing rates.
        y1 = np.maximum(0, u1)
        y2 = np.maximum(0, u2)
        y = y1 - y2  # oscillator output
        
        # Compute coupling inputs.
        I1 = np.zeros(num_oscillators)
        I2 = np.zeros(num_oscillators)
        for i in range(num_oscillators):
            for j in range(num_oscillators):
                I1[i] += K[i, j] * (y1[j] - y1[i])
                I2[i] += K[i, j] * (y2[j] - y2[i])
        
        # Update dynamics using Euler integration.
        du1 = (-u1 - beta_param * v1 - w_inh * y2 + s_vec + I1) / tau
        du2 = (-u2 - beta_param * v2 - w_inh * y1 + s_vec + I2) / tau
        dv1 = (-v1 + y1) / tau_prime
        dv2 = (-v2 + y2) / tau_prime
        
        u1 += du1 * dt
        u2 += du2 * dt
        v1 += dv1 * dt
        v2 += dv2 * dt
        
        # Map oscillator output to joint angles.
        theta = np.zeros(num_oscillators)
        for i in range(num_oscillators):
            theta[i] = b[i] + g_map[i] * y[i]
            theta[i] = clip_angle(theta[i], i)
        theta_history.append(theta.copy())
    
    theta_history = np.array(theta_history)  # shape: (sim_steps, num_oscillators)
    return np.array(time_history), theta_history

# =============================================================================
# 4. COST FUNCTION: COMPARE SIMULATED OUTPUT WITH TARGET PATTERN
# =============================================================================

def compute_cost(sim_time, sim_theta):
    """
    For each time in sim_time, compute the target joint angles from the Fourier pattern,
    then compute the mean squared error (MSE) between simulation and target.
    The target is re-ordered to match the Matsuoka simulation order.
    """
    sim_steps, num_joints = sim_theta.shape
    error_sum = 0.0
    count = 0
    for i, t in enumerate(sim_time):
        # Get target pattern (using Fourier pattern)
        target_dict = turtle_motion_pattern(t)
        # Build target vector in Fourier order.
        target_fourier = np.array([target_dict[name] for name in fourier_order])
        # Reorder target vector to match the simulation order using order_map.
        target_matsuoka = target_fourier[order_map]
        error_sum += np.mean((sim_theta[i, :] - target_matsuoka)**2)
        count += 1
    mse = error_sum / count
    return mse

# =============================================================================
# 5. BLACK-BOX FUNCTION FOR BAYESIAN OPTIMIZATION
# =============================================================================

def cpg_blackbox(tau, tau_prime, beta_param, w_inh):
    # Bundle parameters into a vector.
    params = [tau, tau_prime, beta_param, w_inh]
    # Run simulation for a short period.
    sim_time, sim_theta = simulate_cpg(params, gait_option_default, sim_steps=2000, dt=0.005)
    # Compute cost (we want to minimize the MSE)
    cost = compute_cost(sim_time, sim_theta)
    # BayesianOptimization maximizes the function, so return the negative cost.
    return -cost

# =============================================================================
# 6. BAYESIAN OPTIMIZATION
# =============================================================================

# Define bounds for the parameters.
# Adjust these bounds as needed for your system.
pbounds = {
    'tau': (0.1, 1.0),
    'tau_prime': (0.1, 1.0),
    'beta_param': (0.5, 5.0),
    'w_inh': (1.0, 5.0)
}

optimizer = BayesianOptimization(
    f=cpg_blackbox,
    pbounds=pbounds,
    verbose=2,  # verbose output
    random_state=42
)

print("Starting Bayesian Optimization...")
optimizer.maximize(init_points=5, n_iter=20)

optimal_params = optimizer.max['params']
print("Optimal parameters found:")
print(optimal_params)

# =============================================================================
# 7. RUN SIMULATION WITH OPTIMAL PARAMETERS AND PLOT RESULTS
# =============================================================================

# Convert optimal_params dict to list in the order: [tau, tau_prime, beta_param, w_inh]
opt_params = [optimal_params['tau'], optimal_params['tau_prime'], optimal_params['beta_param'], optimal_params['w_inh']]
sim_time, sim_theta = simulate_cpg(opt_params, gait_option_default, sim_steps=2000, dt=0.005)

# Build target trajectory for comparison.
target_traj = []
for t in sim_time:
    target_dict = turtle_motion_pattern(t)
    target_fourier = np.array([target_dict[name] for name in fourier_order])
    # Reorder target to match simulation order.
    target_traj.append(target_fourier[order_map])
target_traj = np.array(target_traj)

# Plot each joint's trajectory: simulation vs. target.
num_joints = sim_theta.shape[1]
plt.figure(figsize=(12, 8))
joint_names_sim = matsuoka_order
for j in range(num_joints):
    plt.subplot(3, 2, j+1)
    plt.plot(sim_time, sim_theta[:, j], label='Simulated')
    plt.plot(sim_time, target_traj[:, j], '--', label='Target')
    plt.title(joint_names_sim[j])
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")
    plt.legend()
plt.tight_layout()
plt.show()
