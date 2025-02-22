import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#                          LOAD MUJOCO MODEL
# -----------------------------------------------------------------------------
model_path = r'c:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/assets/turtlev1/testrobot1.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

def get_actuator_index(model, name):
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

# Actuator names as defined in your XML (position control actuators)
actuator_names = [
    "pos_backright", 
    "pos_backleft", 
    "pos_frontrighthip", 
    "pos_frontrightflipper",
    "pos_frontlefthip", 
    "pos_frontleftflipper"
]
actuator_indices = [get_actuator_index(model, name) for name in actuator_names]

# -----------------------------------------------------------------------------
#                   DEFINE GAIT OPTIONS AS SEPARATE FUNCTIONS
# -----------------------------------------------------------------------------
def gait_option_default(num_oscillators=6):
    """
    Default gait: balanced duty cycle with slightly longer support for the front flippers.
    Returns:
      beta: fraction of cycle allocated to support phase for each oscillator.
      K: coupling matrix to coordinate phase relationships.
    """
    beta = np.array([0.5, 0.5, 0.6, 0.6, 0.6, 0.6])
    K = np.zeros((num_oscillators, num_oscillators))
    # Coupling between rear flippers and corresponding front flipper:
    k_rear = 0.5
    K[0, 3] = k_rear  # back right with front right flipper
    K[1, 5] = k_rear  # back left with front left flipper
    K[3, 0] = k_rear
    K[5, 1] = k_rear
    # Coupling between front flipper hip and flipper:
    k_front = 0.5
    K[2, 3] = k_front  # front right hip to front right flipper
    K[3, 2] = k_front
    K[4, 5] = k_front  # front left hip to front left flipper
    K[5, 4] = k_front
    # Optional small coupling between left and right front hips:
    k_lr = 0.5
    K[2, 4] = k_lr
    K[4, 2] = k_lr
    return beta, K

def gait_option_alternate(num_oscillators=6):
    """
    Alternate gait: different duty cycles (shorter support for rear, longer for front)
    and modified coupling strengths.
    Returns:
      beta: fraction of cycle allocated to support phase for each oscillator.
      K: coupling matrix to coordinate phase relationships.
    """
    beta = np.array([0.4, 0.4, 0.7, 0.7, 0.7, 0.7])
    K = np.zeros((num_oscillators, num_oscillators))
    # Increase rear coupling for more synchronization:
    k_rear = 0.7
    K[0, 3] = k_rear
    K[1, 5] = k_rear
    K[3, 0] = k_rear
    K[5, 1] = k_rear
    # Slightly reduce front coupling:
    k_front = 0.3
    K[2, 3] = k_front
    K[3, 2] = k_front
    K[4, 5] = k_front
    K[5, 4] = k_front
    # No left-right coupling for variation.
    return beta, K

# -----------------------------------------------------------------------------
#                      SIMULATION FUNCTION WITH PLOTTING
# -----------------------------------------------------------------------------
def run_simulation(gait_fn, sim_steps=10000, dt=0.001):
    """
    Run the simulation using the gait defined by gait_fn and plot joint angles.
    
    Parameters:
      gait_fn: function returning (beta, K) for the gait.
      sim_steps: total simulation steps.
      dt: simulation time step.
    """
    num_oscillators = 6
    # Get gait parameters (beta and coupling matrix K)
    beta, K = gait_fn(num_oscillators)
    
    # Oscillator parameters
    alpha = 10.0         # convergence gain
    mu = 10.0             # limit cycle parameter (amplitude ~sqrt(mu))
    a_param = 5.0       # steepness for frequency modulation sigmoid
    
    # Intrinsic swing frequency (rad/s) for each oscillator (e.g., 1 Hz swing phase)
    omega_sw = np.array([2*np.pi*1.0] * num_oscillators)
    # Calculate support frequency from swing frequency and beta
    omega_st = omega_sw * (1 - beta) / beta

    # Initialize oscillator states (x, y for each oscillator)
    x = np.random.uniform(-0.1, 0.1, size=num_oscillators)
    y = np.random.uniform(-0.1, 0.1, size=num_oscillators)
    
    # Mapping parameters: offsets and gains for converting oscillator output to joint angles.
    b = np.array([0.0, 0.0, 0.175, 0.463, -0.175, -0.463])
    g = np.array([0.5, 0.5, 0.9, 0.5, 0.9, 0.5])
    
    # Control ranges (min, max) for each actuator as defined in the XML.
    ctrl_min = np.array([-0.524, -1.571, -1.22, -0.64, -1.571, -1.57])
    ctrl_max = np.array([ 1.571,  0.524,  1.571,  1.571,  1.22,  0.64])
    
    def clip_angle(theta, idx):
        return np.clip(theta, ctrl_min[idx], ctrl_max[idx])
    
    # Prepare to log joint angles over time for plotting.
    theta_history = []  # will be a list of arrays (one per simulation step)
    time_history = []   # record time at each step

    # Launch the MuJoCo viewer for visualization.
    viewer = mujoco.viewer.launch_passive(model, data)
    
    # Simulation loop
    for step in range(sim_steps):
        current_time = step * dt
        time_history.append(current_time)
        
        # Compute frequency for each oscillator using sigmoid blending.
        omega = np.zeros(num_oscillators)
        for i in range(num_oscillators):
            omega[i] = (omega_st[i] / (1 + np.exp(-a_param * y[i]))) + \
                       (omega_sw[i] / (1 + np.exp(a_param * y[i])))
        
        # Compute coupling contributions.
        coupling_x = np.zeros(num_oscillators)
        coupling_y = np.zeros(num_oscillators)
        for i in range(num_oscillators):
            for j in range(num_oscillators):
                coupling_x[i] += K[i, j] * (x[j] - x[i])
                coupling_y[i] += K[i, j] * (y[j] - y[i])
        
        # Update oscillator states using Euler integration.
        for i in range(num_oscillators):
            r_sq = x[i]**2 + y[i]**2
            dx = alpha * (mu - r_sq) * x[i] - omega[i] * y[i] + coupling_x[i]
            dy = alpha * (mu - r_sq) * y[i] + omega[i] * x[i] + coupling_y[i]
            x[i] += dx * dt
            y[i] += dy * dt
        
        # Map oscillator state to joint angles (using the x-component).
        theta = np.zeros(num_oscillators)
        for i in range(num_oscillators):
            theta[i] = b[i] + g[i] * x[i]
            theta[i] = clip_angle(theta[i], i)
        
        # Log joint angles for plotting.
        theta_history.append(theta.copy())
        
        # Send position control commands to actuators.
        for i, act_idx in enumerate(actuator_indices):
            data.ctrl[act_idx] = theta[i]
        
        # Step the simulation and update viewer.
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
    
    # Convert logged data to a numpy array for plotting.
    theta_history = np.array(theta_history)  # shape: (sim_steps, num_oscillators)
    
    # Plot the joint angles over time.
    plt.figure(figsize=(10, 6))
    for i in range(num_oscillators):
        plt.plot(time_history, theta_history[:, i], label=f'Joint {i+1}')
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")
    plt.title("Joint Angles from CPG")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------------
#                              MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Run simulation with default gait and plot joint angles.
    run_simulation(gait_option_default, sim_steps=10000, dt=0.001)
    # You can switch to the alternate gait by passing gait_option_alternate.
