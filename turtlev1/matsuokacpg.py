import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt  # Optional: for plotting joint angles

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
    Default gait: uses a balanced tonic input and moderate coupling.
    Returns:
      s: tonic input vector for each oscillator.
      K: coupling matrix to coordinate phase relationships.
    """
    # Tonic input: can modulate duty cycle; here, front oscillators have slightly higher tonic.
    s = np.array([1.0, 1.0, 1.2, 1.2, 1.2, 1.2])
    # Define coupling matrix (6x6) as in the Hopf example.
    K = np.zeros((num_oscillators, num_oscillators))
    # Couple rear flipper oscillator with corresponding front flipper oscillator.
    k_rear = 0.5
    K[0, 3] = k_rear; K[3, 0] = k_rear
    K[1, 5] = k_rear; K[5, 1] = k_rear
    # Couple front hip and front flipper oscillators.
    k_front = 0.5
    K[2, 3] = k_front; K[3, 2] = k_front
    K[4, 5] = k_front; K[5, 4] = k_front
    # Optional small coupling between left and right front hips for symmetry.
    k_lr = 0.1
    K[2, 4] = k_lr; K[4, 2] = k_lr
    return s, K

def gait_option_alternate(num_oscillators=6):
    """
    Alternate gait: 
      - Group A: Front right hip (index 2), front right flipper (index 3), and back left (index 1)
        are in phase.
      - Group B: Front left hip (index 4), front left flipper (index 5), and back right (index 0)
        are in phase.
      - The two groups are in anti-phase so that in a cycle one group performs, then the other.
      
    Returns:
      s: Tonic input vector for each oscillator.
      K: Coupling matrix enforcing the desired phase relationships.
    """
    # For simplicity, we set the tonic input the same for all oscillators.
    s = np.array([1.0] * num_oscillators)
    
    # Define groups by oscillator index.
    group_A = [2, 3, 1]  # front right hip, front right flipper, back left
    group_B = [4, 5, 0]  # front left hip, front left flipper, back right
    
    # Initialize coupling matrix.
    K = np.zeros((num_oscillators, num_oscillators))
    
    k_intra = 0.5   # Positive coupling within the same group.
    k_inter = -0.5  # Negative coupling between groups (anti-phase).
    
    # Set intra-group coupling for Group A.
    for i in group_A:
        for j in group_A:
            if i != j:
                K[i, j] = k_intra
    # Set intra-group coupling for Group B.
    for i in group_B:
        for j in group_B:
            if i != j:
                K[i, j] = k_intra
    # Set inter-group coupling (symmetric).
    for i in group_A:
        for j in group_B:
            K[i, j] = k_inter
            K[j, i] = k_inter
    return s, K

def gait_option_alternate2(num_oscillators=6):
    """
    Alternate gait: uses a slightly reduced tonic input for the rear and increased for the front,
    with different coupling strengths.
    Returns:
      s: tonic input vector for each oscillator.
      K: coupling matrix to coordinate phase relationships.
    """
    s = np.array([0.8, 0.8, 1.4, 1.4, 1.4, 1.4])
    K = np.zeros((num_oscillators, num_oscillators))
    # Stronger coupling for rear flippers.
    k_rear = 0.7
    K[0, 3] = k_rear; K[3, 0] = k_rear
    K[1, 5] = k_rear; K[5, 1] = k_rear
    # Weaker coupling between front hip and flipper.
    k_front = 0.3
    K[2, 3] = k_front; K[3, 2] = k_front
    K[4, 5] = k_front; K[5, 4] = k_front
    # No left-right hip coupling for this gait.
    return s, K

# -----------------------------------------------------------------------------
#                      SIMULATION FUNCTION (Matsuoka Oscillators)
# -----------------------------------------------------------------------------
def run_simulation(gait_fn, sim_steps=10000, dt=0.001):
    """
    Run the simulation using Matsuoka oscillator dynamics and a specified gait.
    
    Parameters:
      gait_fn: function returning (s, K) for the gait.
      sim_steps: total number of simulation steps.
      dt: simulation time step.
    """
    num_oscillators = 6
    # Retrieve gait parameters: tonic inputs and coupling matrix.
    s_vec, K = gait_fn(num_oscillators)
    
    # Matsuoka oscillator parameters (these can be tuned)
    tau = 0.3         # Time constant for membrane potential dynamics.
    tau_prime = 0.3   # Time constant for adaptation.
    beta_param = 2.5  # Adaptation strength.
    w_inh = 2.0       # Mutual inhibition weight.
    
    # Initialize state variables for each oscillator.
    # u1, u2: membrane potentials; v1, v2: adaptation variables.
    u1 = np.random.uniform(-0.1, 0.1, size=num_oscillators)
    u2 = np.random.uniform(-0.1, 0.1, size=num_oscillators)
    v1 = np.zeros(num_oscillators)
    v2 = np.zeros(num_oscillators)
    
    # Mapping parameters: offsets and gains for converting oscillator output to joint angles.
    b = np.array([1.0, 1.0, 0.175, 0.463, -0.175, -0.463])
    g = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    
    # Control ranges for each actuator as defined in your XML.
    ctrl_min = np.array([-0.524, -1.571, -1.22, -0.64, -1.571, -1.57])
    ctrl_max = np.array([ 1.571,  0.524,  1.571,  1.571,  1.22,  0.64])
    
    def clip_angle(theta, idx):
        return np.clip(theta, ctrl_min[idx], ctrl_max[idx])
    
    # For logging joint angles (optional: for plotting)
    theta_history = []
    time_history = []
    
    # Launch the MuJoCo viewer.
    viewer = mujoco.viewer.launch_passive(model, data)
    
    # Simulation loop.
    for step in range(sim_steps):
        current_time = step * dt
        time_history.append(current_time)
        
        # Compute firing rates (outputs of the neurons).
        y1 = np.maximum(0, u1)
        y2 = np.maximum(0, u2)
        
        # Compute oscillator output.
        y = y1 - y2  # This is the signal to map to joint angles.
        
        # Compute coupling inputs for each oscillator.
        I1 = np.zeros(num_oscillators)
        I2 = np.zeros(num_oscillators)
        for i in range(num_oscillators):
            for j in range(num_oscillators):
                I1[i] += K[i, j] * (y1[j] - y1[i])
                I2[i] += K[i, j] * (y2[j] - y2[i])
        
        # Update the oscillator states using Euler integration.
        # u1 and u2 dynamics.
        du1 = (-u1 - beta_param * v1 - w_inh * y2 + s_vec + I1) / tau
        du2 = (-u2 - beta_param * v2 - w_inh * y1 + s_vec + I2) / tau
        # Adaptation variable dynamics.
        dv1 = (-v1 + y1) / tau_prime
        dv2 = (-v2 + y2) / tau_prime
        
        # Euler update.
        u1 += du1 * dt
        u2 += du2 * dt
        v1 += dv1 * dt
        v2 += dv2 * dt
        
        # Compute joint angles from oscillator outputs.
        theta = np.zeros(num_oscillators)
        for i in range(num_oscillators):
            theta[i] = b[i] + g[i] * y[i]
            theta[i] = clip_angle(theta[i], i)
        
        theta_history.append(theta.copy())
        
        # Send commands to actuators (position control).
        for i, act_idx in enumerate(actuator_indices):
            data.ctrl[act_idx] = theta[i]
        
        # Step the simulation and update the viewer.
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
    
    # Optional: Plot joint angles over time.
    theta_history = np.array(theta_history)  # shape: (sim_steps, num_oscillators)
    plt.figure(figsize=(10, 6))
    for i in range(num_oscillators):
        plt.plot(time_history, theta_history[:, i], label=f'Joint {i+1}')
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")
    plt.title("Joint Angles from Matsuoka Oscillator CPG")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------------
#                              MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # To run simulation with the default gait:
    print("Running simulation with default gait (Matsuoka-based)...")
    run_simulation(gait_option_default, sim_steps=15000, dt=0.001)
    
    # To run simulation with an alternate gait, uncomment the following lines:
    # print("Running simulation with alternate gait (Matsuoka-based)...")
    # run_simulation(gait_option_alternate, sim_steps=15000, dt=0.001)
