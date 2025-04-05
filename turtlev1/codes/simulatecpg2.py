import mujoco
import mujoco.viewer
import time
import numpy as np

# Load the model and create a simulation instance
model_path = 'c:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/assets/turtlev1/testrobot1.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# for i in range(model.njnt):
#     joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
#     joint_range = model.jnt_range[i]
#     print(f"Joint: {joint_name}, Range: {joint_range}")


# Function to get actuator index by name
def get_actuator_index(model, name):
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

# Define actuator names
actuator_names = [
    "pos_frontleftflipper", "pos_frontrightflipper", 
    "pos_backleft", "pos_backright", 
    "pos_frontlefthip", "pos_frontrighthip"
]

# Get actuator indices
actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

# Define the Hopf Oscillator-based CPG Model
class CoupledHopfCPG:
    def __init__(self, num_oscillators=6, dt=0.01):
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.alpha = 5.0  # Convergence speed
        self.mu = 1.0     # Amplitude
        # self.omega = np.array([0.2] * num_oscillators)  # Base frequency
        self.omega = np.array([0.88, 0.88, 0.9, 0.9, 0.88, 0.88]) / 2 # Frequency tuning

        
        # Initialize oscillator states
        self.x = np.zeros(num_oscillators)
        self.y = np.zeros(num_oscillators)

        # self.x = np.random.rand(num_oscillators)
        # self.y = np.random.rand(num_oscillators)
        
        self.phases = np.random.rand(num_oscillators) * 2 * np.pi  # Random phase init

        # Phase coupling matrix (synchronizes movement)
        # self.K = np.array([
        #     [ 0, -0.3,  0.3,  0,  0,  0],
        #     [-0.3,  0,  0,  0.3,  0,  0],
        #     [ 0,  0.3,  0.3, 0,  0,  0],
        #     [ 0,  0.3, -0.3,  0,  0,  0],
        #     [ 0,  0,  0,  0,  0.3, -0.3],
        #     [ 0,  0,  0,  0, -0.3,  0.3]
        # ])

        # New Coupling Matrix for In-Phase Motion (Synchronization)
        self.K = np.full((num_oscillators, num_oscillators), np.pi)  # All positive
        np.fill_diagonal(self.K, 0)  # No self-coupling

    def update(self):
        """ Updates the Hopf oscillators with phase coupling. """
        r = np.sqrt(self.x**2 + self.y**2)

        # Compute phase synchronization term
        phase_sync = np.sum(self.K * np.sin(np.subtract.outer(self.phases, self.phases)), axis=1)

        # Hopf oscillator equations with phase coupling
        dx = self.alpha * (self.mu - r**2) * self.x - self.omega * self.y + phase_sync
        dy = self.alpha * (self.mu - r**2) * self.y + self.omega * self.x

        self.x += dx * self.dt
        self.y += dy * self.dt

        # Update phase values
        self.phases = np.arctan2(self.y, self.x)

        return self.x  # Joint angles

# Initialize the Hopf CPG for the robot
cpg = CoupledHopfCPG(num_oscillators=6)

# Function to return actuator positions with CPG control
def turtle_motion_pattern():
    """ Generates joint control signals using coupled Hopf CPG. """
    joint_angles = cpg.update()

    # Define MuJoCo joint limits
    # joint_min = np.array([-1.2, -1.2, -0.8, -0.8, -0.5, -0.5])
    # joint_max = np.array([1.2, 1.2, 0.8, 0.8, 0.5, 0.5])

    # Updated joint limits from MuJoCo
    joint_min = np.array([-1.571, -0.64, -1.571, -0.53, -1.571, -1.22])  # Min angles
    joint_max = np.array([0.64, 1.571, 0.53, 1.571, 1.22, 1.571])  # Max angles


    
    # Scale and map joint angles
    joint_scaled = joint_min + (joint_angles / np.max(np.abs(joint_angles) + 1e-5)) * (joint_max - joint_min)

    return {
        "pos_frontleftflipper": joint_scaled[0],
        "pos_frontrightflipper": joint_scaled[1],
        "pos_backleft": joint_scaled[2],
        "pos_backright": joint_scaled[3],
        "pos_frontlefthip": joint_scaled[4],
        "pos_frontrighthip": joint_scaled[5]
    }

# Launch the viewer with Hopf CPG-based control
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    while viewer.is_running():
        # Current simulation time
        current_time = time.time() - start_time

        # Get the desired positions for the flippers and hips from the CPG
        target_positions = turtle_motion_pattern()

        # Apply position control to actuators
        for name, value in target_positions.items():
            data.ctrl[actuator_indices[name]] = value

        # Step the simulation
        mujoco.mj_step(model, data)

        # Sync the viewer
        viewer.sync()
