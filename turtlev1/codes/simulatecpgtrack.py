import mujoco
import mujoco.viewer
import time
import numpy as np

# Load the model and create a simulation instance
model_path = 'c:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/assets/turtlev1/testrobot1.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

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
        self.alpha = 15.0  # Faster convergence speed
        self.mu = 1.0     # Amplitude stabilization parameter
        self.omega = np.array([0.58, 0.58, 0.6, 0.6, 0.58, 0.58]) # Frequency tuning

        # Initialize oscillator states
        self.x = np.random.rand(num_oscillators)
        self.y = np.random.rand(num_oscillators)
        self.amplitudes = np.sqrt(self.x**2 + self.y**2)
        self.prev_amplitudes = np.copy(self.amplitudes)

        # Initialize random phase values between 0 to 2π
        # self.phases = np.random.rand(num_oscillators) * 2 * np.pi

        self.phases = np.array([0, np.pi, np.pi/2, -np.pi/2, -np.pi/4, -np.pi/4])

        self.prev_phases = np.copy(self.phases)

        # Phase coupling matrix for in-phase motion
        self.K = np.full((num_oscillators, num_oscillators), 0.2)  
        np.fill_diagonal(self.K, 0)  # No self-coupling

        # Track convergence metrics
        self.prev_phase_diffs = np.zeros((num_oscillators, num_oscillators))

    def update(self):
        """ Updates the Hopf oscillators with phase coupling for proper synchronization. """
        r = np.sqrt(self.x**2 + self.y**2)

        # Compute phase synchronization term
        phase_sync = np.sum(self.K * np.sin(np.subtract.outer(self.phases, self.phases)), axis=1)

        # Hopf oscillator equations with phase coupling
        dx = self.alpha * (self.mu - r**2) * self.x - self.omega * self.y + phase_sync
        dy = self.alpha * (self.mu - r**2) * self.y + self.omega * self.x

        self.x += dx * self.dt
        self.y += dy * self.dt

        # Update phase values correctly
        new_phases = np.arctan2(self.y, self.x)

        # Compute phase differences
        phase_diffs = np.abs(np.subtract.outer(new_phases, new_phases))
        phase_diff_change = np.max(np.abs(phase_diffs - self.prev_phase_diffs))
        self.prev_phase_diffs = phase_diffs

        # Compute amplitude stability
        new_amplitudes = np.sqrt(self.x**2 + self.y**2)
        amplitude_diff = np.max(np.abs(new_amplitudes - self.prev_amplitudes))
        self.prev_amplitudes = new_amplitudes

        # Compute frequency stability
        frequency_change = np.max(np.abs((new_phases - self.prev_phases) / self.dt))
        self.prev_phases = new_phases

        # Return joint angles and convergence metrics
        return self.x, phase_diff_change, amplitude_diff, frequency_change

# Initialize the Hopf CPG for the robot
cpg = CoupledHopfCPG(num_oscillators=6)

# Function to return actuator positions with CPG control
def turtle_motion_pattern():
    """ Generates joint control signals using coupled Hopf CPG. """
    joint_angles, _, _, _ = cpg.update()

    # Updated joint limits from MuJoCo
    joint_min = np.array([-1.571, -0.64, -1.571, -0.53, -1.571, -1.22])  # Min angles
    joint_max = np.array([0.64, 1.571, 0.53, 1.571, 1.22, 1.571])  # Max angles

    # Scale joint angles
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
    converged = False
    convergence_threshold = 0.1  # Threshold for convergence detection

    while viewer.is_running():
        # Current simulation time
        current_time = time.time() - start_time

        # Get the CPG-generated joint positions
        joint_positions, phase_change, amplitude_diff, frequency_change = cpg.update()

        # Check for convergence
        if not converged and phase_change < convergence_threshold and amplitude_diff < convergence_threshold and frequency_change < convergence_threshold:
            converged = True
            print(f"CPG Converged at t = {current_time:.2f} seconds")

        # Get target positions from CPG
        target_positions = turtle_motion_pattern()

        # Apply position control to actuators
        for name, value in target_positions.items():
            data.ctrl[actuator_indices[name]] = value

        # Step the simulation
        mujoco.mj_step(model, data)

        # Sync the viewer
        viewer.sync()
