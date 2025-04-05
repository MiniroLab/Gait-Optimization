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

# Define phase shifts for each joint (in radians)
sync_phase_shifts = {
    "pos_frontleftflipper": 0.0,    
    "pos_frontrightflipper": np.pi,  
    "pos_backleft": np.pi/2,  
    "pos_backright": -np.pi/2,  
    "pos_frontlefthip": -np.pi/4,  
    "pos_frontrighthip": -np.pi/4  
}

# Define phase shifts for each joint (in radians)
diag_phase_shifts = {
    "pos_frontleftflipper": 0.0,    
    "pos_frontrightflipper": np.pi,  
    "pos_backleft": np.pi/2,  
    "pos_backright": -np.pi/2,  
    "pos_frontlefthip": -np.pi/4,  
    "pos_frontrighthip": -np.pi/4  
}


freq_tuner = 1.5  # Frequency tuning factor

# Define the Fourier-based joint control functions with phase control
def joint_angle(t_real, A, B, omega, phase):
    return A * np.sin(omega * t_real + phase) + B * np.cos(omega * t_real + phase)

# Function to return actuator positions with phase shifts
def turtle_motion_pattern(t_real):
    return {
        "pos_frontleftflipper": joint_angle(t_real, 0.23, -1.15, 0.88 * freq_tuner, sync_phase_shifts["pos_frontleftflipper"]),
        "pos_frontrightflipper": joint_angle(t_real, 0.23, -1.15, 0.88 * freq_tuner, sync_phase_shifts["pos_frontrightflipper"]),
        "pos_backleft": joint_angle(t_real, -1.3307, 0.5679, 0.9 * freq_tuner, sync_phase_shifts["pos_backleft"]),
        "pos_backright": joint_angle(t_real, -1.3307, 0.5679, 0.9 * freq_tuner, sync_phase_shifts["pos_backright"]),
        "pos_frontlefthip": joint_angle(t_real, -0.8, -0.57, 0.88 * freq_tuner, sync_phase_shifts["pos_frontlefthip"]),
        "pos_frontrighthip": joint_angle(t_real, -0.8, -0.57, 0.88 * freq_tuner, sync_phase_shifts["pos_frontrighthip"])
    }

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    while viewer.is_running():
        # Current simulation time
        current_time = time.time() - start_time

        # Get the desired positions for the flippers and hips
        target_positions = turtle_motion_pattern(current_time)

        # Apply position control to actuators
        for name, value in target_positions.items():
            data.ctrl[actuator_indices[name]] = value

        # Step the simulation
        mujoco.mj_step(model, data)

        # Sync the viewer
        viewer.sync()
