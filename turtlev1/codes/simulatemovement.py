import mujoco
import mujoco.viewer
import time
import numpy as np

# Load the model and create a simulation instance
model = mujoco.MjModel.from_xml_path('c:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/assets/turtlev1/testrobot1.xml')
data = mujoco.MjData(model)

# Helper function to get actuator index by name
def get_actuator_index(model, name):
    for i in range(model.nu):
        if model.names[model.name_actuatoradr[i]:].startswith(name.encode()):
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

# Define the motion patterns for the flippers and hips
def turtle_motion_pattern(time_step):
    """
    Generate a sea-turtle-like swimming pattern.
    Returns a dictionary of desired positions for the actuators.
    """
    amplitude = 3  # Flipper swing amplitude
    frequency = 0.5  # Frequency of motion (in Hz)
    
    # Sine wave motion for front and back flippers
    front_flipper_motion = amplitude * np.sin(2 * np.pi * frequency * time_step)
    front_right_flipper_motion = amplitude * np.sin(2 * np.pi * frequency * time_step)
    front_left_flipper_motion = -1 * amplitude * np.sin(2 * np.pi * frequency * time_step)
    back_right_flipper_motion = -0.5 * front_flipper_motion  # Opposite phase for back flippers
    back_left_flipper_motion = 0.5 * front_flipper_motion  # Opposite phase for back flippers


    # Hips move in a coordinated way with the flippers
    left_hip_motion = 1.5 * front_flipper_motion
    right_hip_motion = 1.5 * front_flipper_motion


    return {
        'pos_frontleftflipper': front_left_flipper_motion,
        'pos_frontrightflipper': front_right_flipper_motion,
        'pos_backleft': back_left_flipper_motion,
        'pos_backright': back_right_flipper_motion,
        'pos_frontlefthip': left_hip_motion,
        'pos_frontrighthip': right_hip_motion
    }

# Fetch actuator indices for all actuators
actuator_names = [
    "pos_frontleftflipper", "pos_frontrightflipper", 
    "pos_backleft", "pos_backright", 
    "pos_frontlefthip", "pos_frontrighthip"
]

actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

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
