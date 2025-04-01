import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import os
import sys


# Load the model and create a simulation instance

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from setup import load_model_from_yaml

if "yaml_path" not in globals(): #if yaml not provided, use model and data from this
    raise ValueError("yaml_path must be provided.")
yaml_path = globals().get("yaml_path")

model, data = load_model_from_yaml.get_model_and_data_from_yaml(yaml_path)

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

# Prepare data structures to store time and control values
time_data = []
ctrl_data = {name: [] for name in actuator_names}

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

        # Record time and control data
        time_data.append(current_time)
        for name in actuator_names:
            ctrl_data[name].append(data.ctrl[actuator_indices[name]])

        # Sync the viewer
        viewer.sync()

# ----------------------------
# Plot the stored joint angles
# ----------------------------
plt.figure(figsize=(10, 6))
for name in actuator_names:
    plt.plot(time_data, ctrl_data[name], label=name)

plt.title('Actuator Control Values Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Control Value (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
