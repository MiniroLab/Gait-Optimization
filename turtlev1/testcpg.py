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

# Actuator names (as defined in your XML)
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
#                 SIMULATION PARAMETERS & OSCILLATOR INITIALIZATION
# -----------------------------------------------------------------------------
dt = 0.001
sim_steps = 15000
num_osc = 6

# Hopf oscillator parameters:
alpha = 10.0
mu = 1.0
omega_const = 2 * np.pi * 1.0  # 1 Hz natural frequency

# Initialize each oscillator's state (x, y) randomly near zero
x = np.random.uniform(-0.1, 0.1, size=num_osc)
y = np.random.uniform(-0.1, 0.1, size=num_osc)

# -----------------------------------------------------------------------------
#                DEFINE JOINT LIMITS & MAPPING PARAMETERS
# -----------------------------------------------------------------------------
# Control ranges for each actuator (order: backright, backleft, frontrighthip, 
# frontrightflipper, frontlefthip, frontleftflipper)
ctrl_min = np.array([-0.524, -1.571, -1.22, -0.64, -1.571, -1.57])
ctrl_max = np.array([ 1.571,  0.524,  1.571,  1.571,  1.22,  0.64])

# --- Hip mapping (for front hips: indices 2 and 4) ---
# We choose the lower and upper limits from the control range for hip joints.
hip_lower = ctrl_min[2]   # e.g. -1.22 rad
hip_upper = ctrl_max[2]   # e.g. 1.571 rad

# --- Flipper mapping (for front flippers: indices 3 and 5) ---
# Compute midpoints and amplitudes from control range.
flipper_mid_right = (ctrl_max[3] + ctrl_min[3]) / 2.0
flipper_amp_right = (ctrl_max[3] - ctrl_min[3]) / 2.0
flipper_mid_left  = (ctrl_max[5] + ctrl_min[5]) / 2.0
flipper_amp_left  = (ctrl_max[5] - ctrl_min[5]) / 2.0

# --- Back flipper mapping (indices 0 and 1) ---
back_mid_right = (ctrl_max[0] + ctrl_min[0]) / 2.0
back_amp_right = (ctrl_max[0] - ctrl_min[0]) / 2.0
back_mid_left  = (ctrl_max[1] + ctrl_min[1]) / 2.0
back_amp_left  = (ctrl_max[1] - ctrl_min[1]) / 2.0

# Define the phase threshold for the hip mapping.
# Here, we assume that during the first half of the oscillator cycle (phase < π)
# the hips remain at the lower limit (support phase), and during the second half they go to the upper limit.
phi_threshold = np.pi

# -----------------------------------------------------------------------------
#                          SIMULATION LOOP WITH CPG MAPPING
# -----------------------------------------------------------------------------
theta_history = []
time_history = []

# Launch the MuJoCo viewer.
viewer = mujoco.viewer.launch_passive(model, data)

for step in range(sim_steps):
    t = step * dt
    time_history.append(t)
    
    # Update each oscillator with Hopf dynamics.
    for i in range(num_osc):
        r_sq = x[i]**2 + y[i]**2
        dx = alpha * (mu - r_sq) * x[i] - omega_const * y[i]
        dy = alpha * (mu - r_sq) * y[i] + omega_const * x[i]
        x[i] += dx * dt
        y[i] += dy * dt
    
    # Compute the phase for each oscillator using arctan2.
    phase = np.arctan2(y, x)
    # Convert phase from [-pi, pi] to [0, 2pi]
    phase = np.where(phase < 0, phase + 2 * np.pi, phase)
    
    # Initialize joint angles array.
    theta = np.zeros(num_osc)
    
    # --- Map Hip Oscillators (indices 2 and 4) ---
    # Use a piecewise mapping: if the oscillator phase is less than the threshold,
    # set the joint angle to the lower limit; otherwise, set it to the upper limit.
    for idx in [2, 4]:
        if phase[idx] < phi_threshold:
            theta[idx] = hip_lower
        else:
            theta[idx] = hip_upper

    # --- Map Front Flipper Oscillators (indices 3 and 5) ---
    # Use a sinusoidal mapping from the oscillator phase.
    theta[3] = flipper_mid_right + flipper_amp_right * np.sin(phase[3])
    theta[5] = flipper_mid_left  + flipper_amp_left  * np.sin(phase[5])
    
    # --- Map Back Flipper Oscillators (indices 0 and 1) ---
    # Similarly, use sinusoidal mapping.
    theta[0] = back_mid_right + back_amp_right * np.sin(phase[0])
    theta[1] = back_mid_left  + back_amp_left  * np.sin(phase[1])
    
    # Log the joint angles.
    theta_history.append(theta.copy())
    
    # Send the computed joint angles as position commands.
    for i, act_idx in enumerate(actuator_indices):
        data.ctrl[act_idx] = theta[i]
    
    # Step the simulation and update the viewer.
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)

# Convert log data to a numpy array and plot the joint angles.
theta_history = np.array(theta_history)  # shape: (sim_steps, num_osc)
plt.figure(figsize=(10, 6))
for i in range(num_osc):
    plt.plot(time_history, theta_history[:, i], label=f'Joint {i+1}')
plt.xlabel("Time (s)")
plt.ylabel("Joint Angle (rad)")
plt.title("CPG Synchronous Gait Joint Angles with Phase-Dependent Mapping")
plt.legend()
plt.grid(True)
plt.show()
