import mujoco
import mujoco.viewer
import time
import numpy as np

# -----------------------------------------------------------------------------
#                             HOPF CPG CONTROLLER
# -----------------------------------------------------------------------------

def hopf_step(x, y, alpha, mu, omega, dt):
    """
    Integrate one step of the Hopf oscillator using a simple Euler method.
    
    d(x)/dt = alpha*(mu - (x^2 + y^2))*x - omega*y
    d(y)/dt = alpha*(mu - (x^2 + y^2))*y + omega*x
    
    :param x, y: current oscillator states
    :param alpha: convergence/gain parameter
    :param mu: controls steady-state amplitude^2
    :param omega: intrinsic oscillator frequency
    :param dt: integration time step
    :return: updated (x, y)
    """
    r_sq = x*x + y*y
    dx = alpha*(mu - r_sq)*x - omega*y
    dy = alpha*(mu - r_sq)*y + omega*x

    # Euler integration
    x_new = x + dx * dt
    y_new = y + dy * dt
    return x_new, y_new

# -----------------------------------------------------------------------------
#                           MUJOCO MODEL SETUP
# -----------------------------------------------------------------------------

model_path = 'c:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/assets/turtlev1/testrobot1.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Helper function to get actuator indices by name
def get_actuator_index(model, name):
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

# Define your relevant actuator names here
actuator_names = [
    "pos_frontleftflipper", "pos_frontrightflipper", 
    "pos_backleft", "pos_backright", 
    "pos_frontlefthip", "pos_frontrighthip"
]

# Get actuator indices
actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

# -----------------------------------------------------------------------------
#                       CPG PARAMETERS & INITIAL STATES
# -----------------------------------------------------------------------------

# Global Hopf parameters (feel free to tune)
alpha = 10.0   # gain (convergence rate)
mu    = 0.04   # sets steady-state amplitude^2 = mu => amplitude ~ sqrt(mu)
base_freq = 0.8  # base oscillator frequency (rad/s) for each joint

# Per-actuator oscillator states and frequencies:
# You can adjust "omega" to create different base frequencies or phase offsets
# or keep them identical for a synchronous gait. 
cpg_state = {}
for name in actuator_names:
    cpg_state[name] = {
        "x": 0.01 * np.random.randn(),  # small random init
        "y": 0.01 * np.random.randn(),
        "omega": base_freq  # same freq => synchronous if started similarly
    }

# Mapping from oscillator state -> joint angle
# Each joint can have a custom offset and gain
# angle = offset + gain * x   (where x is the x-component of Hopf state)
actuator_output_map = {
    "pos_frontleftflipper":  {"offset": -1.0,  "gain": 0.5},
    "pos_frontrightflipper": {"offset": 1.0,  "gain": 0.5},
    "pos_backleft":          {"offset": -1.3,  "gain": 0.4},
    "pos_backright":         {"offset": 1.3,  "gain": 0.4},
    "pos_frontlefthip":      {"offset": 0.8,  "gain": 0.3},
    "pos_frontrighthip":     {"offset": 0.8,  "gain": 0.3}
}

# -----------------------------------------------------------------------------
#                           MAIN SIMULATION LOOP
# -----------------------------------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    # Use the model timestep if you like, or choose a small dt for the oscillator
    # integration. Often dt is in the order of 0.001 for stable numeric updates.
    dt_cpg = 0.01  
    
    last_update_time = time.time()
    
    while viewer.is_running():
        now = time.time()
        loop_dt = now - last_update_time
        last_update_time = now

        # --- 1) Integrate each Hopf oscillator by however many small steps fit in loop_dt ---
        steps = int(np.floor(loop_dt / dt_cpg))
        for _ in range(steps):
            for name in actuator_names:
                x_old = cpg_state[name]["x"]
                y_old = cpg_state[name]["y"]
                omega = cpg_state[name]["omega"]
                x_new, y_new = hopf_step(x_old, y_old, alpha, mu, omega, dt_cpg)
                cpg_state[name]["x"] = x_new
                cpg_state[name]["y"] = y_new
        
        # --- 2) Compute new target positions from the Hopf states ---
        for name in actuator_names:
            x_val = cpg_state[name]["x"]
            offset = actuator_output_map[name]["offset"]
            gain   = actuator_output_map[name]["gain"]
            
            # Final command to the joint (you can adjust to use y, or combine x+y, etc.)
            cmd = offset + gain * x_val
            
            data.ctrl[actuator_indices[name]] = cmd

        # --- 3) Step the Mujoco simulation once ---
        mujoco.mj_step(model, data)

        # --- 4) Sync the viewer so we can see it ---
        viewer.sync()
