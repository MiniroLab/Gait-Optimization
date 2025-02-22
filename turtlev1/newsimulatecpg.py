import mujoco
import mujoco.viewer
import time
import numpy as np

# -----------------------------------------------------------------------------
#                           HOPF OSCILLATOR DYNAMICS
# -----------------------------------------------------------------------------
def hopf_step(x, y, alpha, mu, omega, dt, coupling, xall, yall, index):
    """
    Integrate one step of the Hopf oscillator (with coupling) using Euler's method.

    x, y: current oscillator state for this joint
    alpha: convergence/gain parameter
    mu: radius^2 of limit cycle
    omega: intrinsic frequency (rad/s)
    dt: integration time step
    coupling: 2D matrix K_{ij} of coupling strengths
    xall, yall: lists of x,y states for all oscillators
    index: which oscillator we're updating

    Returns: (x_new, y_new)
    """
    # Current radius squared
    r_sq = x*x + y*y
    
    # Base Hopf dynamics
    dx = alpha*(mu - r_sq)*x - omega * y
    dy = alpha*(mu - r_sq)*y + omega * x
    
    # Add linear coupling from all other oscillators
    for j in range(len(xall)):
        if j == index:
            continue
        K_ij = coupling[index, j]
        dx += K_ij * (xall[j] - x)
        dy += K_ij * (yall[j] - y)
    
    # Euler integration
    x_new = x + dx * dt
    y_new = y + dy * dt
    return x_new, y_new


# -----------------------------------------------------------------------------
#                           LOAD MUJOCO MODEL
# -----------------------------------------------------------------------------
model_path = 'c:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/assets/turtlev1/testrobot1.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

def get_actuator_index(model, name):
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")


# -----------------------------------------------------------------------------
#             DEFINE ACTUATOR NAMES & GET INDICES
# -----------------------------------------------------------------------------
# We'll keep the same order for building our coupling matrix
actuator_names = [
    "pos_frontleftflipper",
    "pos_frontrightflipper",
    "pos_backleft",
    "pos_backright",
    "pos_frontlefthip",
    "pos_frontrighthip"
]
actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}


# -----------------------------------------------------------------------------
#          JOINT LIMITS (ctrlrange) PER ACTUATOR (from your MJCF)
# -----------------------------------------------------------------------------
joint_limits = {
    "pos_frontleftflipper":  (-1.57,  0.64),
    "pos_frontrightflipper": (-0.64,  1.571),
    "pos_backleft":          (-1.571, 0.524),
    "pos_backright":         (-0.524, 1.571),
    "pos_frontlefthip":      (-1.571, 1.22),
    "pos_frontrighthip":     (-1.22,  1.571)
}


# -----------------------------------------------------------------------------
#           CPG PARAMETERS & INITIAL OSCILLATOR STATES
# -----------------------------------------------------------------------------
alpha     = 10.0   # Gain, convergence speed
mu        = 0.08   # sets amplitude^2 => amplitude ~ sqrt(0.04) = 0.2
base_freq = 0.8    # baseline oscillator frequency (rad/s)
global_freq_scale = 2.5  # scale all oscillator frequencies if desired

oscillators = {}
for name in actuator_names:
    # small random init so they don't all start at exactly the same point
    oscillators[name] = {
        "x": 0.02 * np.random.randn(),
        "y": 0.02 * np.random.randn(),
        "omega": base_freq   # can individually adjust if needed
    }


# -----------------------------------------------------------------------------
#        COUPLING MATRIX (K) FOR IN-PHASE VS. OUT-OF-PHASE
# -----------------------------------------------------------------------------
#
#   Indices: 
#     0 -> pos_frontleftflipper
#     1 -> pos_frontrightflipper
#     2 -> pos_backleft
#     3 -> pos_backright
#     4 -> pos_frontlefthip
#     5 -> pos_frontrighthip
#
#   We'll define "left side" as indices [0, 2, 4] and "right side" as [1, 3, 5].
#   We want in-phase on same side => positive coupling
#   We want out-of-phase across sides => negative coupling
#   Magnitude of coupling can be tuned.

num_joints = len(actuator_names)
K = np.zeros((num_joints, num_joints))

in_phase_coupling  = 0.3  # same side
out_phase_coupling = -0.3 # opposite side

# Helper sets of indices
left_indices  = [0, 2, 4]  # frontleftflipper, backleft, frontlefthip
right_indices = [1, 3, 5]  # frontrightflipper, backright, frontrighthip

for i in range(num_joints):
    for j in range(num_joints):
        if i == j:
            continue  # no self-coupling
        both_left  = (i in left_indices) and (j in left_indices)
        both_right = (i in right_indices) and (j in right_indices)
        cross_side = (i in left_indices and j in right_indices) or \
                     (i in right_indices and j in left_indices)
        
        if both_left or both_right:
            # same side => in-phase
            K[i, j] = in_phase_coupling
        elif cross_side:
            # opposite side => out-of-phase
            K[i, j] = out_phase_coupling


# -----------------------------------------------------------------------------
#        MAPPING FROM OSCILLATOR STATE -> JOINT ANGLE
# -----------------------------------------------------------------------------
# We'll define angle = offset + gain * x_i, then clamp to [min, max].
# You can tweak offset/gain if you want a different nominal posture or amplitude.
joint_output_map = {
    "pos_frontleftflipper":  {"offset": -0.8, "gain": 3.0},  # example offset
    "pos_frontrightflipper": {"offset":  0.8, "gain": 3.0},
    "pos_backleft":          {"offset": -0.5,  "gain": 1.0},
    "pos_backright":         {"offset":  0.5,  "gain": 1.0},
    "pos_frontlefthip":      {"offset":  0.3,  "gain": -1.0},
    "pos_frontrighthip":     {"offset":  0.3,  "gain": 1.0}
}

# You can always refine these offsets/gains after testing 
# to ensure the motion is well-centered in each joint range.

# -----------------------------------------------------------------------------
#                          MAIN SIMULATION LOOP
# -----------------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()

    # We'll do a smaller integration step for the CPG
    dt_cpg = 0.001  
    
    last_loop_time = time.time()

    while viewer.is_running():
        now = time.time()
        loop_dt = now - last_loop_time
        last_loop_time = now
        
        # ---------------------------------------------------------------------
        # 1) Integrate the Hopf oscillators
        #    We'll do multiple small sub-steps if loop_dt is large
        # ---------------------------------------------------------------------
        steps = int(np.floor(loop_dt / dt_cpg))
        for _ in range(steps):
            # Gather all x, y for coupling
            x_all = [oscillators[name]["x"] for name in actuator_names]
            y_all = [oscillators[name]["y"] for name in actuator_names]

            for i, name in enumerate(actuator_names):
                x_i = oscillators[name]["x"]
                y_i = oscillators[name]["y"]
                # scale the intrinsic freq if needed
                omega_i = oscillators[name]["omega"] * global_freq_scale

                x_new, y_new = hopf_step(
                    x_i, y_i, alpha, mu, omega_i, dt_cpg, 
                    K, x_all, y_all, i
                )
                oscillators[name]["x"] = x_new
                oscillators[name]["y"] = y_new

        # ---------------------------------------------------------------------
        # 2) Convert oscillator states to joint angles & apply control
        # ---------------------------------------------------------------------
        for name in actuator_names:
            x_i = oscillators[name]["x"]
            offset = joint_output_map[name]["offset"]
            gain   = joint_output_map[name]["gain"]

            angle_raw = offset + gain * x_i  # simple linear mapping

            # Clamp to [min, max]
            min_angle, max_angle = joint_limits[name]
            angle_clamped = np.clip(angle_raw, min_angle, max_angle)

            data.ctrl[actuator_indices[name]] = angle_clamped

        # ---------------------------------------------------------------------
        # 3) Step Mujoco simulation once and sync the viewer
        # ---------------------------------------------------------------------
        mujoco.mj_step(model, data)
        viewer.sync()
