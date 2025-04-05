import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#                           HOPF OSCILLATOR DYNAMICS
# -----------------------------------------------------------------------------
def hopf_step(x, y, alpha, mu, omega, dt, coupling, xall, yall, index):
    """
    Integrate one step of the Hopf oscillator (with coupling) using Euler's method.
    """
    r_sq = x*x + y*y
    
    # Base Hopf dynamics (limit cycle)
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x
    
    # Add linear coupling from other oscillators
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
#                         LOAD MUJOCO MODEL
# -----------------------------------------------------------------------------
model_path = "c:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/assets/turtlev1/testrobot1.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

def get_actuator_index(model, name):
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

# -----------------------------------------------------------------------------
#            DEFINE ACTUATOR NAMES & GET INDICES
# -----------------------------------------------------------------------------
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
#        CPG PARAMETERS & INITIAL OSCILLATOR STATES
# -----------------------------------------------------------------------------
alpha    = 10.0   # Gain, convergence speed
mu       = 0.04   # radius^2 => amplitude ~ sqrt(0.04) = 0.2
a_param  = 10.0   # logistic steepness for stance/swing freq blending

# We define separate stance & swing frequencies (rad/s)
stance_freq = 2.0
swing_freq  = 4.0

# Random initial states so they don't start in exact sync
oscillators = {}
for name in actuator_names:
    oscillators[name] = {
        "x": 0.02 * np.random.randn(),
        "y": 0.02 * np.random.randn()
    }

# -----------------------------------------------------------------------------
#        COUPLING MATRIX (K) FOR IN-PHASE VS. OUT-OF-PHASE
# -----------------------------------------------------------------------------
num_joints = len(actuator_names)
K = np.zeros((num_joints, num_joints))

in_phase_coupling  = 0.8
out_phase_coupling = -0.8

left_indices  = [0, 2, 4]  # frontleftflipper, backleft, frontlefthip
right_indices = [1, 3, 5]  # frontrightflipper, backright, frontrighthip

for i in range(num_joints):
    for j in range(num_joints):
        if i == j:
            continue
        both_left  = (i in left_indices) and (j in left_indices)
        both_right = (i in right_indices) and (j in right_indices)
        cross_side = (i in left_indices and j in right_indices) or \
                     (i in right_indices and j in left_indices)
        
        if both_left or both_right:
            K[i, j] = in_phase_coupling
        elif cross_side:
            K[i, j] = out_phase_coupling

# -----------------------------------------------------------------------------
#        MAPPING OSCILLATOR STATE -> JOINT ANGLE (+ PHASE OFFSET)
# -----------------------------------------------------------------------------
# In this mapping, we apply a user-defined "phase_offset" to each joint so that:
#   x_phase = x * cos(phase_offset) - y * sin(phase_offset)
# and then compute the joint angle as:
#   angle = offset + gain * x_phase
# This approach allows fine tuning of the relative timing between joint motions.


# Configuration for a gait where the flipper oscillators remain in sync and the hips are shifted backward.
phase_offsets_syncbackwards = {
    # Flippers: both front and rear are in-phase (0 rad offset).
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.0,
    "pos_backleft":          0.0,
    "pos_backright":         0.0,
    # Hips: shifted by -pi/2 to ensure they point downward.
    "pos_frontlefthip":     -np.pi/2,
    "pos_frontrighthip":    -np.pi/2
}

# Configuration for turning left:
phase_offsets_turnleft = {
    # Flippers: front left flipper remains at 0, front right is shifted by pi (180°) to generate an asymmetry.
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": np.pi,
    # Rear flippers: back left is shifted by -pi, while back right remains at 0.
    "pos_backleft":         -np.pi,
    "pos_backright":         0.0,
    # Hips: both are shifted by -pi/2 to maintain a downward orientation.
    "pos_frontlefthip":     -np.pi/2,
    "pos_frontrighthip":    -np.pi/2
}

# Configuration for turning right:
phase_offsets_turnright = {
    # Flippers: front left at 0, front right at -pi to induce the opposite asymmetry.
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": -np.pi,
    # Rear flippers: back left shifted by pi, back right remains at 0.
    "pos_backleft":          np.pi,
    "pos_backright":         0.0,
    # Hips: both are shifted by +pi/2 so that they point downward.
    "pos_frontlefthip":      np.pi/2,
    "pos_frontrighthip":     np.pi/2
}

# Configuration for a balanced turn (moderate offset):
phase_offsets_turn = {
    # Flippers: moderate offset (pi/4) for both front flippers.
    "pos_frontleftflipper":  np.pi/4,
    "pos_frontrightflipper": np.pi/4,  # same for right front flipper
    # Rear flippers: both set to pi to create contrast with the front.
    "pos_backleft":          np.pi,
    "pos_backright":         np.pi,
    # Hips: one hip is shifted downward (-pi/2) and the other upward (pi/2) to induce turning.
    "pos_frontlefthip":     -np.pi/2,
    "pos_frontrighthip":     np.pi/2
}

# Configuration for a forward synchronized gait:
phase_offsets_syncforward = {
    # Flippers: all set to pi so that their motion is shifted by 180° compared to the baseline.
    "pos_frontleftflipper":  np.pi,
    "pos_frontrightflipper": np.pi,
    "pos_backleft":          np.pi,
    "pos_backright":         np.pi,
    # Hips: both shifted by -pi/2 to maintain a downward orientation.
    "pos_frontlefthip":     -np.pi/2,
    "pos_frontrighthip":    -np.pi/2
}

# phase_offsets_diagforward = {
#     # Suppose we want flippers in-phase with each other, so 0 for left, maybe pi for right, etc.
#     # Tweak as desired to anchor hips downward while flippers sweep.
#     "pos_frontleftflipper":  np.pi/4,
#     "pos_frontrightflipper": 3*np.pi/4, #out-of-phase with left flipper
#     "pos_backleft":          -np.pi,  # or some shift you want
#     "pos_backright":        np.pi,
#     "pos_frontlefthip":     3*np.pi/4,  # e.g., hips trailing or leading
#     "pos_frontrighthip":    -3*np.pi/4
# }

# Configuration for an alternating diagonal gait (backward diagonal):
phase_offsets_diagbackward = {
    # Diagonal pairing: front left and back right are in-phase,
    # while front right is shifted by -pi and back left is shifted by pi.
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": -np.pi,  # opposite phase to front left
    "pos_backleft":          np.pi,   # opposite phase to back right
    "pos_backright":         0.0,
    # Hips: assign complementary offsets to maintain downward orientation.
    "pos_frontlefthip":     -np.pi/2,
    "pos_frontrighthip":     np.pi/2
}

# Configuration for an alternating diagonal gait (forward diagonal):
phase_offsets_diagforward = {
    # Diagonal pairing: front left and back right are out-of-phase with front right and back left.
    # Here, front left is shifted by -pi, front right remains at 0,
    # back left remains at 0, and back right is shifted by pi.
    "pos_frontleftflipper":  -np.pi,
    "pos_frontrightflipper":  0.0,      # opposite phase relative to front left
    "pos_backleft":           0.0,      # in-phase with front right
    "pos_backright":          np.pi,    # opposite phase relative to front right
    # Hips: both are shifted so as to maintain a downward orientation.
    "pos_frontlefthip":      -np.pi/2,
    "pos_frontrighthip":      np.pi/2
}

# We still have a linear offset/gain for each actuator's final angle.
joint_output_map = {
    "pos_frontleftflipper":  {"offset": -0.8, "gain": 3.0},
    "pos_frontrightflipper": {"offset":  0.8, "gain": 3.0},
    "pos_backleft":          {"offset": -0.5, "gain": 1.0},
    "pos_backright":         {"offset":  0.5, "gain": 1.0},
    "pos_frontlefthip":      {"offset":  0.3, "gain": -1.0},
    "pos_frontrighthip":     {"offset":  0.3, "gain":  1.0}
}

# -----------------------------------------------------------------------------
#                       DATA COLLECTION FOR PLOTTING
# -----------------------------------------------------------------------------
time_data = []
ctrl_data = {name: [] for name in actuator_names}

# -----------------------------------------------------------------------------
#                       MAIN SIMULATION LOOP
# -----------------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    dt_cpg = 0.001
    last_loop_time = time.time()

    while viewer.is_running():
        now = time.time()
        loop_dt = now - last_loop_time
        last_loop_time = now
        
        # ---------------------------------------------------------------------
        # 1) Integrate the Hopf oscillators with stance/swing frequency
        # ---------------------------------------------------------------------
        steps = int(np.floor(loop_dt / dt_cpg))
        
        for _ in range(steps):
            x_all = [oscillators[name]["x"] for name in actuator_names]
            y_all = [oscillators[name]["y"] for name in actuator_names]

            for i, name in enumerate(actuator_names):
                x_i = oscillators[name]["x"]
                y_i = oscillators[name]["y"]
                
                # Blend stance & swing frequencies with logistic function:
                # If y_i > 0 => more stance freq; if y_i < 0 => more swing freq.
                freq = (stance_freq / (1.0 + np.exp(-a_param*y_i))) + \
                       (swing_freq  / (1.0 + np.exp( a_param*y_i)))

                x_new, y_new = hopf_step(
                    x_i, y_i, alpha, mu, freq, dt_cpg,
                    K, x_all, y_all, i
                )
                oscillators[name]["x"] = x_new
                oscillators[name]["y"] = y_new

        # ---------------------------------------------------------------------
        # 2) Convert oscillator states to joint angles & apply control
        # ---------------------------------------------------------------------
        for name in actuator_names:
            x_i = oscillators[name]["x"]
            y_i = oscillators[name]["y"]

            # Phase offset output
            delta_phi = phase_offsets_diagforward[name]
            x_phase = x_i * np.cos(delta_phi) - y_i * np.sin(delta_phi)

            # Then map to final angle
            offset = joint_output_map[name]["offset"]
            gain   = joint_output_map[name]["gain"]
            angle_raw = offset + gain * x_phase

            # Clamp
            min_angle, max_angle = joint_limits[name]
            angle_clamped = np.clip(angle_raw, min_angle, max_angle)
            
            data.ctrl[actuator_indices[name]] = angle_clamped
        
        # ---------------------------------------------------------------------
        # 3) Step Mujoco simulation and record data
        # ---------------------------------------------------------------------
        mujoco.mj_step(model, data)
        sim_time = now - start_time

        # Record data
        time_data.append(sim_time)
        for name in actuator_names:
            ctrl_data[name].append(data.ctrl[actuator_indices[name]])

        viewer.sync()

# -----------------------------------------------------------------------------
#                           PLOTTING AFTER SIMULATION
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for name in actuator_names:
    plt.plot(time_data, ctrl_data[name], label=name)

plt.title('Joint Control with Phase Offsets')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
