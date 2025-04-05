import mujoco
import mujoco.viewer
import time
import numpy as np

# -----------------------------------------------------------------------------
#                            LOAD MUJOCO MODEL
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
#           ACTUATORS & THEIR PHYSICAL RANGES (ctrlrange) FROM YOUR MODEL
# -----------------------------------------------------------------------------
# You provided these values (min, max) from the <position> tags.
joint_limits = {
    "pos_frontleftflipper":  (-1.57,  0.64),
    "pos_frontrightflipper": (-0.64,  1.571),
    "pos_backleft":          (-1.571, 0.524),
    "pos_backright":         (-0.524, 1.571),
    "pos_frontlefthip":      (-1.571, 1.22),
    "pos_frontrighthip":     (-1.22,  1.571)
}

# We'll extract actuator names (keys of our dictionary) in a fixed order:
actuator_names = list(joint_limits.keys())
actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

# -----------------------------------------------------------------------------
#            HOPF CPG CLASS WITH PHASE COUPLING
# -----------------------------------------------------------------------------
class CoupledHopfCPG:
    def __init__(self, num_oscillators=6, dt=0.01):
        self.num_oscillators = num_oscillators
        self.dt = dt
        
        # Hopf parameters
        self.alpha = 5.0   # Convergence speed
        self.mu    = 1.0   # Sets radius^2 => amplitude ~ sqrt(1.0)=1
        # We can give each oscillator a different base frequency if desired
        # Dividing by 2 just to make them a bit slower:
        self.omega = np.array([0.88, 0.88, 0.9, 0.9, 0.88, 0.88]) / 2.0
        
        # Initial states (x,y)
        # Using small random init so they don't all start identically at 0
        self.x = 0.1 * np.random.randn(num_oscillators)
        self.y = 0.1 * np.random.randn(num_oscillators)

        # We'll track "phases" only if we want to do a phase-based coupling
        # or visualize it. Here, we also do a direct cartesian approach.
        self.phases = np.arctan2(self.y, self.x)

        # Coupling matrix K_ij: a positive matrix leads to in-phase sync.
        self.K = np.full((num_oscillators, num_oscillators), np.pi)
        np.fill_diagonal(self.K, 0)  # no self-coupling

    def update(self):
        """
        Updates the Hopf oscillators (x_i, y_i) with simple phase coupling.
        Returns x[] as a convenient handle for 'angle' approximation,
        but you could also combine x,y or compute a direct phase if desired.
        """
        # Current radius per oscillator
        r = np.sqrt(self.x**2 + self.y**2)

        # Phase offsets = phases_i - phases_j
        # shape => (num_osc, num_osc)
        phase_diffs = np.subtract.outer(self.phases, self.phases)
        
        # Sum up coupling terms for each i: sum_j( K_ij * sin(phase_i - phase_j) )
        # We'll do the "phase sync" as a simple additive term in dx,dy
        phase_sync = np.sum(self.K * np.sin(phase_diffs), axis=1)

        # Hopf oscillator eqns with an additive "phase_sync" for dx
        dx = self.alpha*(self.mu - r**2)*self.x - self.omega*self.y + phase_sync
        dy = self.alpha*(self.mu - r**2)*self.y + self.omega*self.x
        
        # Euler integration
        self.x += dx * self.dt
        self.y += dy * self.dt

        # Update phases from new (x,y)
        self.phases = np.arctan2(self.y, self.x)

        # We'll return x[] as the primary "joint angle" source
        return self.x

# -----------------------------------------------------------------------------
#         INITIALIZE THE HOPF CPG FOR THE 6 ACTUATORS
# -----------------------------------------------------------------------------
cpg = CoupledHopfCPG(num_oscillators=6, dt=0.01)

# -----------------------------------------------------------------------------
#           HELPER: MAP x_i -> A JOINT ANGLE WITHIN [min,max]
# -----------------------------------------------------------------------------
def map_to_joint_range(joint_name, x_val):
    """
    Map x_val (which should be near [-1,1] at steady state) 
    into the actuator's control range [min, max].
    """
    min_angle, max_angle = joint_limits[joint_name]
    center     = 0.5*(min_angle + max_angle)
    half_range = 0.5*(max_angle - min_angle)

    # If the Hopf oscillator amplitude is ~1, x_val is in [-1,1].
    # So we can do:
    angle = center + half_range * x_val

    # Just to be safe, clamp to the [min_angle, max_angle]
    angle = np.clip(angle, min_angle, max_angle)
    return angle

# -----------------------------------------------------------------------------
#           GENERATE ACTUATOR POSITIONS FROM THE CPG
# -----------------------------------------------------------------------------
def turtle_motion_pattern():
    """
    Uses the CoupledHopfCPG to get a new x[] state, 
    then maps x_i -> each joint's angle within its physical limits.
    """
    x_vals = cpg.update()  # array of length 6
    
    outputs = {}
    for i, joint_name in enumerate(actuator_names):
        # For now, we only use x_i (not y_i). 
        # If you like, you could do a different function of (x_i, y_i).
        joint_angle = map_to_joint_range(joint_name, x_vals[i])
        outputs[joint_name] = joint_angle
    
    return outputs

# -----------------------------------------------------------------------------
#                           MAIN SIMULATION LOOP
# -----------------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        # 1) Get the updated angles from the Hopf CPG
        target_positions = turtle_motion_pattern()

        # 2) Apply position control to actuators
        for name, angle in target_positions.items():
            data.ctrl[actuator_indices[name]] = angle

        # 3) Step the simulation
        mujoco.mj_step(model, data)

        # 4) Sync the viewer
        viewer.sync()
