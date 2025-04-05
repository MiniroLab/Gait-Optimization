import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# =============================================================================
#           LOAD MUJOCO MODEL AND SET UP SENSOR/ACTUATOR LOOKUP
# =============================================================================

model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/turtlev1/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Build sensor lookup (if sensors are defined in the XML)
sensor_name2id = {}
for i in range(model.nsensor):
    name_adr = model.sensor_adr[i]
    name_chars = []
    for c in model.names[name_adr:]:
        if c == 0:
            break
        name_chars.append(chr(c))
    sensor_name = "".join(name_chars)
    sensor_name2id[sensor_name] = i

def get_sensor_data(data, model, sensor_name2id, sname):
    if sname not in sensor_name2id:
        return None
    sid = sensor_name2id[sname]
    dim = model.sensor_dim[sid]
    start_idx = model.sensor_adr[sid]
    return data.sensordata[start_idx : start_idx + dim].copy()

def get_actuator_index(model, name):
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

# =============================================================================
#             DEFINE ACTUATOR NAMES, INDICES, AND JOINT LIMITS
# =============================================================================

actuator_names = [
    "pos_frontleftflipper",
    "pos_frontrightflipper",
    "pos_backleft",
    "pos_backright",
    "pos_frontlefthip",
    "pos_frontrighthip"
]
actuator_indices = {name: get_actuator_index(model, name) for name in actuator_names}

joint_limits = {
    "pos_frontleftflipper":  (-0.3, 1.571),
    "pos_frontrightflipper": (-0.3, 1.571),
    "pos_backleft":          (-0.3, 1.571),
    "pos_backright":         (-0.3, 1.571),
    "pos_frontlefthip":      (-0.6, 1.571),
    "pos_frontrighthip":     (-0.6, 1.571)
}

# =============================================================================
#            HOPF OSCILLATOR PARAMETERS & INITIALIZATION
# =============================================================================

# Number of actuated joints (each driven by a Hopf oscillator)
N = 6

# Hopf oscillator constants
alpha = 10.0      # Convergence rate to the limit cycle
mu = 0.04          # amplitude^2 => amplitude = sqrt(mu) = 1.0

# Coupling strength coefficient (inter-oscillator sync)
K_cp = 0.8

# Frequency modulation parameters for gait phases
omega_swing = 1 * np.pi  # Swing phase frequency (e.g., 1 Hz -> 2π rad/s)
beta = 0.6              # Proportion of the gait cycle in the support phase
omega_stance = (1.0 - beta) / beta * omega_swing

# Exponential slope for the smooth transition between ω_stance and ω_swing
a_exp = 10.0

# Phase offsets to generate a desired gait pattern
# phase_offsets = [0.0, 0.0, 0.0, 0.0, 0.25, 0.25] # sync phase offsets
phase_offsets = [0.0, 0.5, 0.5, 0.0, 0.25, 0.75] # diag phase offsets

    # Phase offsets corresponding to each actuator:
    # "pos_frontleftflipper"  -> 0.0
    # "pos_frontrightflipper" -> 0.5
    # "pos_backleft"          -> 0.25
    # "pos_backright"         -> 0.75
    # "pos_frontlefthip"      -> 0.25
    # "pos_frontrighthip"     -> 0.75

# Initialize oscillator states (x_i, y_i) for each joint
x = np.zeros(N)
y = np.zeros(N)
for i in range(N):
    phase0 = phase_offsets[i] * 2.0 * np.pi
    x[i] = np.sqrt(mu) * np.cos(phase0)
    y[i] = np.sqrt(mu) * np.sin(phase0)

# =============================================================================
#            DEFINE FUNCTIONS FOR OSCILLATOR DYNAMICS & COUPLING
# =============================================================================

def compute_omega(yi):
    """
    Compute the instantaneous oscillator frequency ω based on y.
    Smoothly blend between ω_stance and ω_swing.
    """
    return (
        (omega_stance / (1.0 + np.exp(-a_exp * yi))) +
        (omega_swing  / (1.0 + np.exp( a_exp * yi)))
    )

def update_cpgs(x, y, dt):
    """
    Update the state of N coupled Hopf oscillators for one time step dt.
    Returns updated (x_next, y_next).
    """
    x_next = np.zeros_like(x)
    y_next = np.zeros_like(y)
    
    # Current amplitude and frequency
    r = np.sqrt(x**2 + y**2)
    omega_vals = [compute_omega(y[i]) for i in range(N)]
    
    # Precompute phase differences φ_ij = 2π(phase_offsets[i] - phase_offsets[j])
    phi_matrix = np.zeros((N, N))
    for i_osc in range(N):
        for j_osc in range(N):
            if i_osc != j_osc:
                phi_matrix[i_osc, j_osc] = 2.0 * np.pi * (
                    phase_offsets[i_osc] - phase_offsets[j_osc]
                )
    
    # Hopf + coupling
    for i_osc in range(N):
        w_i = omega_vals[i_osc]
        cx, cy = 0.0, 0.0
        for j_osc in range(N):
            if i_osc == j_osc:
                continue
            phi_ij = phi_matrix[i_osc, j_osc]
            cx += x[j_osc] * np.cos(phi_ij) - y[j_osc] * np.sin(phi_ij)
            cy += x[j_osc] * np.sin(phi_ij) + y[j_osc] * np.cos(phi_ij)
        
        dx = alpha * (mu - r[i_osc]**2) * x[i_osc] - w_i * y[i_osc] + K_cp * cx
        dy = alpha * (mu - r[i_osc]**2) * y[i_osc] + w_i * x[i_osc] + K_cp * cy
        
        x_next[i_osc] = x[i_osc] + dx * dt
        y_next[i_osc] = y[i_osc] + dy * dt
    
    return x_next, y_next

def map_oscillator_to_jointv1(x_val, min_angle, max_angle):
    """
    Linearly map x_val from [-sqrt(mu), +sqrt(mu)] to [min_angle, max_angle].
    This ensures the full joint range can be used if x_val covers the oscillator's
    entire amplitude range.
    """
    amplitude = np.sqrt(mu)
    # np.interp(value, [in_min, in_max], [out_min, out_max])
    return np.interp(x_val, [-amplitude, amplitude], [min_angle, max_angle])



def map_oscillator_to_joint(x_val, min_angle, max_angle):
    """
    Linearly map x_val from [-sqrt(mu), +sqrt(mu)] to [min_angle, max_angle].
    Uses the following approach:
    
      - offset = (min_angle + max_angle) / 2
      - gain   = (max_angle - min_angle) / 2
      - Normalize x_val by sqrt(mu) so that it is in [-1, 1]
      - desired_angle = offset + gain * (x_val / sqrt(mu))
    
    This ensures that when x_val reaches ±sqrt(mu), the joint command reaches the corresponding min or max.
    """
    amplitude = np.sqrt(mu)
    offset = (min_angle + max_angle) / 2.0
    gain = ((max_angle - min_angle) / 2.0)
    desired_angle = offset + gain * (x_val / amplitude)
    return desired_angle


# =============================================================================
#             SET UP TIME PARAMETERS AND DATA LOGGING
# =============================================================================

dt_cpg = 0.001      # Time step for oscillator updates
time_ds = 60.0     # Total simulation duration (seconds)

time_data = []
cpg_outputs = {name: [] for name in actuator_names}  # Joint commands over time

# We track instantaneous frequencies for demonstration:
inst_freq_history = [[] for _ in range(N)]
omega_st_history_val = []
omega_sw_history_val = []

# =============================================================================
#                        MAIN SIMULATION LOOP
# =============================================================================

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    last_loop_time = start_time

    while viewer.is_running():
        now = time.time()
        sim_time = now - start_time
        loop_dt = now - last_loop_time
        last_loop_time = now
        
        # End after 30 seconds
        if sim_time >= time_ds:
            print(f"Reached {time_ds:.1f} s of simulation.")
            break

        # ------------------------------------------------------------
        #  Multiple oscillator sub-steps, based on elapsed real time
        # ------------------------------------------------------------
        steps = int(np.floor(loop_dt / dt_cpg))
        for _ in range(steps):
            x, y = update_cpgs(x, y, dt_cpg)

        # ------------------------------------------------------------
        #  Record instantaneous frequencies (after final sub-step)
        # ------------------------------------------------------------
        for i_osc in range(N):
            current_omega = compute_omega(y[i_osc])
            inst_freq_history[i_osc].append(current_omega)
        omega_st_history_val.append(omega_stance)
        omega_sw_history_val.append(omega_swing)
        
        # ------------------------------------------------------------
        #  Map oscillator outputs to joint control commands
        # ------------------------------------------------------------
        # for i_osc, name in enumerate(actuator_names):
        #     min_angle, max_angle = joint_limits[name]
        #     # Convert x[i_osc] from [-√mu, +√mu] -> [min_angle, max_angle]
        #     desired_angle = map_oscillator_to_joint(x[i_osc], min_angle, max_angle)
        #     data.ctrl[actuator_indices[name]] = desired_angle
        #     cpg_outputs[name].append(desired_angle)


        # ------------------------------------------------------------
        #  Map oscillator outputs to joint control commands
        # ------------------------------------------------------------
        for i_osc, name in enumerate(actuator_names):
            min_angle, max_angle = joint_limits[name]
            # Convert x[i_osc] from [-sqrt(mu), +sqrt(mu)] to the full joint range
            desired_angle = map_oscillator_to_joint(x[i_osc], min_angle, max_angle)
            # Clamp the output to be safe (though the mapping should cover the range)
            desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
            data.ctrl[actuator_indices[name]] = desired_angle_clamped
            cpg_outputs[name].append(desired_angle_clamped)


        time_data.append(sim_time)
        
        # Step MuJoCo simulation forward
        mujoco.mj_step(model, data)
        viewer.sync()

# =============================================================================
#                         PLOTTING AFTER SIMULATION
# =============================================================================

# Plot joint commands driven by Hopf CPGs
plt.figure(figsize=(10, 6))
for name in actuator_names:
    plt.plot(time_data, cpg_outputs[name], label=name)
plt.title('Joint Commands Driven by Hopf CPGs (Full Joint Range)')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the evolution of oscillator frequencies
plt.figure(figsize=(10, 6))
for i_osc in range(N):
    plt.plot(time_data, inst_freq_history[i_osc], label=f'ω_inst_{i_osc}')
plt.plot(time_data, omega_st_history_val, 'k--', label='ω_stance (ref)')
plt.plot(time_data, omega_sw_history_val, 'r--', label='ω_swing (ref)')
plt.title('Oscillator Frequencies Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (rad/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
