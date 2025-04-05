import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
from pymoo.core.evaluator import Evaluator  # Import Evaluator
from pymoo.core.population import Population # Import Population

# =============================================================================
#                               HOPF OSCILLATOR DYNAMICS
# =============================================================================
def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
    """
    Integrate one step of the Hopf oscillator (with coupling) using Euler's method.

    The base dynamics:
      dx/dt = α (μ - (x² + y²)) x - ω y
      dy/dt = α (μ - (x² + y²)) y + ω x

    The coupling term is added to dy:
      dy += λ * Δ_i,
    where
      Δ_i = Σ_{j≠i} [ y_j cos(θ_{ji}) - x_j sin(θ_{ji}) ],
      and θ_{ji} = 2π (φ_i - φ_j) with φ's defined in phase_offsets.
    """
    r_sq = x * x + y * y
    # Base Hopf oscillator dynamics (limit cycle)
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    # Compute coupling term using the desired phase offsets:
    delta = 0.0
    # Construct a list of phase offsets corresponding to the order in actuator_names
    phase_list = [phase_offsets[name] for name in actuator_names]
    for j in range(len(x_all)):
        if j == index:
            continue
        theta_ji = 2.0 * np.pi * (phase_list[index] - phase_list[j])
        delta += y_all[j] * np.cos(theta_ji) - x_all[j] * np.sin(theta_ji)
    dy += coupling * delta

    # Euler integration
    x_new = x + dx * dt
    y_new = y + dy * dt
    return x_new, y_new

# =============================================================================
#                               LOAD MUJOCO MODEL
# =============================================================================
model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/turtlev1/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# =============================================================================
#                           SENSOR & ACTUATOR LOOKUP
# =============================================================================
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
#               DEFINE ACTUATOR NAMES, INDICES, AND JOINT LIMITS
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

# For this example, we use the actuator control range from the model:
joint_limits = {}
for i, name in enumerate(actuator_names):
    ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
    joint_limits[name] = (ctrl_min, ctrl_max)
    print(f"{name}: ctrl range = [{ctrl_min:.3f}, {ctrl_max:.3f}]")

# =============================================================================
#           SELECT BODY FOR COM/ORIENTATION & TOTAL MASS
# =============================================================================
main_body_name = "base"  # Adjust as needed
try:
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)
except Exception as e:
    print("Could not find body named", main_body_name, ":", e)
    main_body_id = 0

base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
base_mass = model.body_mass[base_body_id]
total_mass = np.sum(base_mass)

# =============================================================================
#                       JOINT OUTPUT MAPPING FUNCTION
# =============================================================================
def map_oscillator_to_joint(x_val, min_angle, max_angle):
    """
    Linearly map x_val from [-sqrt(mu), sqrt(mu)] to [min_angle, max_angle] using:
      offset = (min_angle + max_angle)/2,
      gain   = (max_angle - min_angle)/2.
    """
    amplitude = np.sqrt(mu)
    offset = (min_angle + max_angle) / 2.0
    gain = (max_angle - min_angle) / 2.0
    desired_angle = offset + gain * (x_val / amplitude)
    return desired_angle

# Optionally, you can define a joint_output_map dictionary if you want custom gains/offsets:
joint_output_map = {
    "pos_frontleftflipper":  {"offset": 0.435, "gain": 3},
    "pos_frontrightflipper": {"offset": 0.435, "gain": 3},
    "pos_backleft":          {"offset": 0.435, "gain": 1.5},
    "pos_backright":         {"offset": 0.435, "gain": 1.5},
    "pos_frontlefthip":      {"offset": 0.435, "gain": 3},
    "pos_frontrighthip":     {"offset": 0.435, "gain": 3}
}

# =============================================================================
#                           PHASE OFFSETS
# =============================================================================
# Define the phase offsets - crucial for coordinated movement
phase_offsets = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.5,  # 180 degrees out of phase with left
    "pos_backleft":          0.25, # 90 degrees
    "pos_backright":         0.75, # 270 degrees
    "pos_frontlefthip":      0.0,
    "pos_frontrighthip":     0.5
}


# =============================================================================
#                           SIMULATION FUNCTION
# =============================================================================
def run_simulation(params, phase_offsets):
    """
    Runs a single simulation episode with the given parameters.  This function
    encapsulates the core simulation logic, making it suitable for use within
    an optimization loop.

    Args:
        params (list): A list of CPG parameters:
            [stance_freq, swing_freq, a_param, lambda_cpl, A_frontleftflipper,
             A_frontrightflipper, A_backleft, A_backright, A_frontlefthip, A_frontrighthip]
        phase_offsets (dict): Dictionary defining the phase offsets.

    Returns:
        tuple: (vx, dy, CoT, stability, sim_data)  A tuple containing the
               performance metrics and simulation data.
    """
    # Unpack parameters
    (stance_freq, swing_freq, a_param, lambda_cpl,
     A_frontleftflipper, A_frontrightflipper, A_backleft, A_backright,
     A_frontlefthip, A_frontrighthip) = params

    # Amplitude scaling for each actuator
    amplitude_factors = {
        "pos_frontleftflipper":  A_frontleftflipper,
        "pos_frontrightflipper": A_frontrightflipper,
        "pos_backleft":          A_backleft,
        "pos_backright":         A_backright,
        "pos_frontlefthip":      A_frontlefthip,
        "pos_frontrighthip":     A_frontrighthip
    }

    # Initialize oscillator states
    oscillators = {}
    mu = 0.5 # Define mu here.  It's a Hopf oscillator parameter.
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name] = {
            "x": np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn(),
            "y": np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()
        }

    # Set up time parameters
    dt_cpg = 0.001
    time_duration = 30.0
    alpha = 5 # Define alpha, another Hopf parameter
    omega = 1 # Define omega, another Hopf parameter

    # Data logging
    time_data = []
    ctrl_data = {name: [] for name in actuator_names}
    cpg_outputs = {name: [] for name in actuator_names}
    com_positions = []
    body_orientations = []
    power_consumption = []

    sensor_data_history = {
        "torque_backright": [],
        "torque_backleft": [],
        "torque_frontrighthip": [],
        "torque_frontrightflipper": [],
        "torque_frontlefthip": [],
        "torque_frontleftflipper": [],
        "base_acc": [],
        "base_gyro": []
    }
    jointact_sensor_map = {
        "torque_backright":           "sens_jointactfrc_backright",
        "torque_backleft":            "sens_jointactfrc_backleft",
        "torque_frontrighthip":     "sens_jointactfrc_frontrighthip",
        "torque_frontrightflipper": "sens_jointactfrc_frontrightflipper",
        "torque_frontlefthip":      "sens_jointactfrc_frontlefthip",
        "torque_frontleftflipper":  "sens_jointactfrc_frontleftflipper"
    }
    base_imu_map = {
        "base_gyro": "sens_base_gyro",
        "base_acc":   "sens_base_acc"
    }

    actuator_torque_history = {name: [] for name in actuator_names}
    joint_velocity_history  = {name: [] for name in actuator_names}

    # Simulation loop using the mujoco API
    start_time = time.time()
    sim_time = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and sim_time < time_duration:
            now = time.time()
            sim_time = now - start_time
            loop_dt = sim_time - (time_data[-1] if time_data else 0.0)

            # Integrate Hopf oscillators
            steps = int(np.floor(loop_dt / dt_cpg))
            for _ in range(steps):
                x_all = [oscillators[name]["x"] for name in actuator_names]
                y_all = [oscillators[name]["y"] for name in actuator_names]
                for i, name in enumerate(actuator_names):
                    x_i = oscillators[name]["x"]
                    y_i = oscillators[name]["y"]
                    # Compute instantaneous frequency
                    freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                           (swing_freq  / (1.0 + np.exp(a_param * y_i)))
                    x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                            lambda_cpl, x_all, y_all, i, phase_offsets)
                    oscillators[name]["x"] = x_new
                    oscillators[name]["y"] = y_new

            # Map oscillator outputs to joint controls
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                amplitude_factor = amplitude_factors[name]  # Get amplitude for this actuator.
                offset = joint_output_map[name]["offset"]
                gain = joint_output_map[name]["gain"]
                desired_angle = offset + gain * np.tanh(oscillators[name]["x"])
                desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
                data.ctrl[actuator_indices[name]] = desired_angle_clamped
                cpg_outputs[name].append(desired_angle_clamped)

            time_data.append(sim_time)

            # Step the simulation
            mujoco.mj_step(model, data)

            # Record data
            for name in actuator_names:
                ctrl_data[name].append(data.ctrl[actuator_indices[name]])
            com_positions.append(data.xpos[main_body_id].copy())
            body_orientations.append(data.xmat[main_body_id].copy())
            qvel = data.qvel[:model.nu]
            torque = data.actuator_force[:model.nu]
            instant_power = np.sum(np.abs(torque) * np.abs(qvel))
            power_consumption.append(instant_power)

            for varname, sname in jointact_sensor_map.items():
                val = get_sensor_data(data, model, sensor_name2id, sname)
                if val is not None:
                    sensor_data_history[varname].append(val[0])
            for varname, sname in base_imu_map.items():
                val = get_sensor_data(data, model, sensor_name2id, sname)
                if val is not None:
                    sensor_data_history[varname].append(val.copy())
            for name in actuator_names:
                idx = actuator_indices[name]
                actuator_torque_history[name].append(data.actuator_force[idx])
            for name in actuator_names:
                idx = actuator_indices[name]
                joint_velocity_history[name].append(qvel[idx])
            viewer.sync()

    # --------------------- Performance Metrics -----------------------
    final_time = time_data[-1] if len(time_data) > 0 else 0.0
    displacement = com_positions[-1] - com_positions[0] if len(com_positions) > 1 else np.zeros(3)
    distance_traveled = np.linalg.norm(displacement)
    avg_velocity = distance_traveled / final_time if final_time > 0 else 0.0
    dt_integration = dt_cpg  # Use the CPG timestep for integration
    total_energy = np.sum(power_consumption) * dt_integration if power_consumption else 0.0
    weight = total_mass * 9.81
    cost_of_transport = total_energy / (weight * distance_traveled) if distance_traveled > 0.01 else np.inf

    # Calculate lateral displacement (dy)
    y_positions = [com[1] for com in com_positions]  # Extract Y coordinates
    y_mean = np.mean(y_positions)
    dy = np.sqrt(np.mean((np.array(y_positions) - y_mean)**2)) if y_positions else 0.0

    # Calculate stability (S)
    roll_angles = []
    pitch_angles = []
    for mat in body_orientations:
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        r = mujoco.Mjr(data.qpos)  # Create a Mjr object
        mujoco.mjr_mat2euler(roll_angles, pitch_angles, None, mat)
    if roll_angles and pitch_angles:
      roll_var   = np.var(roll_angles)
      pitch_var  = np.var(pitch_angles)
      stability  = -(roll_var + pitch_var)
    else:
      stability = 0.0

    sim_data = {
        "time_data": time_data,
        "ctrl_data": ctrl_data,
        "cpg_outputs": cpg_outputs,
        "com_positions": com_positions,
        "body_orientations": body_orientations,
        "power_consumption": power_consumption,
        "sensor_data_history": sensor_data_history,
        "actuator_torque_history": actuator_torque_history,
        "joint_velocity_history": joint_velocity_history,
    }
    return avg_velocity, dy, cost_of_transport, stability, sim_data

# =============================================================================
#                           OBJECTIVE FUNCTION
# =============================================================================
class TurtleCPGProblem(Problem):
    """
    Define the multi-objective optimization problem.
    """
    def __init__(self, phase_offsets):
        super().__init__(
            n_var=10,  # Number of design variables
            n_obj=4,   # Number of objectives
            n_constr=0, # Number of constraints
            xl=np.array([1.0, 1.0, 5.0, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),  # Lower bounds
            xu=np.array([3.0, 3.0, 15.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])   # Upper bounds
        )
        self.phase_offsets = phase_offsets
        self.evaluator = Evaluator() # Initialize the evaluator

    def _evaluate(self, x, out):
        """
        Evaluate the objectives for a given set of design variables.

        Args:
            x (numpy.ndarray): A 2D array where each row represents a solution
                and each column a design variable.
            out (dict): A dictionary to store the objective values.
        """
        # Create a Population object from the input x
        pop = Population.new("X", x)

        # Evaluate the solutions using the evaluator
        F = self.evaluator.eval(
            problem=self, # Pass the problem instance
            pop=pop,
            func=lambda params: [run_simulation(p, self.phase_offsets) for p in params],
            mode='vectorized',  # Specify 'vectorized' mode
            nout=4
        )

        # Output the objectives
        out["F"] = np.column_stack([
            -np.array([f[0] for f in F]),  # Negative because we want to maximize velocity
            np.array([f[1] for f in F]),
            np.array([f[2] for f in F]),
            -np.array([f[3] for f in F]) # Negative because we want to maximize stability
        ])

# =============================================================================
#                           PARAMETER BOUNDS
# =============================================================================
# Already defined in the Problem class, but keeping here for reference
bounds = [
    (1.0, 3.0),    # stance_freq
    (1.0, 3.0),    # swing_freq
    (5.0, 15.0),   # a_param
    (0.1, 1.0),    # lambda_cpl
    (0.5, 5.0),    # A_frontleftflipper
    (0.5, 5.0),    # A_frontrightflipper
    (0.5, 5.0),    # A_backleft
    (0.5, 5.0),    # A_backright
    (0.5, 5.0),    # A_frontlefthip
    (0.5, 5.0)     # A_frontrighthip
]

# =============================================================================
#                           INITIAL GUESS
# =============================================================================
# Provide an initial guess for the parameters.  A good initial guess can
# help the optimizer converge more quickly and reliably.
initial_params = [
    2.0,    # stance_freq
    2.0,    # swing_freq
    10.0,   # a_param
    0.5,    # lambda_cpl
    1.0,    # A_frontleftflipper
    1.0,    # A_frontrightflipper
    1.0,    # A_backleft
    1.0,    # A_backright
    1.0,    # A_frontlefthip
    1.0     # A_frontrighthip
]

# =============================================================================
#                           OPTIMIZATION
# =============================================================================
# Perform the optimization using a suitable algorithm.  Here, we use
# NSGA-II, a popular evolutionary algorithm for multi-objective optimization.

# Initialize phase_offsets *before* creating the problem instance.
phase_offsets = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.5,
    "pos_backleft":          0.25,
    "pos_backright":         0.75,
    "pos_frontlefthip":      0.0,
    "pos_frontrighthip":     0.5
}
problem = TurtleCPGProblem(phase_offsets)

algorithm = NSGA2(
    pop_size=50,  # Population size
    n_offsprings=25, # Offspring size
    sampling=np.array([initial_params]), # Initial sample
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 50) # Number of generations

result = minimize(problem,
                  algorithm,
                  termination,
                  seed=1,
                  save_history=True,
                  verbose=True)

# =============================================================================
#                           PARETO FRONT PLOTTING
# =============================================================================
# Plot the Pareto front, in objective space
F = result.F
plt.figure(figsize=(8, 6))
plt.scatter(F[:, 0], F[:, 1], c="blue", s=30)
plt.xlabel("Velocity (m/s)")
plt.ylabel("Lateral Displacement (m)")
plt.title("Pareto Front")
plt.grid(True)
plt.show()

# You can also plot other combinations of objectives to visualize the trade-offs
plt.figure(figsize=(8, 6))
plt.scatter(F[:, 2], F[:, 3], c="red", s=30)
plt.xlabel("Cost of Transport")
plt.ylabel("Stability")
plt.title("Pareto Front")
plt.grid(True)
plt.show()


# Print the optimal parameters from the Pareto front
print("Optimal Solutions (Pareto Front):")
for i, x in enumerate(result.X):
    print(f"Solution {i + 1}:")
    print(f"  Stance Freq: {x[0]:.3f}, Swing Freq: {x[1]:.3f}, a_param: {x[2]:.3f}, lambda: {x[3]:.3f}")
    print(f"  A_frontleftflipper: {x[4]:.3f}, A_frontrightflipper: {x[5]:.3f}, A_backleft: {x[6]:.3f}, A_backright: {x[7]:.3f}")
    print(f"  A_frontlefthip: {x[8]:.3f}, A_frontrighthip: {x[9]:.3f}")
    print(f"  Velocity: {-F[i, 0]:.4f} m/s, Lateral Displacement: {F[i, 1]:.4f} m")
    print(f"  Cost of Transport: {F[i, 2]:.4f}, Stability: {F[i, 3]:.4f}")
