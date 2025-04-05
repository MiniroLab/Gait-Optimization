#!/usr/bin/env python
"""
Multi-Objective Bayesian Optimization for CPG Parameters in a Sea Turtle-Inspired Robot

This script uses BoTorch and MuJoCo to optimize parameters of a CPG-based controller.
The decision vector (params) consists of:
    [omega_stance, omega_swing, a_exp, lambda, A_scale]
with the following bounds:
    omega_stance: [1.0, 3.0] rad/s
    omega_swing:  [1.0, 3.0] rad/s
    a_exp:        [5.0, 15.0]
    lambda:       [0.1, 1.0]
    A_scale:      [0.5, 5.0]

The objectives are computed as:
    f1 = v_x         (maximize forward velocity)
    f2 = - d_y       (minimize lateral displacement)
    f3 = - CoT       (minimize cost of transport)
    f4 = S           (maximize stability)

The energy consumption E is computed as:
    E = sum_{i=1}^{n_a} ∫ |τ_i(t) * ω_i(t)| dt,
and
    CoT = E / (m * g * d)
where m is the robot’s mass, g is gravitational acceleration, and d is the distance traveled.

Dependencies: numpy, torch, gpytorch, botorch, matplotlib, mujoco, mujoco.viewer
"""

import time
import botorch.fit
import botorch.optim.fit
import numpy as np
import torch
import gpytorch
import botorch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

# =============================================================================
#                           SIMULATION FUNCTIONS
# =============================================================================

# --- Hopf Oscillator Dynamics ---
def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets, actuator_names):
    """
    Integrate one step of the Hopf oscillator with coupling using Euler's method.
    """
    r_sq = x * x + y * y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    # Compute coupling term using phase offsets:
    delta = 0.0
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

# --- Simulation Evaluation Function ---
def simulate_robot(params, headless=True):
    """
    Run the MuJoCo simulation with given CPG parameters and return objective metrics.
    params: [omega_stance, omega_swing, a_exp, lambda, A_scale]
    Returns: objectives as a numpy array [v_x, -d_y, -CoT, S]
    """
    # Unpack parameters
    omega_stance, omega_swing, a_exp, coupling_param, amplitude_scale = params

    # Load MuJoCo model and data
    model_path = (
        "c:/Users/chike/Box/TurtleRobotExperiments/"
        "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
        "assets/turtlev1/testrobot1.xml"
    )
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Define actuator names and phase offsets
    actuator_names = [
        "pos_frontleftflipper",
        "pos_frontrightflipper",
        "pos_backleft",
        "pos_backright",
        "pos_frontlefthip",
        "pos_frontrighthip"
    ]
    # Use diagonal phase offsets as an example
    phase_offsets = {
        "pos_frontleftflipper":     0.0,
        "pos_frontrightflipper": 0.5,
        "pos_backleft":             0.5,
        "pos_backright":            0.0,
        "pos_frontlefthip":         0.75,
        "pos_frontrighthip":        0.25
    }

    # Joint limits (extracted from model)
    joint_limits = {}
    for i, name in enumerate(actuator_names):
        ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
        joint_limits[name] = (ctrl_min, ctrl_max)

    # Initialize oscillator states for each actuator using phase offsets.
    mu_val = 0.04
    alpha_val = 10.0
    oscillators = {}
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name] = {
            "x": np.sqrt(mu_val) * np.cos(phase0) + 0.002 * np.random.randn(),
            "y": np.sqrt(mu_val) * np.sin(phase0) + 0.002 * np.random.randn()
        }

    # Set simulation time parameters
    dt_cpg = 0.001
    sim_duration = 30.0  # seconds
    start_time_sim = time.time()
    sim_time = 0.0

    # Data logging arrays
    time_data = []
    com_positions = []
    power_consumption = []
    orientations = []  # We'll store roll and pitch from the main body

    # For energy, we sum instantaneous power = sum(|torque * joint_velocity|) over all actuators
    total_energy = 0.0

    # Main body for COM, orientation, etc.
    main_body_name = "base"
    main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)
    if main_body_id == -1:
        print(f"Warning: Main body '{main_body_name}' not found. Using body index 0 for orientation.")
        main_body_id = 0

    # Headless simulation loop
    while sim_time < sim_duration:
        # Compute simulation time step count based on dt_cpg
        steps = int(np.floor(0.01 / dt_cpg))  # Use 0.01 s wall clock time per loop iteration
        for _ in range(steps):
            # Integrate CPG for each oscillator
            x_all = [oscillators[name]["x"] for name in actuator_names]
            y_all = [oscillators[name]["y"] for name in actuator_names]
            for i, name in enumerate(actuator_names):
                # Determine instantaneous frequency using logistic blending:
                # When oscillator y is low, use omega_stance; when high, use omega_swing.
                y_val = oscillators[name]["y"]
                freq = (omega_stance / (1.0 + np.exp(-a_exp * y_val))) + \
                       (omega_swing  / (1.0 + np.exp(a_exp * y_val)))
                x_new, y_new = hopf_step(oscillators[name]["x"], oscillators[name]["y"],
                                         alpha_val, mu_val, freq, dt_cpg,
                                         coupling_param, x_all, y_all, i, phase_offsets, actuator_names)
                oscillators[name]["x"] = x_new
                oscillators[name]["y"] = y_new

            # Map oscillator outputs to joint commands
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                offset = (min_angle + max_angle) / 2.0
                gain = (max_angle - min_angle) / 2.0
                desired_angle = offset + gain * np.tanh(oscillators[name]["x"]) * amplitude_scale
                desired_angle = np.clip(desired_angle, min_angle, max_angle)
                actuator_id = i = actuator_names.index(name)
                data.ctrl[actuator_id] = desired_angle

            # Step simulation
            mujoco.mj_step(model, data)
            sim_time += dt_cpg

            # Log time and COM position
            time_data.append(sim_time)
            com_positions.append(data.xpos[main_body_id].copy())
            orientations.append(data.xmat[main_body_id].copy())

            # Log power consumption: sum(|torque * joint_velocity|)
            qvel = data.qvel[:model.nu]
            torque = data.actuator_force[:model.nu]
            inst_power = np.sum(np.abs(torque) * np.abs(qvel))
            power_consumption.append(inst_power)
            total_energy += inst_power * dt_cpg

    # Compute performance metrics:
    com_positions = np.array(com_positions)
    if len(com_positions) >= 2:
        displacement = com_positions[-1] - com_positions[0]
        distance_traveled = np.linalg.norm(displacement)
    else:
        distance_traveled = 0.0

    # Forward velocity
    v_x = distance_traveled / sim_duration

    # Lateral displacement (assume Y-axis deviation)
    y_positions = com_positions[:, 1]
    d_y = np.sqrt(np.mean((y_positions - np.mean(y_positions))**2))

    # Cost of Transport
    m_total = np.sum(model.body_mass)
    g_val = 9.81
    CoT = total_energy / (m_total * g_val * distance_traveled) if distance_traveled > 0 else 1e6

    # Stability: extract roll and pitch from orientation matrices.
    # For simplicity, we compute dummy variances.
    # (A proper implementation would convert rotation matrices to Euler angles.)
    # Here we assume orientation fluctuations are proportional to variance in the first two elements.
    # orientations = np.array(orientations)
    # S = 0.0
    # if orientations.ndim == 3 and orientations.shape[0] > 0:
    #     roll_variance = np.var(orientations[:, 0, 0])
    #     pitch_variance = np.var(orientations[:, 1, 1])
    #     S = -(roll_variance + pitch_variance)
    # else:
    #     print("Warning: Orientation data is not in the expected format. Stability (S) set to 0.")

    # Return objectives: We aim to maximize v_x and S, minimize d_y and CoT.
    # For BO, we convert minimization objectives to negative values.
    objectives = np.array([v_x, -d_y, -CoT])
    return objectives

# =============================================================================
#         BO-Torch Multi-Objective Optimization Setup
# =============================================================================

# Define bounds for parameters: [omega_stance, omega_swing, a_exp, lambda, A_scale]
bounds = torch.tensor([
    [1.0, 1.0, 5.0, 0.1, 0.5],
    [3.0, 3.0, 15.0, 1.0, 5.0]
], dtype=torch.double)

# Wrap simulation evaluation for BO
def evaluate_objectives(params_tensor):
    # params_tensor: shape (q, d) or (d,)
    if params_tensor.ndim == 1:
        params_tensor = params_tensor.unsqueeze(0)
    results = []
    for p in params_tensor:
        p_np = p.detach().cpu().numpy()
        # Run simulation (this is computationally expensive)
        obj = simulate_robot(p_np, headless=True)
        results.append(obj)
    return torch.tensor(results, dtype=torch.double)

# Initial design: Latin Hypercube sampling (here, random uniform sampling)
num_initial = 5
dim = bounds.shape[1]
X_init = torch.rand(num_initial, dim, dtype=torch.double) * (bounds[1] - bounds[0]) + bounds[0]
Y_init = evaluate_objectives(X_init)

# Build GP models for each objective (4 objectives)
models = []
for i in range(Y_init.shape[1]):
    model = SingleTaskGP(X_init, Y_init[:, i:i+1])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    botorch.fit.fit_gpytorch_mll(mll)  # or use a custom training loop
    models.append(model)

model_list = ModelListGP(*models)

# Define a reference point for hypervolume (choose slightly worse than observed)
Y_np = Y_init.detach().numpy()
ref_point = torch.tensor([np.min(Y_np[:, 0]) - 0.1,
                          np.max(Y_np[:, 1]) + 0.1,
                          np.max(Y_np[:, 2]) + 0.1,], dtype=torch.double)


partitioning = NondominatedPartitioning(num_outputs=Y_init.shape[1], Y=Y_init)

# Define the acquisition function: qEHVI
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
acq_func = qExpectedHypervolumeImprovement(
    model=model_list,
    ref_point=ref_point,
    partitioning=partitioning,
)

# =============================================================================
#                        BO Optimization Loop
# =============================================================================
num_iterations = 5  # Increase for real runs
q = 1  # One candidate per iteration

for iteration in range(num_iterations):
    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,
        num_restarts=5,
        raw_samples=20,
    )
    candidate = candidate.detach()
    Y_new = evaluate_objectives(candidate)
    # Append new data
    X_init = torch.cat([X_init, candidate], dim=0)
    Y_init = torch.cat([Y_init, Y_new], dim=0)
    
    # Refit GP models
    models = []
    for i in range(Y_init.shape[1]):
        model = SingleTaskGP(X_init, Y_init[:, i:i+1])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        botorch.fit.fit_gpytorch_mll(mll)
        models.append(model)
    model_list = ModelListGP(*models)
    
    partitioning = NondominatedPartitioning(num_outputs=Y_init.shape[1], Y=Y_init)
    acq_func = qExpectedHypervolumeImprovement(
        model=model_list,
        ref_point=ref_point,
        partitioning=partitioning,
    )
    
    print(f"Iteration {iteration+1}: Candidate = {candidate.numpy()}, Objectives = {Y_new.numpy()}")

# Plot Pareto front (projected onto forward velocity and CoT, for example)
Y_final = Y_init.detach().numpy()
plt.figure(figsize=(8,6))
# For plotting, we choose v_x (objective 0) vs. CoT (objective 2; note: we maximized -CoT)
plt.scatter(Y_final[:, 0], -Y_final[:, 2], c='b', marker='o')
plt.xlabel("Forward Velocity (m/s)")
plt.ylabel("Cost of Transport")
plt.title("Pareto Front Projection: Forward Velocity vs. CoT")
plt.grid(True)
plt.show()

# =============================================================================
#             Final Simulation with Selected Pareto Candidate
# =============================================================================
# For demonstration, select candidate with highest forward velocity
best_idx = np.argmax(Y_final[:, 0])
best_params = X_init[best_idx].detach().cpu().numpy()
print("Selected Pareto candidate parameters:", best_params)
print("Objective values for selected candidate:", Y_final[best_idx])

# Run final simulation with viewer to render robot operation
def simulate_robot_with_viewer(params):
    # Similar to simulate_robot, but with viewer for rendering.
    omega_stance, omega_swing, a_exp, coupling_param, amplitude_scale = params
    model_path = (
        "c:/Users/chike/Box/TurtleRobotExperiments/"
        "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
        "assets/turtlev1/testrobot1.xml"
    )
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    actuator_names = [
        "pos_frontleftflipper",
        "pos_frontrightflipper",
        "pos_backleft",
        "pos_backright",
        "pos_frontlefthip",
        "pos_frontrighthip"
    ]
    phase_offsets = {
        "pos_frontleftflipper":  0.0,
        "pos_frontrightflipper": 0.5,
        "pos_backleft":          0.5,
        "pos_backright":         0.0,
        "pos_frontlefthip":      0.75,
        "pos_frontrighthip":     0.25
    }
    joint_limits = {}
    for i, name in enumerate(actuator_names):
        ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
        joint_limits[name] = (ctrl_min, ctrl_max)
    mu_val = 0.04
    alpha_val = 10.0
    oscillators = {}
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name] = {
            "x": np.sqrt(mu_val) * np.cos(phase0) + 0.002 * np.random.randn(),
            "y": np.sqrt(mu_val) * np.sin(phase0) + 0.002 * np.random.randn()
        }
    dt_cpg = 0.001
    sim_duration = 30.0
    start_time_sim = time.time()
    sim_time = 0.0
    while sim_time < sim_duration:
        steps = int(np.floor(0.005 / dt_cpg))
        for _ in range(steps):
            x_all = [oscillators[name]["x"] for name in actuator_names]
            y_all = [oscillators[name]["y"] for name in actuator_names]
            for i, name in enumerate(actuator_names):
                y_val = oscillators[name]["y"]
                freq = (omega_stance / (1.0 + np.exp(-a_exp * y_val))) + \
                       (omega_swing  / (1.0 + np.exp(a_exp * y_val)))
                x_new, y_new = hopf_step(oscillators[name]["x"], oscillators[name]["y"],
                                          alpha_val, mu_val, freq, dt_cpg,
                                          coupling_param, x_all, y_all, i, phase_offsets, actuator_names)
                oscillators[name]["x"] = x_new
                oscillators[name]["y"] = y_new
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                offset = (min_angle + max_angle) / 2.0
                gain = (max_angle - min_angle) / 2.0
                desired_angle = offset + gain * np.tanh(oscillators[name]["x"]) * amplitude_scale
                desired_angle = np.clip(desired_angle, min_angle, max_angle)
                actuator_id = actuator_names.index(name)
                data.ctrl[actuator_id] = desired_angle
            mujoco.mj_step(model, data)
            sim_time += dt_cpg
        # Render the simulation using the viewer
        mujoco.viewer.launch_passive(model, data)
    return

# Launch viewer simulation for final candidate (this may open a window)
simulate_robot_with_viewer(best_params)
