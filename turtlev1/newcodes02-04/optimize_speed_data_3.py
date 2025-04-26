import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import torch
import pandas as pd
from pathlib import Path

from botorch.utils.transforms import unnormalize, normalize
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.optim import optimize_acqf

# =============================================================================
# 1. Create output folder
# =============================================================================

plot_path = Path(
    "C:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/"
    "NnamdiFiles/mujocotest1/assets/Gait-Optimization/data/figs/optplots_set_1"
)

out_dir = plot_path / "bo_results_set_1"
out_dir.mkdir(parents=True, exist_ok=True)



# =============================================================================
# 2. MuJoCo & Simulation Setup
# =============================================================================
model_path = (
    "c:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/"
    "assets/Gait-Optimization/turtlev1/xmls/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

actuator_names = [
    "pos_frontleftflipper",
    "pos_frontrightflipper",
    "pos_backleft",
    "pos_backright",
    "pos_frontlefthip",
    "pos_frontrighthip"
]
def get_actuator_index(model, name):
    for i in range(model.nu):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == name:
            return i
    raise ValueError(f"Actuator '{name}' not found.")
actuator_indices = {n: get_actuator_index(model, n) for n in actuator_names}

joint_limits = {}
for i, name in enumerate(actuator_names):
    ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
    joint_limits[name] = (ctrl_min, ctrl_max)

main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

phase_offsets_s = {
    "pos_frontleftflipper": 0.0,
    "pos_frontrightflipper": 0.0,
    "pos_backleft": 0.0,
    "pos_backright": 0.0,
    "pos_frontlefthip": 0.75,
    "pos_frontrighthip": 0.75
}
phase_offsets = phase_offsets_s

alpha, mu = 20.0, 1.0

def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
    r_sq = x*x + y*y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    delta = 0.0
    phase_list = [phase_offsets[n] for n in actuator_names]
    for j in range(len(x_all)):
        if j == index:
            continue
        theta_ji = 2.0 * np.pi * (phase_list[index] - phase_list[j])
        delta += y_all[j]*np.cos(theta_ji) - x_all[j]*np.sin(theta_ji)
    dy += coupling * delta

    x_new = x + dx*dt
    y_new = y + dy*dt
    return x_new, y_new

joint_output_map = {n:{"offset":0, "gain":1} for n in actuator_names}

# =============================================================================
# 3. Offline data import
# =============================================================================
csv_path = (
    "C:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/"
    "NnamdiFiles/mujocotest1/assets/Gait-Optimization/data/"
    "lhs_simulation_results_sync_set_1.csv"
)
offline_df          = pd.read_csv(csv_path)
param_cols          = ['stance_freq','swing_freq','a_param','lambda_cpl','A_front','A_back','A_hip']
objective_col       = 'Average_Forward_Speed'
offline_params_orig = offline_df[param_cols].values
offline_objectives  = offline_df[[objective_col]].values

# =============================================================================
# 4. Simulation & objective function (updated)
# =============================================================================
def run_simulation(params, sim_duration=30.0, seed=0, warmup_duration=4.0):
    np.random.seed(seed)
    mujoco.mj_resetData(model, data)
    data.time = 0.0

    stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip = params

    oscillators = {}
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name] = {
            "x": np.sqrt(mu)*np.cos(phase0) + 0.002*np.random.randn(),
            "y": np.sqrt(mu)*np.sin(phase0) + 0.002*np.random.randn()
        }
    amplitude_factors = {
        "pos_frontleftflipper":  A_front,
        "pos_frontrightflipper": A_front,
        "pos_backleft":          A_back,
        "pos_backright":         A_back,
        "pos_frontlefthip":      A_hip,
        "pos_frontrighthip":     A_hip
    }

    dt_cpg = 0.001
    
    # Warmup simulation without rendering
    sim_steps = int(warmup_duration / model.opt.timestep)
    for _ in range(sim_steps):
        x_all = [oscillators[n]["x"] for n in actuator_names]
        y_all = [oscillators[n]["y"] for n in actuator_names]
        for i,n in enumerate(actuator_names):
            x_i, y_i = oscillators[n]["x"], oscillators[n]["y"]
            freq = (stance_freq/(1+np.exp(-a_param*y_i))) + \
                   (swing_freq/(1+np.exp(a_param*y_i)))
            x_new,y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                    lambda_cpl, x_all, y_all, i, phase_offsets)
            oscillators[n]["x"],oscillators[n]["y"] = x_new, y_new
        mujoco.mj_step(model, data)

    # Store initial position
    com_start = data.xpos[main_body_id].copy()
    
    # Main simulation without rendering
    sim_steps = int(sim_duration / model.opt.timestep)
    for _ in range(sim_steps):
        x_all = [oscillators[n]["x"] for n in actuator_names]
        y_all = [oscillators[n]["y"] for n in actuator_names]
        for i,n in enumerate(actuator_names):
            x_i, y_i = oscillators[n]["x"], oscillators[n]["y"]
            freq = (stance_freq/(1+np.exp(-a_param*y_i))) + \
                   (swing_freq/(1+np.exp(a_param*y_i)))
            x_new,y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                    lambda_cpl, x_all, y_all, i, phase_offsets)
            oscillators[n]["x"],oscillators[n]["y"] = x_new, y_new

            # Apply control
            mn,mx = joint_limits[n]
            off = joint_output_map[n]["offset"]
            gain = joint_output_map[n]["gain"]
            amp = amplitude_factors[n]
            angle = off + gain*amp*np.tanh(oscillators[n]["x"])
            data.ctrl[actuator_indices[n]] = np.clip(angle, mn, mx)
        mujoco.mj_step(model, data)

    # Calculate displacement
    com_end = data.xpos[main_body_id].copy()
    x_disp = com_end[0] - com_start[0]
    return x_disp/sim_duration

def objective_function(params, orig_bounds):
    pts = unnormalize(params, orig_bounds).detach().numpy()
    out=[]
    for row in pts:
        out.append([-run_simulation(row)])
    return torch.tensor(out, dtype=torch.double)

# =============================================================================
# 5. Prepare data & bounds
# =============================================================================
orig_bounds         = torch.tensor([[0.,0.,5.,0.,0.,0.,0.],[6.,6.,20.,1.,2.,2.,2.]], dtype=torch.double)
norm_bounds         = torch.stack([torch.zeros(7), torch.ones(7)])
offline_params      = torch.tensor(offline_params_orig, dtype=torch.double)
offline_params_norm = normalize(offline_params, orig_bounds)
offline_obj         = -torch.tensor(offline_objectives, dtype=torch.double)

# =============================================================================
# 6. BO loop with diagnostics
# =============================================================================
def run_bayesian_opt(acq_name, num_iterations=100):
    train_x    = offline_params_norm.clone()
    train_y    = offline_obj.clone()
    train_yvar = torch.full_like(train_y, 1e-6)

    best_speed_hist      = []
    best_param_hist      = []
    candidate_param_hist = []
    actual_speed_hist    = []
    diagnostics          = []

    best_speed = (-train_y).max().item()
    best_speed_hist.append(best_speed)
    idx0 = torch.argmax(-train_y)
    p0   = unnormalize(train_x[idx0].unsqueeze(0), orig_bounds).numpy()[0]
    best_param_hist.append(p0)
    candidate_param_hist.append(p0)
    actual_speed_hist.append(best_speed)

    print(f"\n--- {acq_name} Optimization ---")
    for it in range(1, num_iterations+1):
        gp = SingleTaskGP(train_x, train_y, train_Yvar=train_yvar,
                          outcome_transform=Standardize(m=1)).double()
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # 1) ensure train mode for MLL computation
        gp.train()
        gp.likelihood.train()
        posterior_train = gp(train_x)
        mll_val = mll(posterior_train, train_y.squeeze(-1)).item()

        # 2) switch to eval mode for acquisition/posterior on new points
        gp.eval()
        gp.likelihood.eval()
        with torch.no_grad():
            probe = torch.rand(2048, 7, dtype=torch.double).unsqueeze(1)
            ei    = LogExpectedImprovement(gp, best_f=train_y.min(), maximize=False)(probe)
            max_ei    = ei.max().item()
            med_sigma = gp.posterior(probe).variance.sqrt().median().item()

        diagnostics.append((it, max_ei, med_sigma, mll_val))

        if acq_name=="PI":
            acq = ProbabilityOfImprovement(gp, best_f=train_y.min(), maximize=False)
        elif acq_name=="LogEI":
            acq = LogExpectedImprovement(gp, best_f=train_y.min(), maximize=False)
        elif acq_name=="UCB":
            acq = UpperConfidenceBound(gp, beta=2.0, maximize=False)
        else:
            raise ValueError(acq_name)

        candidate,_ = optimize_acqf(
            acq_function=acq,
            bounds=norm_bounds,
            q=1,
            num_restarts=15,
            raw_samples=512,
        )
        cand_params = unnormalize(candidate, orig_bounds).detach().numpy()[0]
        candidate_param_hist.append(cand_params)

        new_y = objective_function(candidate, orig_bounds)
        new_speed = -new_y.item()
        actual_speed_hist.append(new_speed)

        train_x    = torch.cat([train_x, candidate], dim=0)
        train_y    = torch.cat([train_y, new_y], dim=0)
        train_yvar = torch.cat([train_yvar, torch.full((1,1),1e-6)], dim=0)

        if new_speed > best_speed:
            best_speed = new_speed
        best_speed_hist.append(best_speed)
        best_param_hist.append(cand_params)

        print(f"[{acq_name} Iter {it:02d}] cand={new_speed:.4f} best={best_speed:.4f} EI={max_ei:.2e}")

    # save
    pd.DataFrame(best_speed_hist, columns=["best_speed"])\
      .to_csv(out_dir/f"{acq_name}_best_speed.csv", index=True)
    pd.DataFrame(best_param_hist, columns=param_cols)\
      .to_csv(out_dir/f"{acq_name}_best_params.csv", index=True)
    pd.DataFrame(candidate_param_hist, columns=param_cols)\
      .to_csv(out_dir/f"{acq_name}_candidate_params.csv", index=False)
    pd.DataFrame(actual_speed_hist, columns=["candidate_speed"])\
      .to_csv(out_dir/f"{acq_name}_actual_speed.csv", index=False)
    pd.DataFrame(diagnostics, columns=["iter","max_ei","median_sigma","mll"])\
      .to_csv(out_dir/f"{acq_name}_diagnostics.csv", index=False)

    plt.figure(figsize=(5,4))
    plt.plot(best_speed_hist, marker="o")
    plt.title(f"{acq_name}: Best Speed vs Iter")
    plt.xlabel("Iteration"); plt.ylabel("Speed (m/s)"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir/f"{acq_name}_best_speed.png", dpi=150)
    plt.close()

    return {
        "best_speed_history":       np.array(best_speed_hist),
        "best_param_history":       np.array(best_param_hist),
        "candidate_param_history":  np.array(candidate_param_hist),
        "actual_speed_history":     np.array(actual_speed_hist),
        "diagnostics":              diagnostics,
    }

# =============================================================================
# 7. Run & compare acquisition functions
# =============================================================================
acq_functions = ["PI","LogEI","UCB"]
results = {}
for acqf in acq_functions:
    results[acqf] = run_bayesian_opt(acqf, num_iterations=100)

# =============================================================================
# 8. Plotting evolution of parameters & candidate speeds
# =============================================================================
param_names = param_cols
for acq in acq_functions:
    res = results[acq]
    param_hist = res["candidate_param_history"]
    actual_speed_hist = res["actual_speed_history"]
    n_iter = param_hist.shape[0]
    iters = np.arange(n_iter)
    dim = param_hist.shape[1]
    nrows = (dim+1)//2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 3*nrows))
    axes = axes.flatten()
    for d in range(dim):
        ax = axes[d]
        ax.plot(iters, param_hist[:,d], marker='o', color='tab:blue', label=param_names[d])
        ax.set_xlabel("Iteration")
        ax.set_ylabel(param_names[d], color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.set_title(f"{acq}: {param_names[d]} vs Iter")
        ax.grid(True)
        final_val = param_hist[-1,d]
        ax.annotate(f"{final_val:.2f}", xy=(iters[-1], final_val),
                    xytext=(5,0), textcoords="offset points", color='tab:blue')
        ax2 = ax.twinx()
        ax2.plot(iters, actual_speed_hist, marker='x', linestyle='--', color='tab:red')
        ax2.set_ylabel("Candidate Speed", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.suptitle(f"{acq} Candidate Parameter Evolution")
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.savefig(plot_path/f"{acq}_candidate_parameters_vs_iteration.png", format='png')
    fig.savefig(plot_path/f"{acq}_candidate_parameters_vs_iteration.svg", format='svg')
    plt.show()

# =============================================================================
# 9. Comparative speed plots
# =============================================================================
fig, axs = plt.subplots(1,2,figsize=(12,4))
for acq in acq_functions:
    axs[0].plot(results[acq]["best_speed_history"], marker='o', label=acq)
axs[0].set_title("Best Speed vs Iteration")
axs[0].set_xlabel("Iteration"); axs[0].set_ylabel("Best Speed (m/s)")
axs[0].grid(True); axs[0].legend()
for acq in acq_functions:
    axs[1].plot(results[acq]["actual_speed_history"], marker='o', label=acq)
axs[1].set_title("Candidate Speed vs Iteration")
axs[1].set_xlabel("Iteration"); axs[1].set_ylabel("Candidate Speed (m/s)")
axs[1].grid(True); axs[1].legend()
fig.suptitle("Speed History Comparisons")
fig.tight_layout(rect=[0,0.03,1,0.95])
fig.savefig(plot_path/"Speed_History_Comparisons.png", format='png')
fig.savefig(plot_path/"Speed_History_Comparisons.svg", format='svg')
plt.show()

# =============================================================================
# 10. GP Diagnostics Plot
# =============================================================================
plt.figure(figsize=(6,4))
for acq in acq_functions:
    diag = np.array(results[acq]["diagnostics"])
    iters = diag[:,0]
    mlls  = diag[:,3]
    plt.plot(iters, mlls, marker='o', label=acq)
plt.title("GP Marginal Log-Likelihood vs Iteration")
plt.xlabel("Iteration"); plt.ylabel("Marginal Log-Likelihood")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(plot_path/"GP_MLL_Comparison.png", dpi=150)
plt.show()

# =============================================================================
# 11. Final simulation with logging (unchanged)
# =============================================================================
chosen_acq = "LogEI"
final_params = results[chosen_acq]["best_param_history"][-1]
print(f"\nBest parameters found by {chosen_acq}: {final_params}")
print(f"Final best speed: {results[chosen_acq]['best_speed_history'][-1]:.4f} m/s")


def run_simulation_with_logging(params, sim_duration=30.0, seed=42, warmup_duration=4.0):
    """
    Runs the simulation with the given parameter vector:
      params = [stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip]
    Logs time-series data and returns a dictionary of metrics.
    Also logs instantaneous frequency for each joint and prints the average frequency per joint
    and each joint's output range.
    """
    np.random.seed(seed)
    
    # Add CPG state history tracking
    cpg_history_x = {name: [] for name in actuator_names}
    np.random.seed(seed)
    mujoco.mj_resetData(model, data)
    data.time = 0.0

    stance_freq, swing_freq, a_param, lambda_cpl, A_front, A_back, A_hip = params

    # Initialize oscillators for each actuator.
    oscillators = {}
    for name in actuator_names:
        phase0 = phase_offsets[name] * 2.0 * np.pi
        oscillators[name] = {
            "x": np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn(),
            "y": np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()
        }

    amplitude_factors = {
        "pos_frontleftflipper":  A_front,
        "pos_frontrightflipper": A_front,
        "pos_backleft":          A_back,
        "pos_backright":         A_back,
        "pos_frontlefthip":      A_hip,
        "pos_frontrighthip":     A_hip
    }

    dt_cpg = 0.001
    time_data = []
    ctrl_data = {name: [] for name in actuator_names}
    power_consumption = []
    com_positions = []
    actuator_torque_history = {name: [] for name in actuator_names}
    joint_velocity_history  = {name: [] for name in actuator_names}

    # Dictionary to log instantaneous frequency for each joint.
    freq_history = {name: [] for name in actuator_names}

    ######################
    # Warm-up Phase
    ######################
    start_time_overall = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        warmup_start = time.time()
        last_loop_time = warmup_start
        print(f"[INFO] Starting warm-up for {warmup_duration:.1f}s...")
        while viewer.is_running():
            now = time.time()
            sim_time = now - warmup_start
            loop_dt = now - last_loop_time
            last_loop_time = now
            if sim_time >= warmup_duration:
                print("[INFO] Warm-up complete. Starting main logging phase.")
                break
            steps = int(np.floor(loop_dt / dt_cpg))
            for _ in range(steps):
                x_all = [oscillators[name]["x"] for name in actuator_names]
                y_all = [oscillators[name]["y"] for name in actuator_names]
                for i, name in enumerate(actuator_names):
                    x_i = oscillators[name]["x"]
                    y_i = oscillators[name]["y"]
                    freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                           (swing_freq  / (1.0 + np.exp(a_param * y_i)))
                    x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                            lambda_cpl, x_all, y_all, i, phase_offsets)
                    oscillators[name]["x"] = x_new
                    oscillators[name]["y"] = y_new
            mujoco.mj_step(model, data)
            viewer.sync()

        ######################
        # Main Simulation Phase
        ######################
        start_time = time.time()
        last_loop_time = start_time
        while viewer.is_running():
            now = time.time()
            sim_time = now - start_time
            if sim_time >= sim_duration:
                print(f"[INFO] Reached {sim_duration:.1f}s of simulation. Stopping.")
                break

            loop_dt = now - last_loop_time
            last_loop_time = now
            steps = int(np.floor(loop_dt / dt_cpg))
            for _ in range(steps):
                x_all = [oscillators[name]["x"] for name in actuator_names]
                y_all = [oscillators[name]["y"] for name in actuator_names]
                for i, name in enumerate(actuator_names):
                    x_i = oscillators[name]["x"]
                    y_i = oscillators[name]["y"]
                    freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                           (swing_freq  / (1.0 + np.exp(a_param * y_i)))
                    freq_history[name].append(freq)
                    x_new, y_new = hopf_step(x_i, y_i, alpha, mu, freq, dt_cpg,
                                            lambda_cpl, x_all, y_all, i, phase_offsets)
                    oscillators[name]["x"] = x_new
                    oscillators[name]["y"] = y_new
                    cpg_history_x[name].append(x_new)


            # Map oscillator outputs to joint controls
            for name in actuator_names:
                min_angle, max_angle = joint_limits[name]
                off = joint_output_map[name]["offset"]
                gain = joint_output_map[name]["gain"]
                amp_factor = amplitude_factors[name]
                desired_angle = off + gain * amp_factor * np.tanh(oscillators[name]["x"])
                desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
                data.ctrl[actuator_indices[name]] = desired_angle_clamped

            mujoco.mj_step(model, data)
            viewer.sync()

            time_data.append(sim_time)
            for name in actuator_names:
                ctrl_data[name].append(data.ctrl[actuator_indices[name]])
            com_positions.append(data.xpos[main_body_id].copy())

            qvel = data.qvel[:model.nu]
            torque = data.actuator_force[:model.nu]
            instant_power = np.sum(np.abs(torque) * np.abs(qvel))
            power_consumption.append(instant_power)

            for i, name in enumerate(actuator_names):
                actuator_torque_history[name].append(data.actuator_force[actuator_indices[name]])
                joint_velocity_history[name].append(qvel[actuator_indices[name]])

    # After the simulation, compute average frequency and control output range per joint.
    avg_freq = {}
    output_range = {}
    for name in actuator_names:
        freq_arr = np.array(freq_history[name])
        avg_freq[name] = np.mean(freq_arr) if freq_arr.size > 0 else 0.0

    return {
        "time_data": time_data,
        "ctrl_data": ctrl_data,
        "power_consumption": power_consumption,
        "com_positions": com_positions,
        "actuator_torque_history": actuator_torque_history,
        "joint_velocity_history": joint_velocity_history,
        "frequency_history": freq_history,
        "avg_frequency": avg_freq,
        "output_range": output_range,
        "cpg_history_x": cpg_history_x
    }

# Run final simulation with logging using the best parameters from LogEI.
log_data = run_simulation_with_logging(final_params, sim_duration=30.0)

time_data = log_data["time_data"]
ctrl_data = log_data["ctrl_data"]
power_consumption = log_data["power_consumption"]
com_positions = np.array(log_data["com_positions"])
actuator_torque_history = log_data["actuator_torque_history"]
joint_velocity_history = log_data["joint_velocity_history"]
cpg_history_x = log_data["cpg_history_x"]

# =============================================================================
#               PLOTTING TIME-SERIES DATA FROM FINAL SIMULATION
# =============================================================================
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# (0,0): Actuator control signals.
for name in actuator_names:
    axs[0, 0].plot(time_data, ctrl_data[name], label=name)
axs[0, 0].set_title("Actuator Control Signals")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Joint Angle (rad)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# (0,1): COM position over time.
if com_positions.size > 0:
    axs[0, 1].plot(time_data, com_positions[:, 0], label="COM X")
    axs[0, 1].plot(time_data, com_positions[:, 1], label="COM Y")
    axs[0, 1].plot(time_data, com_positions[:, 2], label="COM Z")
axs[0, 1].set_title("COM Position vs Time")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Position (m)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# (1,0): Instantaneous power consumption.
axs[1, 0].plot(time_data, power_consumption, label="Instant Power")
axs[1, 0].set_title("Instantaneous Power Consumption")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Power (W)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# (1,1): Trajectory (X vs. Y).
if com_positions.size > 0:
    axs[1, 1].plot(com_positions[:, 0], com_positions[:, 1], 'b-', label="Trajectory")
axs[1, 1].set_title("Trajectory (X vs Y)")
axs[1, 1].set_xlabel("X (m)")
axs[1, 1].set_ylabel("Y (m)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# (2,0): Actuator torque history.
for name in actuator_names:
    axs[2, 0].plot(time_data, actuator_torque_history[name], label=name)
axs[2, 0].set_title("Actuator Torque Over Time")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Torque (Nm)")
axs[2, 0].legend()
axs[2, 0].grid(True)

# (2,1): Joint velocity history.
for name in actuator_names:
    axs[2, 1].plot(time_data, joint_velocity_history[name], label=name)
axs[2, 1].set_title("Joint Velocity Over Time")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Velocity (rad/s)")
axs[2, 1].legend()
axs[2, 1].grid(True)

# (0,1): CPG Oscillator Outputs Over Time
for name in actuator_names:
    axs[2, 1].plot(range(len(cpg_history_x[name])), cpg_history_x[name], label=name)
axs[2, 1].set_title("CPG Oscillator Outputs Over Time")
axs[2, 1].set_xlabel("Time Step")
axs[2, 1].set_ylabel("Oscillator x Output")
axs[2, 1].legend()
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()
