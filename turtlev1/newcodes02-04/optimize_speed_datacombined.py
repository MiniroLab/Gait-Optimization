import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import torch
import pandas as pd
from pathlib import Path

from botorch.utils.transforms import normalize, unnormalize
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import (
    ProbabilityOfImprovement,
    LogExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.optim import optimize_acqf

# =============================================================================
# 1. Shared MuJoCo & GP setup
# =============================================================================
# Load MuJoCo model once
model_path = Path(
    "C:/Users/chike/Box/TurtleRobotExperiments/"
    "Sea_Turtle_Robot_AI_Powered_Simulations_Project/"
    "NnamdiFiles/mujocotest1/"
    "assets/Gait-Optimization/turtlev1/xmls/testrobot1.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))
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
    raise ValueError(name)
actuator_indices = {n: get_actuator_index(model, n) for n in actuator_names}
joint_limits = {n: tuple(model.actuator_ctrlrange[i]) 
                for i, n in enumerate(actuator_names)}
main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

phase_offsets = {
    "pos_frontleftflipper": 0.0,
    "pos_frontrightflipper": 0.0,
    "pos_backleft": 0.0,
    "pos_backright": 0.0,
    "pos_frontlefthip": 0.75,
    "pos_frontrighthip": 0.75
}

alpha, mu = 20.0, 1.0
dt_cpg = 0.001

def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, idx):
    r_sq = x*x + y*y
    dx = alpha*(mu - r_sq)*x - omega*y
    dy = alpha*(mu - r_sq)*y + omega*x
    phs = [phase_offsets[n] for n in actuator_names]
    for j, (xj, yj) in enumerate(zip(x_all, y_all)):
        if j == idx: continue
        θ = 2*np.pi*(phs[idx] - phs[j])
        dy += coupling*(yj*np.cos(θ) - xj*np.sin(θ))
    return x + dx*dt, y + dy*dt

def run_simulation(params, sim_duration=30.0, seed=45, warmup_duration=4.0):
    np.random.seed(seed)
    mujoco.mj_resetData(model, data)
    data.time = 0.0
    stance, swing, a_param, coupling, A_front, A_back, A_hip = params
    amps = dict(zip(actuator_names, [A_front, A_front, A_back, A_back, A_hip, A_hip]))
    oscillators = {
        n: {
            "x": np.sqrt(mu)*np.cos(2*np.pi*phase_offsets[n]) + 0.002*np.random.randn(),
            "y": np.sqrt(mu)*np.sin(2*np.pi*phase_offsets[n]) + 0.002*np.random.randn()
        }
        for n in actuator_names
    }
    # Warm‑up
    print(f"[INFO] Warm‑up for {warmup_duration}s")
    warmup_steps = int(warmup_duration / model.opt.timestep)
    for _ in range(warmup_steps):
        xs = [oscillators[n]["x"] for n in actuator_names]
        ys = [oscillators[n]["y"] for n in actuator_names]
        for i, n in enumerate(actuator_names):
            x, y = oscillators[n]["x"], oscillators[n]["y"]
            freq = stance/(1+np.exp(-a_param*y)) + swing/(1+np.exp(a_param*y))
            x_new, y_new = hopf_step(x, y, alpha, mu, freq, dt_cpg, coupling, xs, ys, i)
            oscillators[n]["x"], oscillators[n]["y"] = x_new, y_new
        mujoco.mj_step(model, data)
    print("[INFO] Warm‑up complete")

    com_start = data.xpos[main_body_id].copy()
    # Main sim
    main_steps = int(sim_duration / model.opt.timestep)
    for _ in range(main_steps):
        xs = [oscillators[n]["x"] for n in actuator_names]
        ys = [oscillators[n]["y"] for n in actuator_names]
        for i, n in enumerate(actuator_names):
            x, y = oscillators[n]["x"], oscillators[n]["y"]
            freq = stance/(1+np.exp(-a_param*y)) + swing/(1+np.exp(a_param*y))
            x_new, y_new = hopf_step(x, y, alpha, mu, freq, dt_cpg, coupling, xs, ys, i)
            oscillators[n]["x"], oscillators[n]["y"] = x_new, y_new
            mn, mx = joint_limits[n]
            angle = amps[n] * np.tanh(x_new)
            data.ctrl[actuator_indices[n]] = np.clip(angle, mn, mx)
        mujoco.mj_step(model, data)
    com_end = data.xpos[main_body_id].copy()
    speed = (com_end[0] - com_start[0]) / sim_duration
    return speed

def objective_function(params, orig_bounds):
    pts = unnormalize(params, orig_bounds).detach().numpy()
    out = []
    for row in pts:
        out.append([-run_simulation(row)])
    return torch.tensor(out, dtype=torch.double)

# =============================================================================
# 2. Bounds & acquisition setup
# =============================================================================
orig_bounds = torch.tensor([
    [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    [6.0, 6.0,20.0, 1.0, 2.0, 2.0, 2.0],
], dtype=torch.double)
norm_bounds = torch.stack([torch.zeros(7), torch.ones(7)])

acq_functions = ["PI","LogEI","UCB"]

# =============================================================================
# 3. run_bayesian_opt (same as before, writes into given out_dir)
# =============================================================================
def run_bayesian_opt(acq_name, train_x, train_y, train_yvar, orig_bounds, norm_bounds,
                     num_iterations, out_dir):
    # identical to prior run_bayesian_opt, but removes CSV/plot prefix dynamic
    best_speed_hist = []; best_param_hist = []
    cand_param_hist = []; actual_speed_hist = []; diagnostics = []

    # initialize from train_y
    best_speed = (-train_y).max().item()
    best_speed_hist.append(best_speed)
    idx0 = torch.argmax(-train_y)
    p0   = unnormalize(train_x[idx0].unsqueeze(0), orig_bounds).numpy()[0]
    best_param_hist.append(p0); cand_param_hist.append(p0); actual_speed_hist.append(best_speed)

    for it in range(1, num_iterations+1):
        gp = SingleTaskGP(train_x, train_y, train_Yvar=train_yvar,
                          outcome_transform=Standardize(m=1)).double()
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # diagnostics
        gp.train(); gp.likelihood.train()
        post_train = gp(train_x)
        mll_val = mll(post_train, train_y.squeeze(-1)).item()
        gp.eval(); gp.likelihood.eval()
        with torch.no_grad():
            probe = torch.rand(4096, 7, dtype=torch.double).unsqueeze(1)  # Increased probe points from 2048 to 4096 for better coverage
            ei    = LogExpectedImprovement(gp, best_f=train_y.min(), maximize=False)(probe)
            max_ei = ei.max().item()
            med_sig = gp.posterior(probe).variance.sqrt().median().item()
        diagnostics.append((it, max_ei, med_sig, mll_val))

        # choose acquisition
        if acq_name=="PI":
            acq = ProbabilityOfImprovement(gp, best_f=train_y.min(), maximize=False)
        elif acq_name=="LogEI":
            acq = LogExpectedImprovement(gp, best_f=train_y.min(), maximize=False)
        elif acq_name=="UCB":
            acq = UpperConfidenceBound(gp, beta=3.0, maximize=False)  # Increased from 2.0
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=norm_bounds,
            q=1,
            num_restarts=30,  # Increased from 15
            raw_samples=1024, # Increased from 512
        )
        cp = unnormalize(candidate, orig_bounds).detach().numpy()[0]
        cand_param_hist.append(cp)

        new_y = objective_function(candidate, orig_bounds)
        new_speed = -new_y.item()
        actual_speed_hist.append(new_speed)

        train_x    = torch.cat([train_x, candidate], dim=0)
        train_y    = torch.cat([train_y, new_y], dim=0)
        train_yvar = torch.cat([train_yvar, torch.full((1,1),1e-6)], dim=0)

        if new_speed > best_speed:
            best_speed = new_speed
        best_speed_hist.append(best_speed)
        best_param_hist.append(cp)

        print(f"[{acq_name} Iter {it:02d}] cand={new_speed:.4f} best={best_speed:.4f} EI={max_ei:.2e}")

    # save CSVs
    pd.DataFrame(best_speed_hist,      columns=["best_speed"]).to_csv(out_dir/f"{acq_name}_best_speed.csv",      index=True)
    pd.DataFrame(best_param_hist,      columns=param_cols).to_csv(out_dir/f"{acq_name}_best_params.csv",      index=True)
    pd.DataFrame(cand_param_hist,      columns=param_cols).to_csv(out_dir/f"{acq_name}_candidate_params.csv", index=False)
    pd.DataFrame(actual_speed_hist,    columns=["candidate_speed"]).to_csv(out_dir/f"{acq_name}_actual_speed.csv", index=False)
    pd.DataFrame(diagnostics, columns=["iter","max_ei","median_sigma","mll"]).to_csv(out_dir/f"{acq_name}_diagnostics.csv", index=False)

    # Convert list to numpy array for column access
    cand_param_array = np.array(cand_param_hist)
    
    # Create arrays of equal length
    n_iters = len(cand_param_hist)  # Use candidate params as reference length

    # Pad diagnostics metrics with NaN for the last entry
    diag_metrics = np.array(diagnostics)
    if len(diag_metrics) < n_iters:
        # Create a row of NaNs
        nan_row = np.array([[np.nan, np.nan, np.nan, np.nan]])
        # Pad diagnostics with NaN row
        diag_metrics = np.vstack([diag_metrics, nan_row])

    data_dict = {
        **{col: cand_param_array[:, i] for i, col in enumerate(param_cols)},
        'candidate_speed': actual_speed_hist[:n_iters],
        'best_speed': best_speed_hist[:n_iters],
        'max_ei': diag_metrics[:n_iters, 1],
        'median_sigma': diag_metrics[:n_iters, 2], 
        'mll': diag_metrics[:n_iters, 3]
    }
    
    # Verify all arrays have the same length before creating DataFrame
    lengths = [len(v) for v in data_dict.values()]
    assert len(set(lengths)) == 1, f"Arrays have different lengths: {lengths}"
    
    pd.DataFrame(data_dict).to_csv(out_dir/f"{acq_name}_params_speed_combined_set.csv", index=True)

    # plot best speed
    plt.figure(figsize=(5,4))
    plt.plot(best_speed_hist, marker='o')
    plt.title(f"{acq_name}: Best Speed vs Iter")
    plt.xlabel("Iteration"); plt.ylabel("Speed (m/s)"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir/f"{acq_name}_best_speed.png", dpi=150)
    plt.close()

    return {
        "best_speed_history":    np.array(best_speed_hist),
        "candidate_param_hist":  np.array(cand_param_hist),
        "actual_speed_history":  np.array(actual_speed_hist),
        "diagnostics":           diagnostics,
    }

# =============================================================================
# 4. Loop over the 10 LHS sets
# =============================================================================
for set_idx in range(1):
    # paths
    base_data = Path(
        "C:/Users/chike/Box/TurtleRobotExperiments/"
        "Sea_Turtle_Robot_AI_Powered_Simulations_Project/"
        "NnamdiFiles/mujocotest1/assets/Gait-Optimization/data"
    )
    csv_path = base_data / f"combined_lhs_simulation_results_{set_idx}.csv"
    fig_dir  = base_data / f"figs/optplots_combinedset_{set_idx}"
    out_dir  = fig_dir / f"bo_results_set_{set_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # offline_df          = pd.read_csv(csv_path)
    param_cols          = ['stance_freq','swing_freq','a_param','lambda_cpl','A_front','A_back','A_hip']
    objective_col       = 'Average_Forward_Speed'
    # offline_params_orig = offline_df[param_cols].values
    # offline_objectives  = offline_df[[objective_col]].values

    # load offline data
    df      = pd.read_csv(csv_path)
    X_orig  = df[param_cols].values
    Y_orig  = df[[objective_col]].values
    X_norm  = normalize(torch.tensor(X_orig, dtype=torch.double), orig_bounds)
    Y_neg   = -torch.tensor(Y_orig, dtype=torch.double)
    Yvar    = torch.full_like(Y_neg, 1e-6)

    # run BO for each acquisition
    results = {}
    for acqf in acq_functions:
        print(f"\n=== SET {set_idx}, ACQ {acqf} ===")
        results[acqf] = run_bayesian_opt(
            acqf,
            train_x=X_norm.clone(),
            train_y=Y_neg.clone(),
            train_yvar=Yvar.clone(),
            orig_bounds=orig_bounds,
            norm_bounds=norm_bounds,
            num_iterations=100,  # Reduced from 100 since we start with more data
            out_dir=out_dir
        )

    # plot parameter evolution & speed comparisons
    for acqf in acq_functions:
        res = results[acqf]
        cand = res["candidate_param_hist"]
        speeds = res["actual_speed_history"]
        iters = np.arange(cand.shape[0])
        dim = cand.shape[1]
        nrows = (dim+1)//2
        fig, axes = plt.subplots(nrows, 2, figsize=(12,3*nrows))
        axes = axes.flatten()
        for d in range(dim):
            ax = axes[d]
            ax.plot(iters, cand[:,d], marker='o', label=param_cols[d])
            ax.set_xlabel("Iteration"); ax.set_ylabel(param_cols[d]); ax.grid(True)
            ax2 = ax.twinx()
            ax2.plot(iters, speeds, marker='x', linestyle='--', color='red')
            ax2.set_ylabel("Speed (m/s)")
        fig.tight_layout(rect=[0,0.03,1,0.95])
        plt.suptitle(f"Set {set_idx} — {acqf} Candidate Evolution")
        fig.savefig(fig_dir/f"{acqf}_candidate_evolution.png", dpi=150)
        plt.close(fig)

    # comparative speed plot
    fig, ax = plt.subplots(figsize=(6,4))
    for acqf in acq_functions:
        ax.plot(results[acqf]["best_speed_history"], marker='o', label=acqf)
    ax.set_title(f"Set {set_idx}: Best Speed Comparison")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Speed (m/s)"); ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir/"speed_comparison.png", dpi=150)
    plt.close(fig)

    # GP diagnostics plot
    fig, ax = plt.subplots(figsize=(6,4))
    for acqf in acq_functions:
        diag = np.array(results[acqf]["diagnostics"])
        ax.plot(diag[:,0], diag[:,3], marker='o', label=acqf)
    ax.set_title(f"Set {set_idx}: GP MLL over Iterations")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Marginal LL"); ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir/"gp_mll.png", dpi=150)
    plt.close(fig)

    # Add to your code to compare final best speeds
    combined_best = max(results[acqf]["best_speed_history"] for acqf in acq_functions)
    print(f"Best speed from combined dataset: {combined_best:.4f} m/s")

    print(f"=== Finished set {set_idx} ===\n")
