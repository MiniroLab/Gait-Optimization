import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# CPG (Hopf Oscillator) Parameters
# ---------------------------
alpha = 10.0      # Convergence speed
mu = 1    # Desired amplitude squared (set to 1 for a full limit cycle)
a_param = 10.0    # Logistic steepness for frequency blending

# Define stance and swing frequencies (rad/s)
stance_freq = 3.0
swing_freq  = 3.0

# Coupling constant (λ)
lambda_cpl = 1  # Coupling strength (0.0 to 1.0)

# Time step for CPG integration (seconds)
dt_cpg = 0.01

# ---------------------------
# Actuator Names and Phase Offsets
# ---------------------------
actuator_names = ["osc1", "osc2", "osc3", "osc4", "osc5", "osc6"]
phase_offsets = {
    "osc1": 0.0,
    "osc2": 0.0,
    "osc3": 0.0,
    "osc4": 0.0,
    "osc5": 0.75,
    "osc6": 0.75
}

# ---------------------------
# Initialize Oscillator States
# ---------------------------
oscillators = {}
for name in actuator_names:
    phase0 = phase_offsets[name] * 2.0 * np.pi
    oscillators[name] = {
        "x": np.sqrt(mu) * np.cos(phase0),
        "y": np.sqrt(mu) * np.sin(phase0)
    }

# ---------------------------
# Hopf Oscillator Step Function
# ---------------------------
def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_list):
    r_sq = x * x + y * y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x
    
    # Coupling term
    delta = 0.0
    for j in range(len(x_all)):
        if j == index:
            continue
        theta_ji = 2.0 * np.pi * (phase_list[index] - phase_list[j])
        delta += y_all[j] * np.cos(theta_ji) - x_all[j] * np.sin(theta_ji)
    dy += coupling * delta
    
    return x + dx * dt, y + dy * dt

# ---------------------------
# Simulation Loop
# ---------------------------
steps = 1000  # number of simulation steps
history = {name: [] for name in actuator_names}

# Create a phase list in the order of actuator_names
phase_list = [phase_offsets[name] for name in actuator_names]

for _ in range(steps):
    # Get current oscillator values for coupling calculations
    x_all = [oscillators[name]["x"] for name in actuator_names]
    y_all = [oscillators[name]["y"] for name in actuator_names]
    for i, name in enumerate(actuator_names):
        # Compute instantaneous frequency based on oscillator's y value
        omega = (stance_freq / (1.0 + np.exp(-a_param * oscillators[name]["y"]))) + \
                (swing_freq  / (1.0 + np.exp(a_param * oscillators[name]["y"])))
        new_x, new_y = hopf_step(oscillators[name]["x"], oscillators[name]["y"],
                                 alpha, mu, omega, dt_cpg, lambda_cpl,
                                 x_all, y_all, i, phase_list)
        oscillators[name]["x"] = new_x
        oscillators[name]["y"] = new_y
        history[name].append(new_x)  # store the x component (can also store y)

# ---------------------------
# Plot the Oscillator Outputs
# ---------------------------
plt.figure(figsize=(10, 6))
for name in actuator_names:
    plt.plot(history[name], label=name)
plt.xlabel("Time Step")
plt.ylabel("Oscillator x Output")
plt.title("CPG Oscillator Outputs Over Time")
plt.legend()
plt.show()
