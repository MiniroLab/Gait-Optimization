import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import qmc

# =============================================================================
#            Hopf Oscillator Dynamics Function
# =============================================================================
def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
    """
    Integrate one step of the Hopf oscillator (with coupling) using Euler's method.
    
    Base dynamics:
      dx/dt = α (μ - (x²+y²)) x - ω y
      dy/dt = α (μ - (x²+y²)) y + ω x

    Coupling term added to dy:
      dy += λ * Δ_i, where Δ_i = Σ_{j≠i}[ y_j cos(θ_ji) - x_j sin(θ_ji) ],
      with θ_ji = 2π (φ_i - φ_j), and the φ's are provided in phase_offsets.
    """
    r_sq = x * x + y * y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    delta = 0.0
    # Here phase_offsets is a list of values (between 0 and 1) for each oscillator.
    phase_list = [phase_offsets[i] for i in range(len(x_all))]
    for j in range(len(x_all)):
        if j == index:
            continue
        theta_ji = 2.0 * np.pi * (phase_list[index] - phase_list[j])
        delta += y_all[j] * np.cos(theta_ji) - x_all[j] * np.sin(theta_ji)
    dy += coupling * delta

    x_new = x + dx * dt
    y_new = y + dy * dt
    return x_new, y_new

# =============================================================================
#        Parameters and Latin Hypercube Sampling for initial conditions
# =============================================================================
# Here, we assume the design is for 7 parameters, but for visualizing the CPG only,
# we will use fixed values for all except perhaps alpha. For demonstration, we generate
# initial conditions with LHS in a 7D space, but we will only use the oscillator parameters.
# The typical parameters are:
#   alpha = 10.0, mu = 0.04, a_param = 10.0, stance_freq, swing_freq, coupling, etc.
#
# For the oscillator simulation we set some constant parameters.
alpha_val = 10.0     # Convergence speed; try varying this (e.g., 5, 10, 20)
mu_val = 1        # Determines amplitude: sqrt(mu) ~ 0.2
omega = 5.0          # For simplicity, we use a constant oscillator frequency.
coupling = 0.5       # Coupling constant

# Phase offsets for six oscillators:
phase_offsets = [0.0, 0.0, 0.0, 0.0, 0.75, 0.75]  # typical values for 6 oscillators

# For visualization, we choose a total simulation time and timestep:
T = 5.0              # seconds
dt = 0.05
num_steps = int(T/dt)
t_vals = np.linspace(0, T, num_steps)

# =============================================================================
#                   Initialize Oscillator States
# =============================================================================
num_oscillators = 6  # one per actuator
x = np.zeros(num_oscillators)
y = np.zeros(num_oscillators)

# Initialize using the desired phase offsets and adding a small random noise.
for i in range(num_oscillators):
    phase0 = phase_offsets[i] * 2.0 * np.pi
    x[i] = np.sqrt(mu_val) * np.cos(phase0) + 0.002 * np.random.randn()
    y[i] = np.sqrt(mu_val) * np.sin(phase0) + 0.002 * np.random.randn()

# =============================================================================
#                Simulate Oscillator Dynamics Over Time
# =============================================================================
# Preallocate arrays to store the x outputs over time.
X = np.zeros((num_steps, num_oscillators))
Y = np.zeros((num_steps, num_oscillators))

for k in range(num_steps):
    X[k, :] = x
    Y[k, :] = y
    # Save current states for all oscillators and then update each one.
    x_all = x.copy()
    y_all = y.copy()
    for i in range(num_oscillators):
        x[i], y[i] = hopf_step(x[i], y[i], alpha_val, mu_val, omega, dt, coupling, x_all, y_all, i, phase_offsets)

# =============================================================================
#                   Create an Animation of the Oscillator Outputs
# =============================================================================
fig, ax = plt.subplots()
lines = []
for i in range(num_oscillators):
    (line,) = ax.plot([], [], label=f'Oscillator {i+1}')
    lines.append(line)
ax.set_xlim(0, T)
# Set the y-limits to reflect the expected oscillator output (approximately between -sqrt(mu) and +sqrt(mu))
ax.set_ylim(-np.sqrt(mu_val)*1.2, np.sqrt(mu_val)*1.2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Oscillator x output')
ax.set_title(r'CPG Oscillator Evolution ($\alpha=' + f'{alpha_val}$, $\mu={mu_val}$)')
ax.legend()
ax.grid(True)

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate(frame):
    t = t_vals[:frame]
    for i, line in enumerate(lines):
        line.set_data(t, X[:frame, i])
    return lines

anim = animation.FuncAnimation(fig, animate, frames=num_steps, init_func=init,
                               interval=20, blit=True)

plt.show()

# =============================================================================
#           OPTIONAL: Plot Convergence Metrics
# =============================================================================
# Here, one can plot the norm difference between the oscillator output and its expected steady state,
# or simply plot the outputs over time to see the speed of convergence.
steady_state_amplitude = np.sqrt(mu_val)
plt.figure(figsize=(8, 4))
for i in range(num_oscillators):
    plt.plot(t_vals, np.abs(X[:, i]), label=f'Osc {i+1}')
plt.axhline(steady_state_amplitude, color='black', linestyle='--', label='Expected amplitude')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude of x output')
plt.title(r'Convergence of Oscillator Outputs ($\sqrt{\mu} \approx$ ' + f'{steady_state_amplitude:.3f})')
plt.legend()
plt.grid(True)
plt.show()
