import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------
# Hopf Oscillator Dynamics
# -----------------------------
def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
    """
    Integrate one step of the Hopf oscillator (with coupling) using Euler's method.
    
    Dynamics:
      dx/dt = α (μ - (x²+y²)) x - ω y
      dy/dt = α (μ - (x²+y²)) y + ω x
    Coupling:
      dy += λ * Δ_i, with Δ_i = Σ_{j≠i} [ y_j cos(θ_ji) - x_j sin(θ_ji) ],
      and θ_ji = 2π (φ_i - φ_j).
    """
    r_sq = x * x + y * y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x

    delta = 0.0
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

# -----------------------------
# Parameters
# -----------------------------
num_oscillators = 6  # e.g., six joints/actuators
alpha_val = 20    # Convergence speed
mu_val = 1.0        # Sets squared amplitude; expected limit cycle amplitude = sqrt(mu)
omega = 4.0          # Base frequency term for the oscillator
coupling = 0.7       # Coupling strength between oscillators

# For this example, we choose some phase offsets for the oscillators:
# These are given as fractions and will be mapped to an angle between 0 and 2π.
phase_offsets = [0.0, 0.0, 0.0, 0.0, 0.75, 0.75]

# -----------------------------
# Simulation Setup
# -----------------------------
T = 5.0          # total simulation time in seconds
dt = 0.05       # timestep
num_steps = int(T / dt)
t_vals = np.linspace(0, T, num_steps)

# Initialize oscillator states using the phase offsets.
x = np.zeros(num_oscillators)
y = np.zeros(num_oscillators)
for i in range(num_oscillators):
    phase0 = phase_offsets[i] * 2.0 * np.pi
    x[i] = np.sqrt(mu_val) * np.cos(phase0) + 0.002 * np.random.randn()
    y[i] = np.sqrt(mu_val) * np.sin(phase0) + 0.002 * np.random.randn()

# Preallocate arrays to store state trajectories.
X = np.zeros((num_steps, num_oscillators))
Y = np.zeros((num_steps, num_oscillators))

# -----------------------------
# Run simulation
# -----------------------------
for k in range(num_steps):
    X[k, :] = x
    Y[k, :] = y
    x_all = x.copy()
    y_all = y.copy()
    for i in range(num_oscillators):
        x[i], y[i] = hopf_step(x[i], y[i], alpha_val, mu_val, omega, dt, coupling,
                               x_all, y_all, i, phase_offsets)

# -----------------------------
# Animation: Plot Oscillator Trajectories on the Limit Cycle
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 6))

# Draw the expected limit cycle as a circle of radius sqrt(mu)
radius = np.sqrt(mu_val)
theta_circle = np.linspace(0, 2*np.pi, 200)
ax.plot(radius * np.cos(theta_circle), radius * np.sin(theta_circle), 
        '--', color='gray', label='Limit Cycle (radius = {:.2f})'.format(radius))

# Set up the scatter plot for oscillator positions.
scat = ax.scatter(X[0, :], Y[0, :], c='tab:blue', s=80, zorder=5)
ax.set_xlim(-1.5*radius, 1.5*radius)
ax.set_ylim(-1.5*radius, 1.5*radius)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title(r'CPG Limit Cycle Dynamics ($\alpha=${}, $\mu=${})'.format(alpha_val, mu_val))
ax.legend()
ax.grid(True)
ax.set_aspect('equal')

def init():
    scat.set_offsets(np.c_[X[0, :], Y[0, :]])
    return scat,

def animate(frame):
    # Update scatter positions for the current frame.
    positions = np.column_stack((X[frame, :], Y[frame, :]))
    scat.set_offsets(positions)
    return scat,

anim = animation.FuncAnimation(fig, animate, frames=num_steps, init_func=init,
                               interval=20, blit=True)

plt.show()

# -----------------------------
# Plot convergence of oscillator outputs (magnitude of x) as an example
# -----------------------------
plt.figure(figsize=(8, 4))
for i in range(num_oscillators):
    plt.plot(t_vals, np.abs(X[:, i]), label=f'Oscillator {i+1}')
plt.axhline(radius, color='black', linestyle='--', label='Expected amplitude')
plt.xlabel('Time (s)')
plt.ylabel('Magnitude of $x$ output')
plt.title(r'Convergence of Oscillator Outputs ($\sqrt{\mu} \approx$ ' + f'{radius:.3f})')
plt.legend()
plt.grid(True)
plt.show()
