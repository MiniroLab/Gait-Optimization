# Redefining the necessary classes and imports since execution state was reset

import numpy as np
import matplotlib.pyplot as plt
import time

# Define the coupled Hopf oscillator-based CPG model
class CoupledHopfCPG:
    def __init__(self, num_oscillators=6, dt=0.01):
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.alpha = 2.0  # Reduced convergence speed for smoother motion
        self.mu = 1.0     # Maintain amplitude stability
        self.omega = np.array([1.0] * num_oscillators)  # Slower oscillation frequency
        
        # Initialize oscillator states
        self.x = np.zeros(num_oscillators)
        self.y = np.zeros(num_oscillators)
        self.phases = np.random.rand(num_oscillators) * 2 * np.pi  

        # Adjusted Phase coupling matrix for natural gait coordination
        self.K = np.array([
            [ 0, -0.3,  0.3,  0,  0,  0],
            [-0.3,  0,  0,  0.3,  0,  0],
            [ 0,  0.3,  0.3, 0,  0,  0],
            [ 0,  0.3, -0.3,  0,  0,  0],
            [ 0,  0,  0,  0,  0.3, -0.3],
            [ 0,  0,  0,  0, -0.3,  0.3]
        ])

    def update(self):
        """ Updates the Hopf oscillators with phase coupling. """
        r = np.sqrt(self.x**2 + self.y**2)

        # Compute phase synchronization term
        phase_sync = np.sum(self.K * np.sin(np.subtract.outer(self.phases, self.phases)), axis=1)

        # Hopf oscillator equations with phase coupling
        dx = self.alpha * (self.mu - r**2) * self.x - self.omega * self.y + phase_sync
        dy = self.alpha * (self.mu - r**2) * self.y + self.omega * self.x

        self.x += dx * self.dt
        self.y += dy * self.dt

        # Update phase values
        self.phases = np.arctan2(self.y, self.x)

        return self.x  # Joint angles

# Initialize the CPG system
cpg = CoupledHopfCPG(num_oscillators=6)

# Setup real-time plotting
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
lines = [ax.plot([], [], label=f'Joint {i+1}')[0] for i in range(6)]

ax.set_xlim(0, 200)  # Show the last 200 time steps
ax.set_ylim(-1.2, 1.2)  # Based on MuJoCo joint limits
ax.set_xlabel("Time Step")
ax.set_ylabel("Joint Angle Output")
ax.set_title("Real-Time Hopf CPG Oscillations")
ax.legend()
ax.grid()

# Store joint outputs
time_steps = 200  # Rolling window of displayed steps
outputs = np.zeros((time_steps, 6))

# Simulate and update plot in real-time
for t in range(1000):  # Simulating 1000 time steps
    outputs = np.roll(outputs, -1, axis=0)  # Shift data left
    outputs[-1] = cpg.update()  # Get new joint angles

    # Update plot lines
    for i, line in enumerate(lines):
        line.set_xdata(np.arange(time_steps))
        line.set_ydata(outputs[:, i])

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)  # Adjust speed of real-time updates

plt.ioff()
plt.show()
