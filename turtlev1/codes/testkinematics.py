# This script verifies the dynamic and kinematic models of a 2-DoF front flipper
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#############################################
# Updated Parameters (from MuJoCo XML)
#############################################
# Front flipper is modeled as a 2-DoF chain:
#   - Joint 1: Front hip joint (up–down/elevation)
#   - Joint 2: Front flipper joint (fore–aft/rowing)
# Parameters are estimated based on XML inertial and position info.
m1 = 0.0214873      # Mass of the front hip link (kg)
m2 = 0.0291756      # Mass of the front flipper link (kg)
r1 = 0.061          # Approx. distance from hip joint to COM of hip link (m)
r2 = 0.064          # Approx. distance from flipper joint to COM of flipper link (m)
L1 = 0.071          # Estimated distance from hip joint to flipper joint (m) [from XML: ~0.071 m]
L2 = 0.05           # Assumed length from flipper joint to flipper tip (m) (e.g., mesh extent)
I1 = 2.5e-6         # Estimated rotational inertia of the hip link (kg·m^2)
I2 = 2.3e-5         # Estimated rotational inertia of the flipper link (kg·m^2)
g  = 9.81           # Gravity (m/s^2)

#############################################
# 1. Dynamic Model Verification
#############################################
# The dynamic model for the front flipper is given by:
#    M(q)*q_ddot + C(q,q_dot) + G(q) = tau
# where q = [alpha, beta]. We simulate the response with zero torques (tau = [0, 0])
# to observe the free dynamics due to gravity.

def tau_dyn(t):
    # Zero torque input for verification
    return np.array([0.0, 0.0])

def dynamics(t, x):
    # x = [alpha, beta, alpha_dot, beta_dot]
    alpha, beta, alpha_dot, beta_dot = x

    # Mass matrix (from a simplified two-link model)
    M11 = I1 + I2 + m1*r1**2 + m2*(L1**2 + r2**2 + 2*L1*r2*np.cos(beta))
    M12 = I2 + m2*(r2**2 + L1*r2*np.cos(beta))
    M21 = M12
    M22 = I2 + m2*r2**2
    M = np.array([[M11, M12],
                  [M21, M22]])

    # Coriolis/centrifugal terms (simplified)
    C1 = -m2 * L1 * r2 * np.sin(beta) * beta_dot
    C2 = m2 * L1 * r2 * np.sin(beta) * alpha_dot
    C = np.array([C1, C2])

    # Gravity terms (assuming gravity acts downward along z,
    # but here the effective gravitational torque is computed with cosines of the joint angles)
    G1 = (m1*r1 + m2*L1)*g*np.cos(alpha) + m2*r2*g*np.cos(alpha+beta)
    G2 = m2*r2*g*np.cos(alpha+beta)
    G_vec = np.array([G1, G2])
    
    # Compute joint accelerations: M*q_ddot = tau - C - G
    q_ddot = np.linalg.solve(M, tau_dyn(t) - C - G_vec)
    
    return [alpha_dot, beta_dot, q_ddot[0], q_ddot[1]]

# Simulation parameters for dynamic verification
t_start = 0.0
t_end = 5.0  # seconds
num_points = 500
t_eval = np.linspace(t_start, t_end, num_points)

# Initial state: small perturbation from equilibrium (in radians, rad/s)
x0_dyn = [0.1, 0.05, 0.0, 0.0]

# Simulate dynamics
sol_dyn = solve_ivp(dynamics, [t_start, t_end], x0_dyn, t_eval=t_eval, method='RK45')
time_dyn = sol_dyn.t
alpha_dyn = sol_dyn.y[0, :]
beta_dyn  = sol_dyn.y[1, :]

# Plot dynamic simulation results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_dyn, alpha_dyn, 'b-', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('$\\alpha$ [rad]')
plt.title('Dynamic Simulation: Front Flipper Joint Angle $\\alpha$')
plt.subplot(2, 1, 2)
plt.plot(time_dyn, beta_dyn, 'r-', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('$\\beta$ [rad]')
plt.title('Dynamic Simulation: Front Flipper Joint Angle $\\beta$')
plt.tight_layout()
plt.show()

#############################################
# 2. Kinematic Model Verification
#############################################
# For a 2-DoF planar manipulator, the forward kinematics (FK) are:
#   x = L1*cos(alpha) + L2*cos(alpha + beta)
#   y = L1*sin(alpha) + L2*sin(alpha + beta)
# And the inverse kinematics (IK) can be derived using the law of cosines.
# We verify the consistency between FK and IK.

def forward_kinematics(alpha, beta, L1, L2):
    """
    Computes the (x, y) position of the front flipper tip given joint angles.
    """
    x = L1 * np.cos(alpha) + L2 * np.cos(alpha + beta)
    y = L1 * np.sin(alpha) + L2 * np.sin(alpha + beta)
    return x, y

def inverse_kinematics(x, y, L1, L2, elbow_up=True):
    """
    Computes the joint angles (alpha, beta) from end-effector (x, y) position.
    """
    # Compute the distance from the origin to the end-effector
    r = np.sqrt(x**2 + y**2)
    # Compute beta using the law of cosines
    cos_beta = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    if not elbow_up:
        beta = -beta
    # Compute alpha from geometric relations
    alpha = np.arctan2(y, x) - np.arctan2(L2*np.sin(beta), L1 + L2*np.cos(beta))
    return alpha, beta

# Generate test grid of joint angles for FK/IK verification
alphas_test = np.linspace(-np.pi/6, np.pi/6, 5)
betas_test = np.linspace(-np.pi/8, np.pi/8, 5)

alpha_orig = []
beta_orig  = []
alpha_recov = []
beta_recov  = []

for a in alphas_test:
    for b in betas_test:
        alpha_orig.append(a)
        beta_orig.append(b)
        # Forward kinematics to compute tip position
        x_fk, y_fk = forward_kinematics(a, b, L1, L2)
        # Inverse kinematics to recover joint angles
        a_rec, b_rec = inverse_kinematics(x_fk, y_fk, L1, L2, elbow_up=True)
        alpha_recov.append(a_rec)
        beta_recov.append(b_rec)

alpha_orig = np.array(alpha_orig)
beta_orig  = np.array(beta_orig)
alpha_recov = np.array(alpha_recov)
beta_recov  = np.array(beta_recov)

# Compute errors between original and recovered angles
alpha_error = alpha_orig - alpha_recov
beta_error = beta_orig - beta_recov

# Plot comparison of original and recovered joint angles
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(alpha_orig, 'bo-', label='Original $\\alpha$')
plt.plot(alpha_recov, 'rx--', label='Recovered $\\alpha$')
plt.xlabel('Test Index')
plt.ylabel('Angle [rad]')
plt.title('Forward/Inverse Kinematics Verification for $\\alpha$')
plt.legend()

plt.subplot(1,2,2)
plt.plot(beta_orig, 'bo-', label='Original $\\beta$')
plt.plot(beta_recov, 'rx--', label='Recovered $\\beta$')
plt.xlabel('Test Index')
plt.ylabel('Angle [rad]')
plt.title('Forward/Inverse Kinematics Verification for $\\beta$')
plt.legend()

plt.tight_layout()
plt.show()

print("Mean absolute error in alpha: {:.6e} rad".format(np.mean(np.abs(alpha_error))))
print("Mean absolute error in beta: {:.6e} rad".format(np.mean(np.abs(beta_error))))






# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp

# # Define model parameters (example values)
# m1 = 0.5      # Mass of link 1 (kg)
# m2 = 0.3      # Mass of link 2 (kg)
# r1 = 0.25     # Distance to center of mass of link 1 (m)
# r2 = 0.15     # Distance to center of mass of link 2 (m)
# Lf = 0.5      # Length of link 1 (m) (shoulder to elbow)
# I1 = 0.02     # Inertia of link 1 (kg.m^2)
# I2 = 0.01     # Inertia of link 2 (kg.m^2)
# g  = 9.81     # Gravity (m/s^2)

# # Define torque inputs as a function of time (here we set torques to zero)
# def tau(t):
#     return np.array([0.0, 0.0])

# # Define the dynamic model for the 2-DoF front flipper.
# # The state vector is x = [alpha, beta, alpha_dot, beta_dot]
# def dynamics(t, x):
#     alpha, beta, alpha_dot, beta_dot = x

#     # Mass matrix elements (simplified model)
#     M11 = I1 + I2 + m1*r1**2 + m2*(Lf**2 + r2**2 + 2*Lf*r2*np.cos(beta))
#     M12 = I2 + m2*(r2**2 + Lf*r2*np.cos(beta))
#     M21 = M12
#     M22 = I2 + m2*r2**2

#     M = np.array([[M11, M12],
#                   [M21, M22]])
    
#     # Coriolis and centrifugal terms
#     C1 = -m2 * Lf * r2 * np.sin(beta) * beta_dot
#     C2 = m2 * Lf * r2 * np.sin(beta) * alpha_dot
#     C = np.array([C1, C2])
    
#     # Gravity terms
#     G1 = (m1*r1 + m2*Lf)*g*np.cos(alpha) + m2*r2*g*np.cos(alpha+beta)
#     G2 = m2*r2*g*np.cos(alpha+beta)
#     G = np.array([G1, G2])
    
#     # Get torque inputs
#     T = tau(t)
    
#     # Compute accelerations: M * q_ddot = tau - C - G  => q_ddot = M^{-1}(tau - C - G)
#     q_ddot = np.linalg.solve(M, T - C - G)
    
#     return [alpha_dot, beta_dot, q_ddot[0], q_ddot[1]]

# # Set up simulation parameters
# t_start = 0.0
# t_end = 5.0  # seconds
# num_points = 500
# t_eval = np.linspace(t_start, t_end, num_points)

# # Initial state: [alpha, beta, alpha_dot, beta_dot] in radians and rad/s
# x0 = [0.1, 0.05, 0.0, 0.0]

# # Solve the ODE
# sol = solve_ivp(dynamics, [t_start, t_end], x0, t_eval=t_eval, method='RK45')

# # Extract simulation results
# time_sim = sol.t
# alpha_sim = sol.y[0, :]
# beta_sim  = sol.y[1, :]

# # Plot the simulation results
# plt.figure(figsize=(10, 6))

# plt.subplot(2, 1, 1)
# plt.plot(time_sim, alpha_sim, 'b-', linewidth=2, label='Simulated $\\alpha$')
# plt.xlabel('Time [s]')
# plt.ylabel('$\\alpha$ [rad]')
# plt.title('Front Flipper Joint Angle $\\alpha$')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(time_sim, beta_sim, 'r-', linewidth=2, label='Simulated $\\beta$')
# plt.xlabel('Time [s]')
# plt.ylabel('$\\beta$ [rad]')
# plt.title('Front Flipper Joint Angle $\\beta$')
# plt.legend()

# plt.tight_layout()
# plt.show()





# import numpy as np
# import matplotlib.pyplot as plt

# def forward_kinematics(alpha, beta, L1, L2):
#     """
#     Computes the (x, y) position of the end-effector given joint angles.
    
#     Parameters:
#       alpha: angle of the first joint (in radians)
#       beta: angle of the second joint (in radians)
#       L1: length of the first link (shoulder-to-elbow)
#       L2: length of the second link (elbow-to-tip)
    
#     Returns:
#       (x, y): Coordinates of the end-effector
#     """
#     x = L1 * np.cos(alpha) + L2 * np.cos(alpha + beta)
#     y = L1 * np.sin(alpha) + L2 * np.sin(alpha + beta)
#     return x, y

# def inverse_kinematics(x, y, L1, L2, elbow_up=True):
#     """
#     Recovers the joint angles from the end-effector (x, y) position using geometric IK.
    
#     Parameters:
#       x, y: End-effector coordinates
#       L1: Length of the first link
#       L2: Length of the second link
#       elbow_up: Boolean flag to choose the elbow-up solution (default True)
      
#     Returns:
#       (alpha, beta): Estimated joint angles (in radians)
#     """
#     # Distance from origin to the end-effector
#     r = np.sqrt(x**2 + y**2)
    
#     # Law of cosines to compute beta
#     cos_beta = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
#     # Clamp cos_beta to account for numerical errors
#     cos_beta = np.clip(cos_beta, -1.0, 1.0)
#     beta = np.arccos(cos_beta)
#     if not elbow_up:
#         beta = -beta
    
#     # Compute alpha using the geometry of the manipulator
#     alpha = np.arctan2(y, x) - np.arctan2(L2 * np.sin(beta), L1 + L2 * np.cos(beta))
#     return alpha, beta

# # Define link lengths (example values)
# L1 = 0.5  # Length from shoulder to elbow (meters)
# L2 = 0.3  # Length from elbow to tip (meters)

# # Create a grid of test joint angles
# alphas = np.linspace(-np.pi/4, np.pi/4, 5)  # Range for alpha
# betas = np.linspace(-np.pi/6, np.pi/6, 5)    # Range for beta

# # Lists to store the original and recovered angles
# alpha_orig = []
# beta_orig  = []
# alpha_recov = []
# beta_recov  = []

# # Test over a grid of angles
# for a in alphas:
#     for b in betas:
#         alpha_orig.append(a)
#         beta_orig.append(b)
        
#         # Forward kinematics: compute end-effector position
#         x, y = forward_kinematics(a, b, L1, L2)
        
#         # Inverse kinematics: recover joint angles from (x, y)
#         a_rec, b_rec = inverse_kinematics(x, y, L1, L2, elbow_up=True)
#         alpha_recov.append(a_rec)
#         beta_recov.append(b_rec)

# # Convert lists to arrays for comparison
# alpha_orig = np.array(alpha_orig)
# beta_orig  = np.array(beta_orig)
# alpha_recov = np.array(alpha_recov)
# beta_recov  = np.array(beta_recov)

# # Compute errors between original and recovered angles
# alpha_error = alpha_orig - alpha_recov
# beta_error = beta_orig - beta_recov

# # Plot original vs. recovered joint angles
# plt.figure(figsize=(12, 5))

# plt.subplot(1,2,1)
# plt.plot(alpha_orig, 'bo-', label='Original $\\alpha$')
# plt.plot(alpha_recov, 'rx--', label='Recovered $\\alpha$')
# plt.xlabel('Test Index')
# plt.ylabel('Angle [rad]')
# plt.title('Comparison of $\\alpha$')
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(beta_orig, 'bo-', label='Original $\\beta$')
# plt.plot(beta_recov, 'rx--', label='Recovered $\\beta$')
# plt.xlabel('Test Index')
# plt.ylabel('Angle [rad]')
# plt.title('Comparison of $\\beta$')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Optionally, print mean absolute errors
# print("Mean absolute error in alpha: {:.6f} rad".format(np.mean(np.abs(alpha_error))))
# print("Mean absolute error in beta: {:.6f} rad".format(np.mean(np.abs(beta_error))))

