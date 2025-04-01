import mujoco as mj
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from setup import load_model_from_yaml

if "yaml_path" not in globals(): #if yaml not provided, use model and data from this
    raise ValueError("yaml_path must be provided.")
yaml_path = globals().get("yaml_path")

model, data = load_model_from_yaml.get_model_and_data_from_yaml(yaml_path)



# Find Joint ID and Joint Qpos Address
joint_id_dict = {}
joint_QPos_dict = {}

for joint_id in range(model.njnt):
    joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
    qpos_address = model.jnt_qposadr[joint_id]
    qvel_address = model.jnt_dofadr[joint_id]
    joint_type = model.jnt_type[joint_id]  # 0: free, 1: ball, 2: slide, 3: hinge
    print(f"Joint {joint_id} ({joint_name}): QPOS Address {qpos_address}, QVEL Address {qvel_address} Joint Type {joint_type}")
    joint_id_dict[joint_name] = joint_id
    joint_QPos_dict[joint_name] = qpos_address

# Simulation Step time
sim_step_time = model.opt.timestep
print(f"Simulation step time: {sim_step_time} seconds")



import numpy as np

class HopfCPG:
    def __init__(self, num_oscillators=4, alpha=20, mu=1, omega_swing=np.pi, beta=0.5, 
                 b=100, lambda_coupling=0.7, dt=0.05):
        """
        Initializes the Hopf CPG system with given parameters.

        :param num_oscillators: Number of coupled oscillators
        :param alpha: Convergence rate
        :param mu: Amplitude control
        :param omega_swing: Swing phase frequency
        :param beta: Load factor
        :param b: Transition sharpness parameter
        :param lambda_coupling: Coupling strength
        :param dt: Simulation time step
        """
        self.num_oscillators = num_oscillators
        self.alpha = alpha
        self.mu = mu
        self.omega_swing = omega_swing
        self.beta = beta
        self.b = b
        self.lambda_coupling = lambda_coupling
        self.dt = dt

        # Compute stance frequency
        self.omega_stance = (1 - beta) / beta * omega_swing

        # Initialize oscillator states
        self.x = np.random.uniform(-1, 1, num_oscillators)
        self.y = np.random.uniform(-1, 1, num_oscillators)
        self.y_dot = np.zeros(num_oscillators)  # Store velocity of y

        # Define phase shifts (ensuring coordination)
        self.theta = 2*np.pi*np.array([ 0 , 0.5  ,0.25 , 0.75, 0.5])
        #self.theta = 2*np.pi*np.array([ 0.25 , 0.5  ,0.5 , 0.25, 0.75])
        self.theta = 2*np.pi*np.array([ 0 , 0.5  ,0.5 , 0, 0.75])
        

    def step(self):
        """
        Advances the CPG state by one timestep using Euler integration.
        """
        omega = np.zeros(self.num_oscillators)  # Adaptive frequency
        delta = np.zeros(self.num_oscillators)  # Coupling influence

        # Compute adaptive frequency and coupling
        for j in range(self.num_oscillators):
            # Compute adaptive frequency ω_j
            omega[j] = self.omega_stance / (np.exp(-self.b * self.y[j]) + 1) + \
                       self.omega_swing / (np.exp(self.b * self.y[j]) + 1)

            # Compute coupling influence
            delta[j] = np.sum([self.y[k] * np.cos(self.theta[k] - self.theta[j]) - 
                               self.x[k] * np.sin(self.theta[k] - self.theta[j]) 
                               for k in range(self.num_oscillators) if k != j])

        # Update oscillator states using Euler integration
        y_old = self.y.copy()
        for j in range(self.num_oscillators):
            dx = self.alpha * ((self.mu ** 2 - self.x[j] ** 2 - self.y[j] ** 2) * self.x[j]) - omega[j] * self.y[j]
            dy = self.alpha * ((self.mu ** 2 - self.x[j] ** 2 - self.y[j] ** 2) * self.y[j]) + omega[j] * self.x[j] + \
                 self.lambda_coupling * delta[j]

            # Update states
            self.x[j] += dx * self.dt
            self.y[j] += dy * self.dt

            self.y_dot = y_old - self.y

        self.y = np.clip(self.y, -self.mu, self.mu)
        return self.y, self.x, self.y_dot   # The y-values are used as the control signals for joints

    def map_to_joint_angles(self , inverse = False, spinal = False):
        """
        Maps the CPG outputs to joint angles for shoulder and leg joints.
        
        :return: shoulder_angles, leg_angles (both are numpy arrays)
        """

        
        # Shoulder angles: Direct mapping using K * y
        shoulder_angles = 1.4 * self.y
    
        # Leg angles: You will define the mapping based on y_d
                
        leg_angles = np.where(self.y_dot >= 0, 0, 0.5*np.pi*(1-np.abs(self.y)**2))    

        # Apply inverse if needed
        if inverse:
            shoulder_angles *= -1
            leg_angles *= -1

        if spinal:
            spinal_angle = 0.7 * self.y
            return spinal_angle[4]

        return [shoulder_angles, leg_angles]
    
    def spinal_joint_control(self, operating_point=1, smooth_factor=1.5, max_amplitude=0.7):
        """
        Generates a spinal joint oscillation with a variable amplitude based on the operating point.
    
        :param operating_point: Steering input (-1 to 1). 0 = straight, -1 = max left, 1 = max right.
        :param smooth_factor: Controls smoothness of transition. Higher values make transitions sharper.
        :param max_amplitude: Maximum oscillation amplitude when moving straight.
        :return: Adjusted spinal joint angle.
        """
        # Ensure operating point stays in range [-1, 1]
        operating_point = np.clip(operating_point, -1, 1)

        # Smoothly scale oscillation amplitude based on turning amount
        smooth_amplitude = max_amplitude * (1 - np.tanh(smooth_factor * abs(operating_point)))  

        # Compute spinal joint angle using scaled oscillation
        spinal_angle = smooth_amplitude * self.y[4] + operating_point  

        # Clip the final spinal joint movement to avoid unrealistic bending
        spinal_angle = np.clip(spinal_angle, -np.pi/2, np.pi/2)

        return spinal_angle

    def reset(self):
        """
        Resets the oscillator states.
        """
        self.x = np.random.uniform(-1, 1, self.num_oscillators)
        self.y = np.random.uniform(-1, 1, self.num_oscillators)

    def set_parameters(self, alpha=None, mu=None, omega_swing=None, beta=None, lambda_coupling=None):
        """
        Updates the CPG parameters dynamically.
        """
        if alpha is not None:
            self.alpha = alpha
        if mu is not None:
            self.mu = mu
        if omega_swing is not None:
            self.omega_swing = omega_swing
            self.omega_stance = (1 - self.beta) / self.beta * omega_swing  # Recalculate stance frequency
        if beta is not None:
            self.beta = beta
            self.omega_stance = (1 - beta) / beta * self.omega_swing  # Recalculate stance frequency
        if lambda_coupling is not None:
            self.lambda_coupling = lambda_coupling



cpg = HopfCPG(num_oscillators = 5, dt = sim_step_time)


# Define a simple sinusoidal gait controller
def gait_controller(model, data, operating_point = 0):
    """Manually sets joint angles (not physics-based)"""
   
    cpg.step()

    data.ctrl[0] = cpg.spinal_joint_control(operating_point)
    # Shoulder 1 - Front Left
    # data.qpos[joint_QPos_dict['shoulder1_joint']] = cpg.map_to_joint_angles(inverse = True)[0][0]   #[pi/2 , -pi/2]
    # data.qpos[joint_QPos_dict['leg1_joint']] = cpg.map_to_joint_angles(inverse = False)[1][0]      #[0 , pi/2]

    data.ctrl[1] = cpg.map_to_joint_angles(inverse = True)[0][0] 
    data.ctrl[2] = cpg.map_to_joint_angles(inverse = False)[1][0] 

    

    # # Shoulder 2 - Front Right
    # data.qpos[joint_QPos_dict['shoulder2_joint']] = cpg.map_to_joint_angles(inverse = False)[0][1]      #[-pi/2, pi/2]
    # data.qpos[joint_QPos_dict['leg2_joint']] = cpg.map_to_joint_angles(inverse = True)[1][1]   #[0, -pi/2]

    data.ctrl[3] = cpg.map_to_joint_angles(inverse = False)[0][1] 
    data.ctrl[4] = cpg.map_to_joint_angles(inverse = True)[1][1] 
    
    # Shoulder 3 - Back Left
    # data.qpos[joint_QPos_dict['shoulder3_joint']] = cpg.map_to_joint_angles(inverse = True)[0][2]   #[pi/2 , -pi/2]
    # data.qpos[joint_QPos_dict['leg3_joint']] = cpg.map_to_joint_angles(inverse = False)[1][2]      #[0, +pi/2]
    data.ctrl[5] = cpg.map_to_joint_angles(inverse = True)[0][2] 
    data.ctrl[6] = cpg.map_to_joint_angles(inverse = False)[1][2] 

    # Shoulder 4 - Back Right
    # data.qpos[joint_QPos_dict['shoulder4_joint']] = cpg.map_to_joint_angles(inverse = False)[0][3]        #[-pi/2, pi/2]
    # data.qpos[joint_QPos_dict['leg4_joint']] = cpg.map_to_joint_angles(inverse = True)[1][3]  #[0, -pi/2]
    data.ctrl[7] = cpg.map_to_joint_angles(inverse = False)[0][3] 
    data.ctrl[8] = cpg.map_to_joint_angles(inverse = True)[1][3] 
       

         

# Register the controller (so it's called automatically in mj_step)
mj.set_mjcb_control(gait_controller)

# Launch the MuJoCo model viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    simulation_time = 30.0  # Run for 10 seconds

    while viewer.is_running() and (time.time() - start_time) < simulation_time:
        mj.mj_step(model, data)  # Controller is called inside mj_step()
        viewer.sync()  # Sync the viewer with the simulation



