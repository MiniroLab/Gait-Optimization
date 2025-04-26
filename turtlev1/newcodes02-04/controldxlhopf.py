#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Example: CPG-Controlled Dynamixel XL330 Motors (6 actuators)
#
# This script uses a Hopf oscillator–based CPG to generate rhythmic
# joint control commands and sends these goal positions via the Dynamixel SDK.
#
# Motor IDs 1 to 6 correspond to:
#   1: pos_frontleftflipper
#   2: pos_frontrightflipper
#   3: pos_backleft          (back motor)
#   4: pos_backright         (back motor)
#   5: pos_frontlefthip
#   6: pos_frontrighthip
#
# Ensure that the motors are running Dynamixel XL330 firmware and are connected to
# the COM port specified below.
################################################################################

import os, sys, time, numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dynamixel_sdk import *    # Uses Dynamixel SDK library

# ------------------------------------------------------------------------------
# Utility: Get a keypress (for exiting the loop)
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import tty, termios, select
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

# ------------------------------------------------------------------------------
# Control table addresses for Dynamixel XL330 (assumed similar to PRO series)
ADDR_TORQUE_ENABLE      = 64
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132

LEN_GOAL_POSITION       = 4
LEN_PRESENT_POSITION    = 4

PROTOCOL_VERSION        = 2.0

# Dynamixel settings
DXL_IDS = [1, 2, 3, 4, 5, 6]       # Motor IDs for the six motors
DEVICENAME = 'COM8'                # Adjust as needed (e.g., "COM8" on Windows or "/dev/ttyUSB0" on Linux)
BAUDRATE = 57600

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# Define example control range (ticks)
DXL_MINIMUM_POSITION_VALUE = 100
DXL_MAXIMUM_POSITION_VALUE = 4000
DXL_MOVING_STATUS_THRESHOLD = 20

# ------------------------------------------------------------------------------
# Initialize PortHandler and PacketHandler
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    getch()
    quit()

if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    getch()
    quit()

# Enable torque for all six motors
for dxl_id in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{dxl_id}] {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ID:{dxl_id}] {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Dynamixel [ID:{dxl_id}] has been successfully connected")

# Initialize GroupSyncWrite for sending goal positions
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

# Optionally, initialize GroupSyncRead for present positions
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
for dxl_id in DXL_IDS:
    if not groupSyncRead.addParam(dxl_id):
        print(f"[ID:{dxl_id}] groupSyncRead addparam failed")
        quit()

# ------------------------------------------------------------------------------
# Define actuator names corresponding to each motor
actuator_names = [
    "pos_frontleftflipper",   # Motor ID 1
    "pos_frontrightflipper",  # Motor ID 2
    "pos_backleft",           # Motor ID 3
    "pos_backright",          # Motor ID 4
    "pos_frontlefthip",       # Motor ID 5
    "pos_frontrighthip"       # Motor ID 6
]

# Define joint limits for mapping oscillator outputs (same for each here)
joint_limits = {}
for name in actuator_names:
    joint_limits[name] = (DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE)
    print(f"{name}: ctrl range = [{DXL_MINIMUM_POSITION_VALUE}, {DXL_MAXIMUM_POSITION_VALUE}]")

# Define the joint output mapping (offset and gain for the tanh-based mapping) (500 - 1500)
joint_output_map = {
    "pos_frontleftflipper":  {"offset": 2048, "gain": 1000}, 
    "pos_frontrightflipper": {"offset": 2048, "gain": -1000},
    "pos_backleft":          {"offset": 2048, "gain": 1000},
    "pos_backright":         {"offset": 2048, "gain": -1000},
    "pos_frontlefthip":      {"offset": 2048, "gain": 1000},
    "pos_frontrighthip":     {"offset": 2048, "gain": -1000}
}

# ------------------------------------------------------------------------------
# Hopf Oscillator Dynamics Function
def hopf_step(x, y, alpha, mu, omega, dt, coupling, x_all, y_all, index, phase_offsets):
    r_sq = x*x + y*y
    dx = alpha * (mu - r_sq) * x - omega * y
    dy = alpha * (mu - r_sq) * y + omega * x
    delta = 0.0
    phase_list = [phase_offsets[name] for name in actuator_names]
    for j in range(len(x_all)):
        if j == index:
            continue
        theta_ji = 2.0 * np.pi * (phase_list[index] - phase_list[j])
        delta += y_all[j] * np.cos(theta_ji) - x_all[j] * np.sin(theta_ji)
    dy += coupling * delta
    x_new = x + dx * dt
    y_new = y + dy * dt
    return x_new, y_new

# ------------------------------------------------------------------------------
# CPG (CPG parameters and oscillator initialization)
alpha_cpg = 20.0   # Convergence speed for oscillator dynamics
mu = 1.0           # Amplitude parameter
a_param = 10.0     # Parameter for frequency blending
stance_freq = 3.0  # Stance frequency (rad/s)
swing_freq = 4.0   # Swing frequency (rad/s)
lambda_cpl = 0.8   # Coupling strength


# phase_offsets = [0.0, 0.0, 0.0, 0.0, 0.75, 0.75] # sync phase offsets
phase_offsets_s = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.0,
    "pos_backleft":          0.0,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.75
}

# phase_offsets = [0.0, 0.5, 0.5, 0.0, 0.25, 0.75] # diag phase offsets
phase_offsets_d = {
    "pos_frontleftflipper":  0.0,
    "pos_frontrightflipper": 0.5,
    "pos_backleft":          0.5,
    "pos_backright":         0.0,
    "pos_frontlefthip":      0.75,
    "pos_frontrighthip":     0.25
}

phase_offsets = phase_offsets_s  # Choose one of the above phase offset sets

# Initialize oscillator states (each oscillator gets a slight random noise)
oscillators = {}
for name in actuator_names:
    phase0 = phase_offsets[name] * 2.0 * np.pi
    oscillators[name] = {
        "x": np.sqrt(mu) * np.cos(phase0) + 0.002 * np.random.randn(),
        "y": np.sqrt(mu) * np.sin(phase0) + 0.002 * np.random.randn()
    }

# Mapping function: maps oscillator output (x) to a motor position (integer ticks)
def map_oscillator_to_position(x_val, min_pos, max_pos):
    amplitude = np.sqrt(mu)
    offset = (min_pos + max_pos) / 2.0
    gain = (max_pos - min_pos) / 2.0
    desired_pos = offset + gain * (x_val / amplitude)
    return int(np.clip(desired_pos, min_pos, max_pos))

# ------------------------------------------------------------------------------
# Simulation / Control Loop Parameters
dt_cpg = 0.001           # Time step for CPG integration (seconds)
time_duration = 30.0     # Total run time in seconds

print("Starting control loop with CPG integration... Press ESC to quit.")
start_time = time.time()
last_loop_time = start_time

# Storage dictionaries
cpg_x = {name: [] for name in actuator_names}
cpg_y = {name: [] for name in actuator_names}
cpg_outputs = {name: [] for name in actuator_names}      # The mapped angles you send to motors
motor_positions = {name: [] for name in actuator_names}  # Actual positions read from Dynamixels
time_history = []  # (Optional) Keep track of time or iteration


try:
    while True:
        now = time.time()
        sim_time = now - start_time
        if sim_time >= time_duration:
            print("Time duration reached. Exiting loop.")
            break

        loop_dt = now - last_loop_time
        last_loop_time = now

        # Determine how many substeps to integrate based on elapsed time
        # steps = int(np.floor(loop_dt / dt_cpg))
        steps = max(1, int(np.floor(loop_dt / dt_cpg)))

        for _ in range(steps):
            x_all = [oscillators[name]["x"] for name in actuator_names]
            y_all = [oscillators[name]["y"] for name in actuator_names]
            for i, name in enumerate(actuator_names):
                x_i = oscillators[name]["x"]
                y_i = oscillators[name]["y"]
                # Compute instantaneous frequency via logistic blending of stance and swing:
                freq = (stance_freq / (1.0 + np.exp(-a_param * y_i))) + \
                       (swing_freq  / (1.0 + np.exp(a_param * y_i)))
                x_new, y_new = hopf_step(x_i, y_i, alpha_cpg, mu, freq, dt_cpg,
                                         lambda_cpl, x_all, y_all, i, phase_offsets)
                # Update oscillator state
                oscillators[name]["x"] = x_new
                oscillators[name]["y"] = y_new

        # After finishing all substeps, record the final oscillator state once per outer iteration:
        for name in actuator_names:
            cpg_x[name].append(oscillators[name]["x"])
            cpg_y[name].append(oscillators[name]["y"])


        # Clear GroupSyncWrite parameters
        groupSyncWrite.clearParam()

        # For each motor, map oscillator output to a goal position and add to sync write
        for dxl_id, name in zip(DXL_IDS, actuator_names):
            # min_pos, max_pos = joint_limits[name]
            # goal_pos = map_oscillator_to_position(oscillators[name]["x"], min_pos, max_pos)
            min_angle, max_angle = joint_limits[name]
            offset = joint_output_map[name]["offset"]
            gain = joint_output_map[name]["gain"]
            # Using tanh for nonlinearity in mapping
            desired_angle = offset + gain * np.tanh(oscillators[name]["x"])
            desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)
            # Convert desired angle to an integer value (tick value)
            goal_pos = int(desired_angle_clamped)

            # Store the mapped angle (the "goal" you sent to the motor)
            cpg_outputs[name].append(desired_angle_clamped)

            # Create parameter byte array in little-endian order
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(goal_pos)),
                DXL_HIBYTE(DXL_LOWORD(goal_pos)),
                DXL_LOBYTE(DXL_HIWORD(goal_pos)),
                DXL_HIBYTE(DXL_HIWORD(goal_pos))
            ]
            if not groupSyncWrite.addParam(dxl_id, param_goal_position):
                print(f"[ID:{dxl_id}] groupSyncWrite addparam failed")
                quit()        
        time_history.append(sim_time)   # This line records the time for each loop iteration

        # Transmit the sync write packet to all motors
        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(packetHandler.getTxRxResult(dxl_comm_result))

        # Optionally, read back present positions for logging (using GroupSyncRead)
        dxl_comm_result = groupSyncRead.txRxPacket()
        if dxl_comm_result == COMM_SUCCESS:
            for dxl_id, name in zip(DXL_IDS, actuator_names):
                if groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                    present_pos = groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    motor_positions[name].append(present_pos)
                    idx = DXL_IDS.index(dxl_id)
                    name = actuator_names[idx]
                    min_angle, max_angle = joint_limits[name]
                    offset = joint_output_map[name]["offset"]
                    gain = joint_output_map[name]["gain"]

                    # Compute the desired angle using the tanh-based mapping:
                    desired_angle = offset + gain * np.tanh(oscillators[name]["x"])
                    desired_angle_clamped = np.clip(desired_angle, min_angle, max_angle)

                    print(f"[ID:{dxl_id}] Goal: {int(desired_angle_clamped)} Present: {present_pos}")
                else:
                    motor_positions[name].append(None)
        else:
            # If the txRxPacket call failed, ensure we append a default value for every motor.
            for name in actuator_names:
                motor_positions[name].append(None)


        # Check for ESC key to exit loop (non-blocking)
        if os.name == 'nt':
            if msvcrt.kbhit():
                if msvcrt.getch() == b'\x1b':  # ESC key
                    print("ESC pressed. Exiting control loop.")
                    break
        else:
            dr, dw, de = select.select([sys.stdin], [], [], 0)
            if dr:
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    print("ESC pressed. Exiting control loop.")
                    break

    # --- Figure 1: Raw Oscillator x and y ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name in actuator_names:
        axs[0].plot(time_history, cpg_x[name], label=f"{name} - x")
        axs[1].plot(time_history, cpg_y[name], label=f"{name} - y")

    axs[0].set_ylabel("Oscillator X")
    axs[1].set_ylabel("Oscillator Y")
    axs[1].set_xlabel("Time (s)")  # or "Iteration"
    axs[0].legend()
    axs[1].legend()
    axs[0].grid(True)
    axs[1].grid(True)

    # --- Figure 2: Mapped Angles (the final commands) ---
    plt.figure(figsize=(10, 5))
    for name in actuator_names:
        plt.plot(time_history, cpg_outputs[name], label=name)
    plt.title("Mapped Angles (Sent to Motors) Over Time")
    plt.xlabel("Time (s)")  # or "Iteration"
    plt.ylabel("Angle (ticks)")
    plt.legend()
    plt.grid(True)

    # --- Figure 3 (Optional): Actual Motor Positions ---
    plt.figure(figsize=(10, 5))
    for name in actuator_names:
        plt.plot(time_history, motor_positions[name], label=name)
    plt.title("Actual Motor Positions (Present Position)")
    plt.xlabel("Time (s)")  # or "Iteration"
    plt.ylabel("Position (ticks)")
    plt.legend()
    plt.grid(True)

    plt.show()



except KeyboardInterrupt:
    print("KeyboardInterrupt caught. Exiting control loop.")

    
# ------------------------------------------------------------------------------
# Disable torque for all motors before exiting
for dxl_id in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{dxl_id}] {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ID:{dxl_id}] {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Dynamixel [ID:{dxl_id}] torque disabled successfully.")

portHandler.closePort()
print("Port closed. Exiting.")
