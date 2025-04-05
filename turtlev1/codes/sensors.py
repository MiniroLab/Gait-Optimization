"""
sensor.py

This module defines helper functions for retrieving sensor data from a MuJoCo model,
as well as functions to extract various simulation metrics such as base pose, joint angles,
IMU readings, center-of-mass (CoM) data, joint positions/velocities, actuator forces,
and joint torques.

Usage:
    import sensor
    sensor_name2id = sensor.build_sensor_name2id(model)
    metrics = sensor.all_metrics(model, data, sensor_name2id)
"""

import numpy as np
import mujoco  # Ensure you have the mujoco Python bindings installed

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def build_sensor_name2id(model):
    """
    Builds and returns a dictionary mapping sensor names to their sensor index in the model.
    """
    sensor_name2id = {}
    for i in range(model.nsensor):
        name_adr = model.sensor_adr[i]
        name_chars = []
        # model.names is an array of int8 values; we extract the string until we hit a 0.
        for c in model.names[name_adr:]:
            if c == 0:
                break
            name_chars.append(chr(c))
        sensor_name = "".join(name_chars)
        sensor_name2id[sensor_name] = i
    return sensor_name2id

def get_sensor_data(data, model, sensor_name2id, sname):
    """
    Retrieves the sensor reading for the given sensor name.
    
    Parameters:
        data: The MuJoCo data object.
        model: The MuJoCo model object.
        sensor_name2id: A dictionary mapping sensor names to their indices.
        sname: The sensor name (string).
        
    Returns:
        A NumPy array containing the sensor reading or None if the sensor is not found.
    """
    if sname not in sensor_name2id:
        return None
    sid = sensor_name2id[sname]
    dim = model.sensor_dim[sid]
    start_idx = model.sensor_adr[sid]
    return data.sensordata[start_idx : start_idx + dim].copy()

def get_actuator_index(model, name):
    """
    Returns the index of the actuator with the given name.
    
    Parameters:
        model: The MuJoCo model object.
        name: The actuator name (string).
        
    Returns:
        The index of the actuator.
        
    Raises:
        ValueError if the actuator is not found.
    """
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name == name:
            return i
    raise ValueError(f"Actuator '{name}' not found in model.")

# -----------------------------------------------------------------------------
# Sensor Metric Functions
# -----------------------------------------------------------------------------

def base_pose(data):
    """
    Returns the global pose of the base (free joint) as a 7-element array.
    The first 7 elements of qpos correspond to the base free joint state.
    """
    return data.qpos[:7]

def joint_angles(data):
    """
    Returns the robot's joint angles (all degrees of freedom beyond the free base).
    Assumes that data.qpos[7:] contains the joint angles.
    """
    return data.qpos[7:]

# --- IMU Sensor Functions (attached to the "imu" site) ---

def imu_position(model, data, sensor_name2id):
    """Returns the global position of the IMU as measured by 'sens_imu_position'."""
    return get_sensor_data(data, model, sensor_name2id, "sens_imu_position")

def imu_orientation(model, data, sensor_name2id):
    """Returns the orientation (quaternion) of the IMU as measured by 'sens_imu_orientation'."""
    return get_sensor_data(data, model, sensor_name2id, "sens_imu_orientation")

def imu_gyro(model, data, sensor_name2id):
    """Returns the angular velocity measured by the IMU's gyro ('sens_imu_gyro')."""
    return get_sensor_data(data, model, sensor_name2id, "sens_imu_gyro")

def imu_local_linvel(model, data, sensor_name2id):
    """Returns the local linear velocity measured by 'sens_imu_local_linvel'."""
    return get_sensor_data(data, model, sensor_name2id, "sens_imu_local_linvel")

def imu_accelerometer(model, data, sensor_name2id):
    """Returns the linear acceleration measured by 'sens_imu_accelerometer'."""
    return get_sensor_data(data, model, sensor_name2id, "sens_imu_accelerometer")

def imu_upvector(model, data, sensor_name2id):
    """Returns the up vector (z-axis) from the IMU as measured by 'sens_imu_upvector'."""
    return get_sensor_data(data, model, sensor_name2id, "sens_imu_upvector")

def imu_forwardvector(model, data, sensor_name2id):
    """Returns the forward vector (x-axis) from the IMU as measured by 'sens_imu_forwardvector'."""
    return get_sensor_data(data, model, sensor_name2id, "sens_imu_forwardvector")

def imu_global_linvel(model, data, sensor_name2id):
    """Returns the global linear velocity of the IMU as measured by 'sens_imu_global_linvel'."""
    return get_sensor_data(data, model, sensor_name2id, "sens_imu_global_linvel")

def imu_global_angvel(model, data, sensor_name2id):
    """Returns the global angular velocity of the IMU as measured by 'sens_imu_global_angvel'."""
    return get_sensor_data(data, model, sensor_name2id, "sens_imu_global_angvel")

# --- Center-of-Mass (CoM) Functions ---

def com_position(model, data, sensor_name2id):
    """
    Returns the global position of the robot's center of mass.
    This sensor ('sens_com_robot') is attached to the 'base' body.
    """
    return get_sensor_data(data, model, sensor_name2id, "sens_com_robot")

def com_velocity(model, data, sensor_name2id):
    """
    Returns the global linear velocity of the robot's center of mass.
    This sensor ('sens_linvel_robot') is attached to the 'base' body.
    """
    return get_sensor_data(data, model, sensor_name2id, "sens_linvel_robot")

# --- Joint Sensor Functions ---

def joint_positions(model, data, sensor_name2id):
    """
    Returns a dictionary mapping each joint name to its measured position.
    """
    sensors = {
        "backright": get_sensor_data(data, model, sensor_name2id, "sens_backright_pos"),
        "backleft": get_sensor_data(data, model, sensor_name2id, "sens_backleft_pos"),
        "frontrighthip": get_sensor_data(data, model, sensor_name2id, "sens_frontrighthip_pos"),
        "frontrightflipper": get_sensor_data(data, model, sensor_name2id, "sens_frontrightflipper_pos"),
        "frontlefthip": get_sensor_data(data, model, sensor_name2id, "sens_frontlefthip_pos"),
        "frontleftflipper": get_sensor_data(data, model, sensor_name2id, "sens_frontleftflipper_pos"),
    }
    return sensors

def joint_velocities(model, data, sensor_name2id):
    """
    Returns a dictionary mapping each joint name to its measured velocity.
    """
    sensors = {
        "backright": get_sensor_data(data, model, sensor_name2id, "sens_backright_vel"),
        "backleft": get_sensor_data(data, model, sensor_name2id, "sens_backleft_vel"),
        "frontrighthip": get_sensor_data(data, model, sensor_name2id, "sens_frontrighthip_vel"),
        "frontrightflipper": get_sensor_data(data, model, sensor_name2id, "sens_frontrightflipper_vel"),
        "frontlefthip": get_sensor_data(data, model, sensor_name2id, "sens_frontlefthip_vel"),
        "frontleftflipper": get_sensor_data(data, model, sensor_name2id, "sens_frontleftflipper_vel"),
    }
    return sensors

# --- Actuator Force and Joint Torque Functions ---

def actuator_forces(model, data, sensor_name2id):
    """
    Returns a dictionary mapping each actuator to its measured force (or torque)
    as provided by the actuator force sensors.
    """
    forces = {
        "backright": get_sensor_data(data, model, sensor_name2id, "sens_force_backright"),
        "backleft": get_sensor_data(data, model, sensor_name2id, "sens_force_backleft"),
        "frontrighthip": get_sensor_data(data, model, sensor_name2id, "sens_force_frontrighthip"),
        "frontrightflipper": get_sensor_data(data, model, sensor_name2id, "sens_force_frontrightflipper"),
        "frontlefthip": get_sensor_data(data, model, sensor_name2id, "sens_force_frontlefthip"),
        "frontleftflipper": get_sensor_data(data, model, sensor_name2id, "sens_force_frontleftflipper"),
    }
    return forces

def joint_torques(model, data, sensor_name2id):
    """
    Returns a dictionary mapping each joint to its net actuator torque.
    This value is obtained from the joint actuator force sensors.
    """
    torques = {
        "backright": get_sensor_data(data, model, sensor_name2id, "sens_jointactfrc_backright"),
        "backleft": get_sensor_data(data, model, sensor_name2id, "sens_jointactfrc_backleft"),
        "frontrighthip": get_sensor_data(data, model, sensor_name2id, "sens_jointactfrc_frontrighthip"),
        "frontrightflipper": get_sensor_data(data, model, sensor_name2id, "sens_jointactfrc_frontrightflipper"),
        "frontlefthip": get_sensor_data(data, model, sensor_name2id, "sens_jointactfrc_frontlefthip"),
        "frontleftflipper": get_sensor_data(data, model, sensor_name2id, "sens_jointactfrc_frontleftflipper"),
    }
    return torques

# --- Combined Metrics Function ---

def all_metrics(model, data, sensor_name2id):
    """
    Returns a comprehensive dictionary of all relevant metrics from the robot.
    This includes base pose, joint angles, IMU readings, center-of-mass (CoM)
    position/velocity, joint positions/velocities, actuator forces, and joint torques.
    """
    metrics = {}
    # Base (free joint) pose and joint angles
    metrics["base_pose"] = data.qpos[:7]
    metrics["joint_angles"] = data.qpos[7:]
    
    # IMU measurements
    metrics["imu_position"] = get_sensor_data(data, model, sensor_name2id, "sens_imu_position")
    metrics["imu_orientation"] = get_sensor_data(data, model, sensor_name2id, "sens_imu_orientation")
    metrics["imu_gyro"] = get_sensor_data(data, model, sensor_name2id, "sens_imu_gyro")
    metrics["imu_local_linvel"] = get_sensor_data(data, model, sensor_name2id, "sens_imu_local_linvel")
    metrics["imu_accelerometer"] = get_sensor_data(data, model, sensor_name2id, "sens_imu_accelerometer")
    metrics["imu_upvector"] = get_sensor_data(data, model, sensor_name2id, "sens_imu_upvector")
    metrics["imu_forwardvector"] = get_sensor_data(data, model, sensor_name2id, "sens_imu_forwardvector")
    metrics["imu_global_linvel"] = get_sensor_data(data, model, sensor_name2id, "sens_imu_global_linvel")
    metrics["imu_global_angvel"] = get_sensor_data(data, model, sensor_name2id, "sens_imu_global_angvel")
    
    # Center-of-Mass metrics
    metrics["com_position"] = get_sensor_data(data, model, sensor_name2id, "sens_com_robot")
    metrics["com_velocity"] = get_sensor_data(data, model, sensor_name2id, "sens_linvel_robot")
    
    # Joint sensors
    metrics["joint_positions"] = joint_positions(model, data, sensor_name2id)
    metrics["joint_velocities"] = joint_velocities(model, data, sensor_name2id)
    
    # Actuator and joint torque sensors
    metrics["actuator_forces"] = actuator_forces(model, data, sensor_name2id)
    metrics["joint_torques"] = joint_torques(model, data, sensor_name2id)
    
    return metrics
