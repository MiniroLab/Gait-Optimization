import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import sys
import time

# Paths
xml_path = 'c:/Users/chike/Box/TurtleRobotExperiments/Sea_Turtle_Robot_AI_Powered_Simulations_Project/NnamdiFiles/mujocotest1/assets/turtlev1/testrobot1.xml'
servo_data_path = "/mnt/data/processed_servo_data.npy"

# Simulation settings
simend = 10  # Simulation time limit
print_camera_config = 0  # Print camera config if set to 1

# Load servo data
servo_data = np.load(servo_data_path)
seq = 0  # Sequence counter

# Initialize GLFW mouse states
button_left = button_middle = button_right = False
lastx = lasty = 0

# Function to get joint ID
def get_joint_id(model, joint_name):
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, joint_name)

# Define actuator names (assuming same order as CSV columns)
actuator_names = [
    "pos_frontleftflipper", "pos_frontrightflipper", 
    "pos_backleft", "pos_backright", 
    "pos_frontlefthip", "pos_frontrighthip"
]

# Load MuJoCo model
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Initialize camera and visualization options
cam = mj.MjvCamera()
opt = mj.MjvOption()
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)

# Initialize GLFW
glfw.init()
window = glfw.create_window(1200, 900, "Turtle Robot Simulation", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Scene and context for rendering
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Initialize camera position
cam.azimuth = 90
cam.elevation = -45
cam.distance = 3
cam.lookat = np.array([1.0, 0.0, 0.0])

# Assign actuator indices
actuator_indices = {name: get_joint_id(model, name) for name in actuator_names}

# Controller function
def controller(model, data):
    global seq
    servo_values = servo_data[seq]

    # Apply actuator values from CSV data
    for idx, name in enumerate(actuator_names):
        data.ctrl[actuator_indices[name]] = servo_values[idx]

    # Update sequence index cyclically
    seq = (seq + 1) % len(servo_data)

# Install controller callback
mj.set_mjcb_control(controller)

# Mouse and keyboard event handlers
def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx, dy = xpos - lastx, ypos - lasty
    lastx, lasty = xpos, ypos

    if not (button_left or button_middle or button_right):
        return

    width, height = glfw.get_window_size(window)
    shift_pressed = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

    action = mj.mjtMouse.mjMOUSE_ZOOM
    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if shift_pressed else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if shift_pressed else mj.mjtMouse.mjMOUSE_ROTATE_V

    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)

# Set GLFW event callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Simulation loop
while not glfw.window_should_close(window):
    time_prev = data.time

    # Step simulation in real time
    while data.time - time_prev < 1.0 / 300.0:
        mj.mj_step(model, data)

    # Stop if simulation time exceeded
    if data.time >= simend:
        break

    # Get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Print camera config if enabled
    if print_camera_config:
        print(f'cam.azimuth = {cam.azimuth}; cam.elevation = {cam.elevation}; cam.distance = {cam.distance}')
        print(f'cam.lookat = np.array([{cam.lookat[0]}, {cam.lookat[1]}, {cam.lookat[2]}])')

    # Update and render scene
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # Swap OpenGL buffers and process events
    glfw.swap_buffers(window)
    glfw.poll_events()

# Terminate GLFW and restart script
glfw.terminate()
os.execl(sys.executable, *sys.argv)
