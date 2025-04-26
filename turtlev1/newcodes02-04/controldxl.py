#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dynamixel_sdk import *
import os


# Cross-platform getch
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

# ======================= CONFIGURATION =======================

# Actuator mappings
ACTUATOR_NAMES = [
    "pos_frontleftflipper",   # ID 1
    "pos_frontrightflipper",  # ID 2
    "pos_backleft",           # ID 3
    "pos_backright",          # ID 4
    "pos_frontlefthip",       # ID 5
    "pos_frontrighthip"       # ID 6
]
DXL_IDS = [1, 2, 3, 4, 5, 6]

# Control table address and lengths
ADDR_PRO_TORQUE_ENABLE      = 64
ADDR_PRO_GOAL_POSITION      = 116
ADDR_PRO_PRESENT_POSITION   = 132
LEN_PRO_GOAL_POSITION       = 4
LEN_PRO_PRESENT_POSITION    = 4

# Protocol
PROTOCOL_VERSION = 2.0

# Serial port and baudrate
DEVICENAME = 'COM8'   # Update as needed
BAUDRATE = 57600

# Torque constants
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# Motion constants
DXL_MINIMUM_POSITION_VALUE = 100
DXL_MAXIMUM_POSITION_VALUE = 4000
DXL_MOVING_STATUS_THRESHOLD = 20

# Motion positions
dxl_goal_position = [DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE]
index = 0

# ======================== INITIALIZATION ========================

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, LEN_PRO_GOAL_POSITION)
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION)

# Open port
if not portHandler.openPort():
    print("❌ Failed to open port")
    getch()
    quit()
print("✅ Port opened")

# Set baudrate
if not portHandler.setBaudRate(BAUDRATE):
    print("❌ Failed to set baudrate")
    getch()
    quit()
print("✅ Baudrate set")

# Enable torque for all motors
for i, dxl_id in enumerate(DXL_IDS):
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{dxl_id:03d}] {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ID:{dxl_id:03d}] {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"🔌 {ACTUATOR_NAMES[i]} (ID {dxl_id}) torque enabled")

# Add all IDs to sync read
for dxl_id in DXL_IDS:
    if not groupSyncRead.addParam(dxl_id):
        print(f"[ID:{dxl_id:03d}] groupSyncRead addparam failed")
        quit()

# ========================== MAIN LOOP ==========================

while True:
    print("\nPress any key to move actuators (ESC to quit)")
    if getch() == chr(0x1b):
        break

    # Convert position to byte array
    param_goal_position = [
        DXL_LOBYTE(DXL_LOWORD(dxl_goal_position[index])),
        DXL_HIBYTE(DXL_LOWORD(dxl_goal_position[index])),
        DXL_LOBYTE(DXL_HIWORD(dxl_goal_position[index])),
        DXL_HIBYTE(DXL_HIWORD(dxl_goal_position[index]))
    ]

    # Add goal position for all motors
    for dxl_id in DXL_IDS:
        if not groupSyncWrite.addParam(dxl_id, param_goal_position):
            print(f"[ID:{dxl_id:03d}] groupSyncWrite addparam failed")
            quit()

    # Send goal positions
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(packetHandler.getTxRxResult(dxl_comm_result))

    groupSyncWrite.clearParam()

    # Wait until all motors reach position
    while True:
        dxl_comm_result = groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(packetHandler.getTxRxResult(dxl_comm_result))

        reached = True
        for i, dxl_id in enumerate(DXL_IDS):
            if not groupSyncRead.isAvailable(dxl_id, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION):
                print(f"[ID:{dxl_id:03d}] groupSyncRead getdata failed")
                quit()
            pos = groupSyncRead.getData(dxl_id, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION)
            print(f"[ID:{dxl_id:03d}] {ACTUATOR_NAMES[i]} Goal:{dxl_goal_position[index]:>4}  Present:{pos:>4}")
            if abs(dxl_goal_position[index] - pos) > DXL_MOVING_STATUS_THRESHOLD:
                reached = False

        if reached:
            break

    index = 1 - index  # Toggle between min and max position

# ======================== SHUTDOWN ========================

groupSyncRead.clearParam()
for dxl_id in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_PRO_TORQUE_ENABLE, TORQUE_DISABLE)

portHandler.closePort()
print("🛑 Port closed and torque disabled on all actuators.")
