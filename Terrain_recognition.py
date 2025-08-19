from pynput.mouse import Button, Controller
import time

mouse = Controller()

while True:
    mouse.click(Button.left, 1)  # Left click
    print("Clicked left mouse button")
    time.sleep(3*60)  # Add a small delay to avoid excessive clicking
    


