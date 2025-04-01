import pandas as pd
import numpy as np


# Load the CSV file to inspect its contents
csv_path = "flipper_angles_radians.csv"
df = pd.read_csv(csv_path)

# Display the first few rows to understand the structure
df.head()

# Rename columns for clarity based on actuator order
df.columns = [
    "pos_frontleftflipper", "pos_frontrightflipper", 
    "pos_backleft", "pos_backright", 
    "pos_frontlefthip", "pos_frontrighthip"
]

# Convert DataFrame to NumPy array for faster indexing
servo_data = df.to_numpy()

# Save processed data for use in the MuJoCo script
servo_data_path = "processed_servo_data.npy"
np.save(servo_data_path, servo_data)

# Display the first few rows with proper column names
import ace_tools as tools
tools.display_dataframe_to_user(name="Processed Servo Data", dataframe=df)
