import pandas as pd

# Initialize an empty list to store DataFrames
all_data = []

# Loop through files lhs_simulation_results_sync_set_1.csv to lhs_simulation_results_sync_set_10.csv
for i in range(1, 6):
    file_name = rf"C:\Users\chike\Box\TurtleRobotExperiments\Sea_Turtle_Robot_AI_Powered_Simulations_Project\NnamdiFiles\mujocotest1\assets\Gait-Optimization\data\lhs_simulation_results_sync_set_{i}.csv"
    df = pd.read_csv(file_name)
    if 'Trial' in df.columns:
        df = df.drop(columns=['Trial'])  # Remove existing Trial column if present
    all_data.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(all_data, ignore_index=True)

# Add a new sequential 'Trial' column from 1 to total number of rows
combined_df.insert(0, 'Trial', range(1, len(combined_df) + 1))

# Save to a new CSV file
combined_df.to_csv("combined_lhs_simulation_results_1-5.csv", index=False)

print("Combined dataset saved as 'combined_lhs_simulation_results_1-5.csv' with sequential Trial numbers.")
