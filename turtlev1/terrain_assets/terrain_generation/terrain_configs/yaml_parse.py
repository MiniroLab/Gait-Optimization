import os
import yaml

# Load the configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract file names
src_file = config["src_file"]
terrain_file = config["terrain_file"]

# Construct full file paths
src_file_path = os.path.join(os.getcwd(), config["paths"]["template_dir"], src_file)
terrain_file_path = os.path.join(os.getcwd(), config["paths"]["saved_dir"], terrain_file)

# Extract parameters
params = config["parameters"]
n = params["n"]
size_range = params["size_range"]
scatter_range = params["scatter_range"]
height_range = params["height_range"]
quat_range = params["quat_range"]
rock_type = params["rock_type"]

# Print values to verify
print(f"Source File Path: {src_file_path}")
print(f"Terrain File Path: {terrain_file_path}")
print(f"Parameters: {params}")
