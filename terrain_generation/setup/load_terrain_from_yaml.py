import time
import numpy as np
import mujoco
import mujoco.viewer
import os
import yaml


current_file_dir = os.path.dirname(os.path.abspath(__file__))

def load_yaml(file_path):
    """Loads the YAML file and returns its contents."""
    
    
    file_path = os.path.join(current_file_dir,"../",  "yamls", file_path)
    
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None

def get_terrain_parameters(yaml_file):
    """Finds the robot and terrain model files based on the YAML file parameters."""
    # Load YAML file
    config = load_yaml(yaml_file)
    if not config:
        return

    template_file = config.get("template_file")
    output_file = config.get("output_file")
    terrain_type = config.get("terrain_type")
    parameters = config.get("parameters")
    save_in_terrain_models = config.get("save_in_terrain_models")

    if not template_file or not output_file or not parameters:
        print("Error: 'template_file' or 'terrain_file' all parameters not specified in YAML file.")
        return
    
    if not terrain_type:
        terrain_type = "fixed"
        

    # Construct file paths
    template_file_path = os.path.join(current_file_dir,"..", "template_terrains", template_file)
    
    
    #if we want to save in terrain models instead. Else, just say no
    
    if save_in_terrain_models == True:
        terrain_file_path = os.path.join(current_file_dir, "..", "..", "experiments", "terrain_models", output_file)
        #terrain_file_path = os.path.join(current_file_dir, "..", "saved_terrains", output_file)
    else:
        terrain_file_path = os.path.join(current_file_dir, "..", "saved_terrains", output_file)
    print("Saving as:", terrain_file_path)
    #if

    # Check if files exist
    template_exists = os.path.isfile(template_file_path)

    # Output results
    if template_file:
        print(f"Template file found: {template_file}")
        return template_file_path, terrain_file_path, parameters, terrain_type
    else:
        if not template_exists:
            print(f"Error: Template file '{template_file}' not found in '{template_file_path}'")
