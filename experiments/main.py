import os
import setup.load_model_from_yaml as load_model_from_yaml
import runpy


# Load the model and create a simulation instance
#Input a yaml file to consume
robot_model_simulation = "salamander_and_fixed.yaml"

import sys

def main():
    if len(sys.argv) <  2:
        global robot_model_simulation
        print(f"Using yaml file {robot_model_simulation} specified in main.py: ")
        robot_model_path, terrain_model_path = load_model_from_yaml.find_model_files(robot_model_simulation)
        simulation_path = load_model_from_yaml.find_simulation_file(robot_model_simulation)
    else:
        robot_model_simulation = sys.argv[1]
        print(f"You passed in: {robot_model_simulation}")
        print(f"Using yaml file {robot_model_simulation} specified in main.py: ")
        robot_model_path, terrain_model_path = load_model_from_yaml.find_model_files(robot_model_simulation)
        simulation_path = load_model_from_yaml.find_simulation_file(robot_model_simulation)
    

    print(robot_model_path)
    print(terrain_model_path)
    print(simulation_path)
    
    runpy.run_path(simulation_path, init_globals={"yaml_path": robot_model_simulation})  

if __name__ == "__main__":
    print("Usage: python main.py <your_argument>")
    print("Usage: You may also specify robot_model_simulation: at the top of main.py and run as normal")
    print("Usage Example: python main.py turtle_and_fixed.yaml")
    print("Running main:")
    main()
