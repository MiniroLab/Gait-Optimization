#This python script takesimport os
import setup.load_terrain_from_yaml as load_terrain_from_yaml
import runpy
import os
import terrain_generators.generate_movable_terrain as generate_movable_terrain
import terrain_generators.generate_fixed_terrain as generate_fixed_terrain
import sys


# Load the model and create a simulation instance
#Input a yaml file to consume
terrain_yaml = "boulder_terrain.yaml"



def main():
    if len(sys.argv) <  2:
        global terrain_yaml        
        template_file_path, output_file_path, parameters = load_terrain_from_yaml.get_terrain_parameters(terrain_yaml)
        print(f"Using yaml file {template_file_path} specified in main.py: ")
        print(f"With these parameters: {parameters.items()}")
        print(f"output is in: {output_file_path}")
        #generate_movable_terrain.append_rocks(template_file_path, output_file_path, **parameters)
        generate_fixed_terrain.append_rocks(template_file_path, output_file_path, **parameters)

    """else:
        robot_model_simulation = sys.argv[1]
        print(f"You passed in: {robot_model_simulation}")
        print(f"Using yaml file {robot_model_simulation} specified in main.py: ")
        robot_model_path, terrain_model_path = load_terrain_from_yaml.find_model_files(robot_model_simulation)
        simulation_path = load_terrain_from_yaml.find_simulation_file(robot_model_simulation)
    """

  

if __name__ == "__main__":
    print("Usage: python main.py <your_argument>")
    print("Usage: You may also specify terrain_yaml: at the top of main.py and run as normal")
    print("Running main:")
    main()
