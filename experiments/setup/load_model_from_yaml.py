import time
import numpy as np
import mujoco
import mujoco.viewer
import os
import yaml
import shutil

current_file_dir = os.path.dirname(os.path.abspath(__file__))


def load_yaml(file_path):
    """Loads the YAML file and returns its contents."""

    file_path = os.path.join(current_file_dir, "../", "yamls", file_path)

    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None


def find_model_files(yaml_file):
    """Finds the robot and terrain model files based on the YAML file parameters."""
    # Load YAML file
    config = load_yaml(yaml_file)
    if not config:
        return

    robot_model = config.get("robot_model")
    terrain_model = config.get("terrain_model")

    if not robot_model or not terrain_model:
        print("Error: 'robot_model' or 'terrain_model' not specified in YAML file.")
        return

    # Define directories
    robot_models_dir = "robot_models"
    terrain_models_dir = "terrain_models"

    # Construct file paths
    robot_model_path = os.path.join(
        current_file_dir, "../", robot_models_dir, robot_model
    )
    terrain_model_path = os.path.join(
        current_file_dir, "../", terrain_models_dir, terrain_model
    )

    # Check if files exist
    robot_exists = os.path.isfile(robot_model_path)
    terrain_exists = os.path.isfile(terrain_model_path)

    # Output results
    if robot_exists and terrain_exists:
        print(f"Robot model found: {robot_model_path}")
        print(f"Terrain model found: {terrain_model_path}")
        return robot_model_path, terrain_model_path
    else:
        if not robot_exists:
            print(
                f"Error: Robot model '{robot_model}' not found in '{robot_models_dir}'"
            )
        if not terrain_exists:
            print(
                f"Error: Terrain model '{terrain_model}' not found in '{terrain_models_dir}'"
            )


def find_simulation_file(yaml_file):
    """Finds the simulation/algorithm to run files based on the YAML file parameters."""
    """The simulation file is the algorithm, ex, a CPG, you'd like to run"""
    # Load YAML file
    config = load_yaml(yaml_file)
    if not config:
        return

    simulation_file = config.get("simulation")

    if not simulation_file:
        print("Error: 'simulation' not specified in YAML file.")
        return

    # Define directories
    simulations_dir = "simulations"

    # Construct file paths
    simulation_path = os.path.join(
        current_file_dir, "..", simulations_dir, simulation_file
    )

    # Check if files exist
    simulation_exists = os.path.isfile(simulation_path)

    # Output results
    if simulation_exists:
        print(f"Simulation found: {simulation_path}")

        return simulation_path
    else:
        print(
            f"Error: Simulation model '{simulation_file}' not found in '{simulation_path}'"
        )


def get_model_and_data_from_yaml(yaml_file):
    robot_model_path, terrain_model_path = find_model_files(yaml_file)

    robot_filename_without_suffix = os.path.splitext(
        os.path.basename(robot_model_path)
    )[0]
    terrain_filename_without_suffix = os.path.splitext(
        os.path.basename(terrain_model_path)
    )[0]

    simulation_environment_filename = (
        f"{robot_filename_without_suffix}-{terrain_filename_without_suffix}.xml"
    )
    simulation_environment_path = os.path.join(
        current_file_dir,
        "..",
        "setup",
        "simulation_environment",
        simulation_environment_filename,
    )

    asset_filename_without_suffix = os.path.splitext(
        os.path.basename(robot_model_path)
    )[0]
    asset_path = os.path.join(
        current_file_dir, "..", "model_assets", asset_filename_without_suffix
    )

    # This is the actual data file that gets returned and in which our simulation is executed it is the output file.
    # it's a merge of the robot with the environment.
    simulation_environment_file_path = create_robot_terrain_simulation_xml(
        robot_model_path, terrain_model_path, asset_path, simulation_environment_path
    )

    model = mujoco.MjModel.from_xml_path(simulation_environment_file_path)
    data = mujoco.MjData(model)

    return model, data


import os
import xml.etree.ElementTree as ET


def create_robot_terrain_simulation_xml(
    robot_xml_path, terrain_path, asset_path, output_path
):
    """
    Modify a MuJoCo XML file to include an <include> tag and an <asset> section for meshes.

    Args:
        robot_xml_path (str): Path to the original MuJoCo XML file.
        terrain_path (str): Path to the terrain file to be included.
        asset_path (str): Directory containing includes and meshes
        output_path (str): Path to save the modified XML file.
    """

    mesh_dir = os.path.join(asset_path, "meshes")

    shutil.copytree(
        os.path.dirname(robot_xml_path),
        os.path.dirname(output_path),
        dirs_exist_ok=True,
    )

    print("ROBOT XML PATH:", robot_xml_path)
    print("TERRAIN PATH", terrain_path)
    print("OUTPUT PATH", output_path)

    # Parse the XML file
    tree = ET.parse(robot_xml_path)
    root = tree.getroot()

    # Ensure it's a MuJoCo XML file
    if root.tag != "mujoco":
        raise ValueError("Invalid MuJoCo XML file")

    # Add the <include> tag at the top
    include_tag = ET.Element("include", file=terrain_path)
    root.insert(0, include_tag)  # Insert at the beginning of <mujoco>

    # Add <asset> tag for meshes
    asset_tag = root.find("asset")
    if asset_tag is None:
        asset_tag = ET.SubElement(root, "asset")

    # Remove existing <mesh> tags only
    for mesh in list(asset_tag.findall("mesh")):
        asset_tag.remove(mesh)

    # Add all meshes in the specified directory
    for mesh_file in os.listdir(mesh_dir):
        if mesh_file.endswith((".stl", ".obj", ".dae", ".STL")):  # Common mesh formats
            ET.SubElement(
                asset_tag,
                "mesh",
                file=os.path.join(mesh_dir, mesh_file),
                name=os.path.splitext(mesh_file)[0],
            )

    # Save the modified XML
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
