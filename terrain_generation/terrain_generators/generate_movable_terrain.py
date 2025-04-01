import xml.etree.ElementTree as ET
import random
import os


"""This works by finding the xml_file, which is assumed to have a 'rocky_terrain_body' tag, and begins appending a bunch of scattered boxes, which act like rocks."""


def append_rocks(
    src_file,
    terrain_file,
    n,
    size_range,
    scatter_range,
    height_range,
    quat_range=[0, 1],
    rock_type="box",
    mass_range=[0.00001, 0.0001],
    sliding_friction_range=[0.4, 0.8],
    torsional_friction_range=[0.001, 0.01],
    rolling_friction_range=[0.1, 0.3],
):
    tree = ET.parse(src_file)
    root = tree.getroot()
    worldbody = root.find(
        ".//worldbody"
    )  # We must append to worldbody for rocks to be movable

    if worldbody is None:
        raise ValueError("Invalid MuJoCo XML: Missing worldbody element.")

    """Randomly generate a rock body with a free joint"""
    for i in range(n):
        
        #Sample from the distributions
        size = [random.uniform(size_range[0], size_range[1]) for _ in range(3)] #(l, w ,h)
        scatter_position = [
            random.uniform(scatter_range[0], scatter_range[1]) for _ in range(2) 
        ] #(x, y)
        height_position = [
            random.uniform(height_range[0], height_range[1]) for _ in range(1)
        ] #(z), works with scatter to create position: (x, y, z)
        mass = [random.uniform(mass_range[0], mass_range[1]) for _ in range(1)] #mass, in kg

        quat = [random.uniform(quat_range[0], quat_range[1]) for _ in range(4)] #some rotation
        quat = [q / sum(quat) for q in quat]  # Normalize quaternion
        
        sliding_friction = random.uniform(sliding_friction_range[0], sliding_friction_range[1])
        torsional_friction = random.uniform(torsional_friction_range[0], torsional_friction_range[1])
        rolling_friction = random.uniform(rolling_friction_range[0], rolling_friction_range[1])

        rock_body = ET.Element(
            "body",
            {
                "name": f"rock{i+1}",
                "pos": f"{scatter_position[0]} {scatter_position[1]} {height_position[0]}",
            },
        )

        rock_geom = ET.Element(
            "geom",
            {
                "type": rock_type,
                "size": f"{size[0]} {size[1]} {size[2]}",
                "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
                "rgba": "0.5 0.4 0.3 1",
                "friction": "0.01 0.01 0.01",
                "solimp": "0.4 0.4 0.001"
            },
        )

        #joint = ET.Element("joint", {"type": "free"}) #limited="true" range="0 0"
        joint = ET.Element("joint", {"type": "free", "damping":"5", "armature": "0.01"}) #limited="true" range="0 0"

        inertial = ET.Element(
            "inertial",
            {
                "pos": f"{scatter_position[0]} {scatter_position[1]} {height_position[0]}",
                "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
                "mass": f"{mass[0]}",
                "diaginertia": "0.1 0.1 0.1",

            },
        )

        rock_body.append(joint)
        rock_body.append(rock_geom)
        rock_body.append(inertial)
        worldbody.append(rock_body)

    wall_size = abs(
        max(scatter_range)
    )  # we want walls to contain the scatter, so its the max of th scatter range

    wall_geom_plusx = ET.Element(
        "geom",
        {
            "name": "+x",
            "type": "plane",
            "size": f"{wall_size} {wall_size} 0.05",
            "zaxis": "1 0 0",
            "pos": f"-{wall_size} 0 -{wall_size}",
        },
    )

    wall_geom_minusx = ET.Element(
        "geom",
        {
            "name": "-x",
            "type": "plane",
            "size": f"{wall_size} {wall_size} 0.05",
            "zaxis": "-1 0 0",
            "pos": f"{wall_size} 0 -{wall_size}",
        },
    )

    wall_geom_plusy = ET.Element(
        "geom",
        {
            "name": "+y",
            "type": "plane",
            "size": f"{wall_size} {wall_size} 0.05",
            "zaxis": "0 1 0",
            "pos": f"0 -{wall_size} -{wall_size}",
        },
    )

    wall_geom_minusy = ET.Element(
        "geom",
        {
            "name": "-y",
            "type": "plane",
            "size": f"{wall_size} {wall_size} 0.05",
            "zaxis": "0 -1 0",
            "pos": f"0 {wall_size} -{wall_size}",
        },
    )

    worldbody.append(wall_geom_plusx)
    worldbody.append(wall_geom_minusx)
    worldbody.append(wall_geom_plusy)
    worldbody.append(wall_geom_minusy)

    tree.write(terrain_file)
    print(f"Appended {n} rocks to {terrain_file} and saved as {terrain_file}.")
