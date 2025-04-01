import xml.etree.ElementTree as ET
import random
import os
from .utils.sampling import sample_scatter_with_coefficient as sample_scatter


current_file_dir = os.path.dirname(os.path.abspath(__file__))


src_file = "template_terrain_fixed.xml"  # This is the NAME of template file we build upon. file_path finds the path to said file
src_file_path = os.path.join(
    current_file_dir, "..", "template_terrains", src_file
)  # get directory to the file since open needs the full path

terrain_file = "fixed_terrain.xml"  # This is the name of our created terrain. it gets sent to saved_t
terrain_file_path = os.path.join(
    current_file_dir, "..", "saved_terrains", terrain_file
)  # full path we save to

n = 500  # number of rocks to make
size_range = [0.005, 0.015]  # Size
scatter_range = [-0.5, 0.5]  # This is the grid area over which the rocks are scattered
height_range = [0, 0.015]  # These are the heights at which the rocks are placed in
quat_range = [0, 1]  # Rotation angles of the rocks
rock_type = "ellipsoid"  # Type of rock shape. cubes => box, pebble => ellipsoid

"""This works by finding the xml_file, which is assumed to have a 'rocky_terrain_body' tag, and begins appending a bunch of scattered boxes, which act like rocks, to said tag"""


def append_rocks(
    src_file,
    dst_file,
    n,
    size_range,
    scatter_range,
    height_range,
    mass_range=[0.00001, 0.0001],
    sliding_friction_range=[0.4, 0.8],
    torsional_friction_range=[0.001, 0.01],
    rolling_friction_range=[0.1, 0.3],
    quat_range=[0, 1],
    rock_type="box",
):
    tree = ET.parse(src_file)
    root = tree.getroot()
    rocky_terrain_body = root.find(".//body[@name='rocky_terrain_body']")

    if rocky_terrain_body is None:
        raise ValueError("Invalid MuJoCo XML: Missing rocky_terrain_body element.")

    previous_scatter_positions = []

    """Randomly generate a rock geom"""
    for i in range(n):
        size = [
            random.uniform(size_range[0], size_range[1]) for _ in range(3)
        ]  # size of the rock
        """This is an optimization to decrease the number of rocks needed to cover the space"""
        """The commented out code below works perfectly ok as well"""
        scatter_position = sample_scatter(scatter_range, previous_scatter_positions, size_range[1]*2)
        
        
        if scatter_position == None:
            print(f"Space has been sufficiently covered by {len(previous_scatter_positions)} rocks")
            n = len(previous_scatter_positions)
            break
        else:
            previous_scatter_positions.append(scatter_position)
        
        #scatter_position = [
        #    random.uniform(scatter_range[0], scatter_range[1]) for _ in range(2)
        #]  # (x, y) position of the rock
        
        

        height_position = random.uniform(
            height_range[0], height_range[1]
        )  # height of the rock

        mass = [
            random.uniform(mass_range[0], mass_range[1]) for _ in range(1)
        ]  # mass, in kg

        quat = [
            random.uniform(quat_range[0], quat_range[1]) for _ in range(4)
        ]  # Generate quaternion
        quat = [q / sum(quat) for q in quat]  # and normalize quaternion

        sliding_friction = random.uniform(
            sliding_friction_range[0], sliding_friction_range[1]
        )
        torsional_friction = random.uniform(
            torsional_friction_range[0], torsional_friction_range[1]
        )
        rolling_friction = random.uniform(
            rolling_friction_range[0], rolling_friction_range[1]
        )

        rock = ET.Element(
            "geom",
            {
                "name": f"rock{i+1}",
                "type": rock_type,
                "size": f"{size[0]} {size[1]} {size[2]}",
                "pos": f"{scatter_position[0]} {scatter_position[1]} {height_position}",
                "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
                "rgba": "0.5 0.4 0.3 1",
                "friction": f"{sliding_friction} {torsional_friction} {rolling_friction}",
            },
        )

        rocky_terrain_body.append(rock)

    tree.write(dst_file)
    print(f"Appended {n} rocks to {src_file} and saved as {terrain_file}.")


if __name__ == "__main__":
    append_rocks(
        src_file_path,
        terrain_file_path,
        n,
        size_range,
        scatter_range,
        height_range,
        rock_type=rock_type,
    )
