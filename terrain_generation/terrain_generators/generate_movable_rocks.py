import xml.etree.ElementTree as ET
import random
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))

src_file = "template_terrain_fixed.xml" #This is the NAME of template file we build upon. file_path finds the path to said file
src_file_path = os.path.join(current_file_dir, "..", "template_terrains", src_file) #get directory to the file since open needs the full path

terrain_file = "movable_terrain.xml" #This is the name of our created terrain. it gets sent to saved_t
terrain_file_path = os.path.join(current_file_dir, "..", "saved_terrains", terrain_file) #full path we save to


n = 300 #number
size_range = [0.005, 0.015] #Size
scatter_range = [-0.3, 0.3] #This is a 2d box in which the rocks are scattered
height_range = [0.3, 0.35] #This is the height at which the rocks are scattered
mass_range = [0.001, 0.01] #Movable rocks require mass t
quat_range = [0, 1] #Rotation angles
rock_type = "ellipsoid" #Type of rock shape. cubes => box, pebble => ellipsoid



"""This works by finding the xml_file, which is assumed to have a 'rocky_terrain_body' tag, and begins appending a bunch of scattered boxes, which act like rocks."""
def append_rocks(src_file, terrain_file_path, n, size_range, scatter_range, height_range, quat_range=[0, 1], rock_type="box", mass_range=[0.0001, 0.0001]):
    tree = ET.parse(src_file)
    root = tree.getroot()
    worldbody = root.find(".//worldbody") #We must append to worldbody for rocks to be movable
    
    if worldbody is None:
        raise ValueError("Invalid MuJoCo XML: Missing worldbody element.")
    
    """Randomly generate a rock body with a free joint"""
    for i in range(n):
        size = [random.uniform(size_range[0], size_range[1]) for _ in range(3)]
        scatter_position = [random.uniform(scatter_range[0], scatter_range[1]) for _ in range(2)]
        height_position = [random.uniform(height_range[0], height_range[1]) for _ in range(1)]
        mass = [random.uniform(mass_range[0], mass_range[1]) for _ in range(1)]
        
        quat = [random.uniform(quat_range[0], quat_range[1]) for _ in range(4)]
        quat = [q / sum(quat) for q in quat]  # Normalize quaternion

        rock_body = ET.Element("body", {"name": f"rock{i+1}", "pos": f"{scatter_position[0]} {scatter_position[1]} {height_position[0]}"})
        
        rock_geom = ET.Element("geom", {
            "type": rock_type,
            "size": f"{size[0]} {size[1]} {size[2]}",
            "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
            "rgba": "0.5 0.4 0.3 1",
        })     
        
        joint = ET.Element("joint", {
            "type": "free"
        })
        
        inertial = ET.Element("inertial", {"pos": f"{scatter_position[0]} {scatter_position[1]} {height_position[0]}", "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}", "mass": f"{mass[0]}"})


        
        rock_body.append(joint)
        rock_body.append(rock_geom)
        rock_body.append(inertial)
        worldbody.append(rock_body)

    wall_size = abs(max(scatter_range)) #we want walls to contain the scatter, so its the max of th scatter range
    
    wall_geom_plusx = ET.Element(
        "geom", {
            "name": "+x",
            "type": "plane",
            "size": f"{wall_size} {wall_size} 0.05",
            "zaxis": "1 0 0",
            "pos": f"-{wall_size} 0 -{wall_size}"
        }
    )
    

    wall_geom_minusx = ET.Element(
        "geom", {
            "name": "-x",
            "type": "plane",
            "size": f"{wall_size} {wall_size} 0.05",
            "zaxis": "-1 0 0",
            "pos": f"{wall_size} 0 -{wall_size}"
        }
    )
    

    wall_geom_plusy = ET.Element(
        "geom", {
            "name": "+y",
            "type": "plane",
            "size": f"{wall_size} {wall_size} 0.05",
            "zaxis": "0 1 0",
            "pos": f"0 -{wall_size} -{wall_size}"
        }
    )
    

    wall_geom_minusy = ET.Element(
        "geom", {
            "name": "-y",
            "type": "plane",
            "size": f"{wall_size} {wall_size} 0.05",
            "zaxis": "0 -1 0",
            "pos": f"0 {wall_size} -{wall_size}"
        }
    )

    worldbody.append(wall_geom_plusx)
    worldbody.append(wall_geom_minusx)
    worldbody.append(wall_geom_plusy)
    worldbody.append(wall_geom_minusy)
    
    tree.write(terrain_file_path)
    print(f"Appended {n} rocks to {terrain_file_path} and saved as {terrain_file}.")
    
    
if __name__ == "__main__":
        
        append_rocks(src_file_path, 
                     terrain_file_path, 
                     n, 
                     size_range, 
                     scatter_range, 
                     height_range,
                     quat_range=quat_range, 
                     rock_type=rock_type)
