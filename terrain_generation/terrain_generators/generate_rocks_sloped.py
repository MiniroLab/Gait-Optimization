import xml.etree.ElementTree as ET
import random
import os
import numpy as np

current_file_dir = os.path.dirname(os.path.abspath(__file__))

src_file = "template_terrain_fixed.xml" #This is the NAME of template file we build upon. file_path finds the path to said file
src_file_path = os.path.join(current_file_dir, "..", "template_terrains", src_file) #get directory to the file since open needs the full path

terrain_file = "sloped_terrain.xml" #This is the name of our created terrain. it gets sent to saved_terrains
terrain_file_path = os.path.join(current_file_dir, "..", "saved_terrains", terrain_file) #full path we save to

n = 500 #number of rocks to make
size_range = [0.005, 0.015] #Size
scatter_range = [-0.4, 0.4] #This is the grid area over which the rocks are scattered
height_range = [0, 0.015] #These are the heights at which the rocks are placed in
quat_range = [0, 1] #Rotation angles of the rocks
rock_type = "ellipsoid" #Type of rock shape. cubes => box, pebble => ellipsoid
slope_angle = -10



"""This works by finding the xml_file, which is assumed to have a 'rocky_terrain_body' tag, and begins appending a bunch of scattered boxes, which act like rocks, to said tag"""
def append_rocks(src_file, dst_file, n, size_range, scatter_range, height_range, quat_range=[0, 1], type="box", slope_angle=0):
    tree = ET.parse(src_file)
    root = tree.getroot()
    rocky_terrain_body = root.find(".//body[@name='rocky_terrain_body']")
    
    rocky_terrain_body.set("axisangle", f"0 1 0 {np.deg2rad(slope_angle)}")
    
    rocky_terrain_plane = ET.Element("geom", {
        "name": "rocky_terrain_plane",
        "size": f"{abs(scatter_range[0])} {scatter_range[1]} 0.05",
        "rgba": "0.1 0.1 0.1 1",
        "type": "plane"
    })
    
    if rocky_terrain_body is None:
        raise ValueError("Invalid MuJoCo XML: Missing rocky_terrain_body element.")

    rocky_terrain_body.append(rocky_terrain_plane)
        
    
    #<body name="rocky_terrain_body" axisangle="0 1 0 0.8">
    

    """Randomly generate a rock geom"""
    for i in range(n):
        size = [random.uniform(size_range[0], size_range[1]) for _ in range(3)] #size of the rock
        scatter_position = [random.uniform(scatter_range[0], scatter_range[1]) for _ in range(2)] #(x, y) position of the rock
        height_position = random.uniform(height_range[0], height_range[1]) #height of the rock
        
        quat = [random.uniform(quat_range[0], quat_range[1]) for _ in range(4)] #Generate quaternion
        quat = [q / sum(quat) for q in quat]  # and normalize quaternion

        rock = ET.Element("geom", {
            "name": f"rock{i+1}",
            "type": type,
            "size": f"{size[0]} {size[1]} {size[2]}",
            "pos": f"{scatter_position[0]} {scatter_position[1]} {height_position}",
            "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
            "rgba": "0.5 0.4 0.3 1"
        })
        
        rocky_terrain_body.append(rock)
    
    tree.write(dst_file)
    print(f"Appended {n} rocks to {src_file} and saved as {terrain_file}.")
    

    
if __name__ == "__main__":
        append_rocks(src_file_path, 
                     terrain_file_path, 
                     n, 
                     size_range, 
                     scatter_range, 
                     height_range, 
                     type=rock_type,
                     slope_angle=slope_angle)