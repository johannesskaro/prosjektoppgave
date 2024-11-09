import open3d as o3d
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import json


def read_data_from_file():
    with open("files/stixel.json", "r") as f:
        data = json.load(f)
    stixel_3d_points = data['stixels']
    plane_params = data["plane_params"]
    plane_params = [plane_params[2], plane_params[0], - plane_params[1], - plane_params[3]]
    water_surface_polygon_points = data["water_surface_polygon_points"]
    return stixel_3d_points, water_surface_polygon_points, plane_params

def create_stixel(vertices, color):

    # Define two triangles to form the quadrilateral
    triangles = np.array([[0, 1, 2], [0, 2, 3]])  # Triangle indices for a quad

    # Create the mesh and assign vertices and triangles
    stixel = o3d.geometry.TriangleMesh()
    stixel.vertices = o3d.utility.Vector3dVector(vertices)
    stixel.triangles = o3d.utility.Vector3iVector(triangles)
    stixel.paint_uniform_color(color)
    
    # Optionally compute vertex normals for better visualization
    stixel.compute_vertex_normals()
    
    return stixel

def create_polygon(water_surface_polygon_points, plane_params, color):

    a, b, c, d = plane_params

    polygon_3d_points = []
    origin = np.array([0, 0])
    polygon_points = np.vstack([origin, water_surface_polygon_points])
    polygon_points = polygon_points[:, [1, 0]]

    for x, y in water_surface_polygon_points:
        z = (-a * x - b * y - d) / c  # Calculate z on the plane
        polygon_3d_points.append([x, y, z])

    # Convert vertices to numpy array and define triangles for Open3D
    vertices = np.array(polygon_3d_points)
    polygon_mesh = o3d.geometry.TriangleMesh()
    polygon_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    polygon_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])  # Assuming 4 points
    polygon_mesh.paint_uniform_color(color)
    return polygon_mesh

def create_back_wall(color, position, width=10, height=10):
    """
    Create a solid back wall at a specific position with a given color.
    """
    wall = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=0.1)
    wall.translate(position)
    wall.paint_uniform_color(color)
    return wall

def plot_scene(stixel_3d_points, water_surface_polygon_points, plane_params):
    plane_params = [plane_params[2], plane_params[0], - plane_params[1], - plane_params[3]]
    a, b, c, d = plane_params

    
    stixel_vertices = np.array(stixel_3d_points)
    stixel_vertices = stixel_vertices[:, :, [2, 0, 1]]  # Swap axes for better visualization
    stixel_vertices[:,:,2] = - stixel_vertices[:,:,2]   # Subtract the base plane distance from the z-coordinate

    depth_min = 0
    depth_max = 60
    norm = mcolors.Normalize(vmin=depth_min, vmax=depth_max)
    cmap = cm.summer   #viridis  #gist_rainbow  # Choose any color map you prefer (e.g., viridis, plasma, coolwarm)


    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

        # Create and add stixels
    for points in stixel_vertices:
        depth = points[0,0]
        color = cmap(norm(depth))
        color = color[:3]
        stixel = create_stixel(points, color)
        vis.add_geometry(stixel)

    # Create and add water surface polygon
    
    polygon_mesh = create_polygon(water_surface_polygon_points, plane_params, color=[0, 0, 1])
    vis.add_geometry(polygon_mesh)


    back_wall = create_back_wall(color=[1, 0.5, 0], position=[-5, 0, -5])
    vis.add_geometry(back_wall)


    # Set up the camera position for a fixed view
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])  # Center of the scene
    ctr.set_front([0.0, 0.0, -1.0])     # Camera front direction
    ctr.set_lookat([0, 0, 0])            # Center of the scene
    ctr.set_up([0.0, -1.0, 0.0])         # Up direction
    ctr.set_zoom(0.8)                    # Zoom level

    # Run the visualizer
    vis.run()
    vis.destroy_window()





if __name__ == '__main__':

    stixel_3d_points, water_surface_polygon_points, plane_params = read_data_from_file()

    plot_scene(stixel_3d_points, water_surface_polygon_points, plane_params)


