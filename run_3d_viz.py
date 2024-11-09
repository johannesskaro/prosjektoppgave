import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import json
from matplotlib.colors import LightSource


def read_data_from_file():
    with open("files/stixel.json", "r") as f:
        data = json.load(f)
    stixel_3d_points = data['stixels']
    plane_params = data["plane_params"]
    water_surface_polygon_points = data["water_surface_polygon_points"]
    return stixel_3d_points, water_surface_polygon_points, plane_params

def calculate_normal(vertices):
    """Calculate the normal vector of a polygon defined by vertices."""
    v1 = vertices[1] - vertices[0]
    v2 = vertices[3] - vertices[0]
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else normal


def apply_lighting(stixel, light1, light2, base_color):
    # Calculate the normal vector of the stixel
    normal = calculate_normal(stixel)
    base_color = base_color[:3] if len(base_color) == 4 else base_color   # Remove alpha channel if present

    light_vector1 = np.array([np.cos(np.radians(light1.azdeg)), np.sin(np.radians(light1.azdeg)), np.sin(np.radians(light1.altdeg))])
    light_vector2 = np.array([np.cos(np.radians(light2.azdeg)), np.sin(np.radians(light2.azdeg)), np.sin(np.radians(light2.altdeg))])


    intensity1 = np.clip(np.dot(normal, light_vector1), 0, 1)
    intensity2 = np.clip(np.dot(normal, light_vector2), 0, 1)

    intensity = (intensity1 + intensity2) / 2

    shaded_color = np.array(base_color) * intensity
    
    # Simulate shading using the normal vector as the elevation map
    # Convert the normal to a 2D elevation map for simplicity
    #shaded_color = light.shade_rgb(np.array(base_color).reshape(1, 1, 3), normal[2])
    return shaded_color  # Return the shaded color as an RGB tuple

def create_gap_rectangle(stixel1, stixel2):
    """Create a rectangle between two stixels to represent the gap."""
    # Define the vertices of the rectangle
    v1 = stixel1[1]
    v2 = stixel2[0]
    v3 = stixel2[3]
    v4 = stixel1[2]
    return [v1, v2, v3, v4]

def create_filling_stixels(stixels_3d_points):
    stixels_with_gaps = stixels_3d_points.copy().tolist()
    for n, stixel in enumerate(stixels_3d_points):
        if n == 0:
            continue
        if np.abs(stixel[0][0] - stixels_3d_points[n-1][1][0]) > 2.5: #Check if there is a too large gap between stixels
            continue
        gap_rectangle = create_gap_rectangle(stixels_3d_points[n-1], stixel)
        stixels_with_gaps.insert(n, gap_rectangle)

    return stixels_with_gaps

def set_stixels_base_to_zero(stixels_3d_points):

    stixel_vertices = np.array(stixels_3d_points)
    stixel_vertices = stixel_vertices[:, :, [2, 0, 1]]  # Swap axes for better visualization
    stixel_vertices[:,:,2] = - stixel_vertices[:,:,2] # z axis is inverted

    # Place the stixels on the ground plane
    for n, vertices in enumerate(stixel_vertices):
        stixel_vertices[n,0,2] = vertices[0,2] - vertices[3,2]
        stixel_vertices[n,1,2] = vertices[1,2] - vertices[2,2]
        stixel_vertices[n,2,2] = vertices[2,2] - vertices[2,2]
        stixel_vertices[n,3,2] = vertices[3,2] - vertices[3,2]

    return stixel_vertices


def plot_scene(ax, stixel_3d_points, water_surface_polygon_points, plane_params, camera_position):
    plane_params = [plane_params[2], plane_params[0], - plane_params[1],  plane_params[3]]
    a, b, c, d = plane_params
    normal = plane_params[:3]
    d = plane_params[3]
    normal_length = np.linalg.norm(normal)
    height = d / normal_length

    depth_min = 0
    depth_max = 60
    norm = mcolors.Normalize(vmin=depth_min, vmax=depth_max)
    cmap = cm.gist_earth  #gist_rainbow  # Choose any color map you prefer (e.g., viridis, plasma, coolwarm)
    ax.clear()

    # Create a LightSource object
    light1 = LightSource(azdeg=150, altdeg=45)  # Adjust azimuth and altitude for light direct
    light2 = LightSource(azdeg=210, altdeg=45)
    #light_direction = np.array([1, -1, -1])
    #light_direction = light_direction / np.linalg.norm(light_direction)

    polygon_3d_points = []
    origin = np.array([[0, 0]])
    polygon_points = np.vstack([origin, water_surface_polygon_points])
    polygon_points = polygon_points[:, [1, 0]]
    polygon_points = polygon_points[~np.isnan(polygon_points).any(axis=1)]

    for x, y in polygon_points:
        #z = (-a * x - b * y - d) / c
        z = 0
        polygon_3d_points.append([x, y, z])

    polygon_3d_points = np.array(polygon_3d_points)
    soft_blue = (0.6, 0.8, 1.0) 
    #shaded_soft_blue = light.shade_rgb(soft_blue, polygon_3d_points[:, 2])

    polygon = Poly3DCollection([polygon_3d_points], color=soft_blue, alpha=1)
    ax.add_collection3d(polygon)

    stixel_vertices = np.array(stixel_3d_points)
    stixel_vertices = set_stixels_base_to_zero(stixel_vertices)
    stixel_vertices = np.array(create_filling_stixels(stixel_vertices))

    # Plot each surface element
    for stixel in stixel_vertices:
        depth = (stixel[0,0] + stixel[1,0]) / 2
        normal = calculate_normal(stixel)
        color = cmap(norm(depth))
        shaded_color = apply_lighting(stixel, light1, light2, color)
        poly = Poly3DCollection([stixel], color=shaded_color)
        ax.add_collection3d(poly)

    # Plot the origin point in red to represent the boat's position
    ax.scatter(0, 0, -height, color='red', s=10, label="Boat Position")
 
    # Set camera position
    ax.view_init(elev=camera_position[0], azim=camera_position[1])

    camera_distance = 1  # Adjust to move closer or further from the object
    ax.dist = camera_distance  # Smaller values bring the camera closer

    # Set plot limits
    ax.set_xlim(20, 0)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-1, 5)

    ax.grid(False)
    ax._axis3don = False  # Hide the 3D axis lines


def animate(i):
    stixel_3d_points, water_surface_polygon_points, plane_params = read_data_from_file()

    # Define camera movement as elevation and azimuth changes
    azim = 0 + 3 * np.sin(np.radians(i * 2))  # Azimuth changes within a 10-15 degree range
    elev = 20 + 10 * np.sin(np.radians(i * 1))  # Elevation changes within a 13-17 degree range
    camera_position = (elev, azim)
    plot_scene(ax, stixel_3d_points, water_surface_polygon_points, plane_params, camera_position)

if __name__ == '__main__':
    dpi = 100
    fig = plt.figure(figsize=(1280/dpi, 720/dpi), dpi=dpi)
    #fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('persp', focal_length=0.2) 
    ani = animation.FuncAnimation(fig, animate, frames=360, interval=50)
    #ani.save('results/3d_animation_scen4_2.mp4', writer='ffmpeg')

    plt.show()