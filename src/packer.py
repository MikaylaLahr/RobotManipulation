import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from shapely.geometry import MultiPoint
from shapely.affinity import translate as shapely_translate
import pybullet as p
import pybullet_data
import time
import random
import rospy
from vision_msgs.msg import Detection3DArray
from robot_manipulation.srv import GeneratePackingPlan, GeneratePackingPlanRequest, GeneratePackingPlanResponse
from robot_manipulation.msg import PackingPlan, PackItem
from geometry_msgs.msg import Pose, Point, Quaternion
from dataclasses import dataclass

# Constants
BIN_WIDTH1, BIN_DEPTH1, BIN_HEIGHT1 = 88, 127, 63
SMALL_Z_THRESHOLD1 = 0.9*63 #BIN_HEIGHT1/2

BIN_WIDTH2, BIN_DEPTH2, BIN_HEIGHT2 = 127, 127, 63
SMALL_Z_THRESHOLD2 = 0.9*63 #BIN_HEIGHT2/2

VOXEL_PITCH = 3.0
STEP_SIZE = int(VOXEL_PITCH)  # Each step should match pitch for clean grid movement

# Grid dimensions (voxel-based)
GRID_W1 = int(BIN_WIDTH1 / VOXEL_PITCH)
GRID_D1 = int(BIN_DEPTH1 / VOXEL_PITCH)
GRID_H1 = int(BIN_HEIGHT1 / VOXEL_PITCH)

GRID_W2 = int(BIN_WIDTH2 / VOXEL_PITCH)
GRID_D2 = int(BIN_DEPTH2 / VOXEL_PITCH)
GRID_H2 = int(BIN_HEIGHT2 / VOXEL_PITCH)

MESH_FOLDER = "/home/group3/interbotix_ws/src/robot_manipulation/resources/meshes"

# -------------------- Load & Voxelize --------------------

def load_and_voxelize_mesh(file_path, pitch=VOXEL_PITCH):
    mesh = trimesh.load_mesh(file_path, file_type='stl')
    mesh.apply_scale(1000)
    
    voxelized = mesh.voxelized(pitch=pitch).as_boxes()
    voxel_grid = mesh.voxelized(pitch=pitch)
    return {
        "original_mesh": mesh,
        "voxel_mesh": voxelized,
        "voxel_coords": voxel_grid.points,
        "id": os.path.basename(file_path),
        "volume": voxelized.volume,
    }

def rotate_mesh(mesh, axis, angle_degrees):
    """
    Rotates the mesh around the specified axis by a given angle (degrees).
    """
    angle_radians = np.radians(angle_degrees)
    if axis == 'x':
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, [1, 0, 0])
    elif axis == 'y':
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, [0, 1, 0])
    elif axis == 'z':
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, [0, 0, 1])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    mesh.apply_transform(rotation_matrix)
    return mesh

# -------------------- Sorting --------------------

# Sample function to generate rotations
def generate_rotations():
    """
    Returns a list of rotation matrices for 90-degree rotations along X, Y, and Z axes.
    """
    return [
        np.eye(3),  # No rotation
        trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3],
        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3],
        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])[:3, :3],
        trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])[:3, :3],
        trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])[:3, :3],
        trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])[:3, :3],
        trimesh.transformations.rotation_matrix(3 * np.pi / 2, [1, 0, 0])[:3, :3],  # 270 degrees along X
        trimesh.transformations.rotation_matrix(3 * np.pi / 2, [0, 1, 0])[:3, :3],  # 270 degrees along Y
        trimesh.transformations.rotation_matrix(3 * np.pi / 2, [0, 0, 1])[:3, :3],  # 270 degrees along Z
    ]

def get_rotation_label(rotation_matrix):
    """
    Get a label for the rotation matrix (e.g., x90, y90, etc.).
    """
    rotation_labels = {
        0: "No Rotation",
        1: "x90",
        2: "y90",
        3: "z90",
        4: "x180",
        5: "y180",
        6: "z180",
        7: "x270",
        8: "y270",
        9: "z270"
    }

    # This is a simple way to get a label; it assumes the rotation matrix is one of the pre-defined ones.
    # Modify this as needed to handle specific rotation matrices more accurately.
    for i, matrix in enumerate(generate_rotations()):
        if np.allclose(rotation_matrix, matrix):
            return rotation_labels.get(i, "Unknown Rotation")

    return "Unknown Rotation"

def compare_180_area_differences(area_array,bb_array):
    """
    Compare surface areas from an array of areas based on 180-degree opposite rotations.
    Assumes the array follows this index order:
    0: "No Rotation", 1: "x90", 2: "y90", 3: "z90",
    4: "x180", 5: "y180", 6: "z180",
    7: "x270", 8: "y270", 9: "z270"
    """
    label_map = {
        0: "No Rotation",
        1: "x90",
        2: "y90",
        3: "z90",
        4: "x180",
        5: "y180",
        6: "z180",
        7: "x270",
        8: "y270",
        9: "z270"
    }

    # Pairs of indices that are 180° apart
    rotation_pairs = [
        (1, 7),  # x90 vs x270
        (2, 8),  # y90 vs y270
        (3, 9),  # z90 vs z270
        (0, 4),  # No rotation vs x180 (could be useful too)
        (0, 5),  # No rotation vs y180
        (0, 6)   # No rotation vs z180
    ]

    print("\n🔁 180° Opposite Rotation Area Differences:")
    rot_order = []
    area_order = []
    bb_order = []
    skip = []
    zero_stop = 1
    for i, j in rotation_pairs:
        diff = abs(area_array[i] - area_array[j])
        if diff != 0:
            maxA = max(area_array[i],area_array[j])
            if maxA == area_array[i]:
                rot_order.append(i)
                area_order.append(area_array[i])
                bb_order.append(bb_array[i])
                skip.append(j)
            elif maxA == area_array[j]:
                rot_order.append(j)
                area_order.append(area_array[j])
                bb_order.append(bb_array[j])
                skip.append(i)
        else:
            if zero_stop != 0:
                rot_order.append(i)
                area_order.append(area_array[i])
                bb_order.append(bb_array[i])
                zero_stop = 0
            rot_order.append(j)
            area_order.append(area_array[j])
            bb_order.append(bb_array[j])
        
        print(f"  {label_map[i]} vs {label_map[j]} → Δ Area: {diff:.4f} ({area_array[i]:.4f} vs {area_array[j]:.4f})")

    print("rotation order",rot_order)
    print("area order",area_order)
    print("bb order",bb_order)
    print("skip",skip)
    # Get indices that would sort 'areas' in descending order
    rot_order = np.array(rot_order)
    area_order = np.array(area_order)
    bb_order = np.array(bb_order)
    skip = np.array(skip)

    sorted_indices = np.argsort(bb_order)
    sorted_bb = bb_order[sorted_indices]
    sorted_vox = area_order[sorted_indices]
    sorted_rotations = [rot_order[i] for i in sorted_indices]
    print("Sorted Vox:", sorted_vox)
    print("Sorted Box", sorted_bb)
    print("Sorted Labels:", sorted_rotations)

    return sorted_rotations

# Function to calculate the area of the bounding box on the XY plane
def calculate_bounding_box_area(min_x,max_x,min_y,max_y):
    """
    Calculate the area of the bounding box on the XY plane.
    This is done by getting the min/max coordinates of the bounding box,
    and then calculating the area as (max_x - min_x) * (max_y - min_y).
    """

    # Calculate the area of the bounding box on the XY plane
    area = (max_x - min_x) * (max_y - min_y)

    return area

# Function to compare bottom area after rotation
def compare_voxels_rotations(voxelized_mesh):
    """
    Compares the surface area in contact with the XY plane for different rotations.
    """
    rotations = generate_rotations()
    surface_areas = []
    bb_areas = []

    num_rotations = len(rotations)
    grid_size = int(np.ceil(np.sqrt(num_rotations)))  # Find the closest square grid size
    fig = plt.figure(figsize=(grid_size * 5, grid_size * 4))
    for idx, rotation_matrix in enumerate(rotations):
        ax = fig.add_subplot(grid_size, grid_size, idx + 1, projection='3d')  # 3x3 grid for 9 subplots

        # Apply the rotation to voxel coordinates
        rotated_voxels1 = np.dot(voxelized_mesh['voxel_coords'], rotation_matrix.T)
        rotated_voxels = np.round(rotated_voxels1 / VOXEL_PITCH).astype(int)

        # Get the bottom-most layer (lowest Z value)
        min_z = np.min(rotated_voxels[:, 2])
        bottom_voxels = rotated_voxels[rotated_voxels[:, 2] == min_z]

        # Calculate the area of the bottom surface
        area = len(bottom_voxels) * (VOXEL_PITCH ** 2)  # Area = number of bottom voxels * area of each voxel
        surface_areas.append(area)

        # Plot the rotated voxels in this subplot using bars
        color = np.random.rand(3,)  # Random color for each object

        min_x, max_x, min_y, max_y, min_z, max_z = get_bounding_box(rotated_voxels1)
        bb_area = calculate_bounding_box_area(min_x,max_x,min_y,max_y)
        bb_areas.append(bb_area)

    # Return surface areas for each rotation
    return surface_areas, bb_areas

# Function to get the bounding box of the entire object (after rotation)
def get_bounding_box(voxels):
    """
    Compute the bounding box of the entire voxelized object (after rotation).
    Returns the min and max coordinates for X, Y, and Z axes.
    """
    min_x, min_y, min_z = np.min(voxels, axis=0)
    max_x, max_y, max_z = np.max(voxels, axis=0)

    return min_x, max_x, min_y, max_y, min_z, max_z

def plot_bounding_box(ax, min_x, max_x, min_y, max_y, min_z, max_z):
    """
    Plot a bounding box for the given min/max coordinates.
    """
    # Define the 8 vertices of the bounding box
    vertices = [
        [min_x, min_y, min_z], [max_x, min_y, min_z], [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z], [max_x, max_y, max_z], [min_x, max_y, max_z]
    ]
    
    # Define the 12 edges connecting the vertices
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    # Plot the edges
    for edge in edges:
        x = [vertices[edge[0]][0], vertices[edge[1]][0]]
        y = [vertices[edge[0]][1], vertices[edge[1]][1]]
        z = [vertices[edge[0]][2], vertices[edge[1]][2]]
        ax.plot(x, y, z, color="r", lw=2)

# -------------------- Placement Helpers --------------------

def overlap(x, y, z, w, d, h, occupied):
    for ox, oy, oz, ow, od, oh in occupied:
        if (x < ox + ow and x + w > ox and
            y < oy + od and y + d > oy and
            z < oz + oh and z + h > oz):
            return True
    return False

def place_mesh_on_floor(mesh, occupied_area, mesh_width, mesh_depth, mesh_height):
    for x in range(0, BIN_WIDTH - int(mesh_width) + 1, STEP_SIZE):
        for y in range(0, BIN_DEPTH - int(mesh_depth) + 1, STEP_SIZE):
            z = 0  # Only placing on the floor initially
            if not overlap(x, y, z, mesh_width, mesh_depth, mesh_height, occupied_area):
                return True, (x, y, z)
    return False, None

def place_mesh_in_bin(mesh, occupied, mesh_width, mesh_depth, mesh_height):
    for z in range(0, BIN_HEIGHT - int(mesh_height) + 1, STEP_SIZE):
        for x in range(0, BIN_WIDTH - int(mesh_width) + 1, STEP_SIZE):
            for y in range(0, BIN_DEPTH - int(mesh_depth) + 1, STEP_SIZE):
                if not overlap(x, y, z, mesh_width, mesh_depth, mesh_height, occupied):
                    return True, (x, y, z)
    return False, None

def voxel_occupied(voxel_coords, translation, occupancy_grid):
    """
    Check if placing a mesh with its voxels at the given translation collides with occupied voxels.
    """
    for point in voxel_coords:
        grid_point = ((point + translation) / VOXEL_PITCH).astype(int)
        x, y, z = grid_point
        if (x < 0 or y < 0 or z < 0 or 
            x >= occupancy_grid.shape[0] or 
            y >= occupancy_grid.shape[1] or 
            z >= occupancy_grid.shape[2] or 
            occupancy_grid[x, y, z]):
            return True
    return False

def mark_voxels(voxel_coords, translation, occupancy_grid):
    """
    After placing a mesh, mark its voxels as occupied.
    """
    for point in voxel_coords:
        grid_point = ((point + translation) / VOXEL_PITCH).astype(int)
        x, y, z = grid_point
        if 0 <= x < occupancy_grid.shape[0] and 0 <= y < occupancy_grid.shape[1] and 0 <= z < occupancy_grid.shape[2]:
            occupancy_grid[x, y, z] = True

def has_voxel_support(voxel_coords, translation, occupancy_grid, support_ratio=0.7):
    """
    Check if at least `support_ratio` of the bottom-facing voxels are supported.
    """
    supported = 0
    bottom_voxels = []

    # Convert all translated voxel positions
    translated_voxels = (voxel_coords + translation) / VOXEL_PITCH
    translated_voxels = translated_voxels.astype(int)

    # Find lowest Z value (bottom layer)
    min_z = np.min(translated_voxels[:, 2])
    bottom_voxels = translated_voxels[translated_voxels[:, 2] == min_z]

    for x, y, z in bottom_voxels:
        below_z = z - 1
        if below_z < 0:
            supported += 1  # Ground level counts as support
        elif (0 <= x < occupancy_grid.shape[0] and
              0 <= y < occupancy_grid.shape[1] and
              occupancy_grid[x, y, below_z]):
            supported += 1

    support_fraction = supported / len(bottom_voxels) if bottom_voxels.any() else 0
    return support_fraction >= support_ratio

def get_footprint_polygon(mesh):
    base_z = mesh.bounds[0][2] + 1e-5
    slice = mesh.section(plane_origin=[0, 0, base_z], plane_normal=[0, 0, 1])
    if slice is None or len(slice.vertices) == 0:
        return None
    base_points = slice.vertices[:, :2]
    return MultiPoint(base_points).convex_hull

# -------------------- Packing --------------------

def pack_meshes(items,GRID_W,GRID_D,GRID_H,BIN_WIDTH,BIN_DEPTH,BIN_HEIGHT,SMALL_Z_THRESHOLD):
    placements = []
    occupancy_grid = np.zeros((GRID_W, GRID_D, GRID_H), dtype=bool)
    rotations = generate_rotations()
    print("rotations",rotations)

    unplaced_meshes = []
    unpacked_meshes = []
    for item in items:
        mesh_info = item.mesh_info
        object_id = item.id

        print("MESH INFO:",mesh_info)
        original_mesh = mesh_info["original_mesh"]
        mesh_id = mesh_info["id"]
        sorted_rotations = mesh_info["rotations"]
        placed = False

        for rot_idx, R in enumerate(rotations):
            print("rot_idx",rot_idx)
            if rot_idx not in sorted_rotations:
                continue 

            # Apply rotation
            rotated_mesh = original_mesh.copy()
            rotated_mesh.apply_transform(np.vstack([np.hstack([R, [[0], [0], [0]]]), [0, 0, 0, 1]]))
            
            # Voxelize rotated mesh
            voxel_grid = rotated_mesh.voxelized(pitch=VOXEL_PITCH)
            voxel_mesh = voxel_grid.as_boxes()
            voxel_coords = voxel_grid.points

            mesh_width = voxel_mesh.bounds[1][0] - voxel_mesh.bounds[0][0]
            mesh_depth = voxel_mesh.bounds[1][1] - voxel_mesh.bounds[0][1]
            mesh_height = voxel_mesh.bounds[1][2] - voxel_mesh.bounds[0][2]

            for z in range(0, BIN_HEIGHT - int(mesh_height) + 1, STEP_SIZE):
                for x in range(0, BIN_WIDTH - int(mesh_width) + 1, STEP_SIZE):
                    for y in range(0, BIN_DEPTH - int(mesh_depth) + 1, STEP_SIZE):
                        translation = np.array([x, y, z]) - voxel_mesh.bounds[0]

                        if voxel_occupied(voxel_coords, translation, occupancy_grid):
                            continue

                        mesh_bottom_z = z
                        mesh_top_z = z + mesh_height
                        mesh_center_z = z + mesh_height / 2

                        if z == 0 and mesh_center_z <= BIN_HEIGHT and mesh_top_z <= 2*BIN_HEIGHT:
                            valid_placement = True
                        elif mesh_center_z <= BIN_HEIGHT and mesh_top_z <= 2*BIN_HEIGHT and has_voxel_support(voxel_coords, translation, occupancy_grid): #z < SMALL_Z_THRESHOLD and has_voxel_support(voxel_coords, translation, occupancy_grid):
                            valid_placement = True
                        else:
                            valid_placement = False

                        if not valid_placement:
                            continue

                        #if z == 0 or has_voxel_support(voxel_coords, translation, occupancy_grid):
                        placements.append({
                            "mesh": voxel_mesh,
                            "original_mesh": rotated_mesh,
                            "voxel_coords": voxel_coords,
                            "translation": tuple(translation),
                            "width": mesh_width,
                            "depth": mesh_depth,
                            "height": mesh_height,
                            "id": f"{mesh_id}_rot{rot_idx}",
                            "rotation_idx": rot_idx,
                            "x": x,
                            "y": y,
                            "z": z,
                            "object_id": object_id
                        })
                        mark_voxels(voxel_coords, translation, occupancy_grid)
                        placed = True
                        break
                    if placed: break
                if placed: break
            if placed:
                break

        if not placed:
            print(f"🔁 Deferring mesh (no support placement): {mesh_id}")
            unplaced_meshes.append(item)

    # --- Second Pass: Place remaining without support, minimizing z ---
    for item in unplaced_meshes:
        mesh_info = item.mesh_info
        original_mesh = mesh_info["original_mesh"]
        mesh_id = mesh_info["id"]
        sorted_rotations = mesh_info["rotations"]
        best_z = BIN_HEIGHT + 1
        best_placement = None

        for rot_idx, R in enumerate(rotations):
            if rot_idx not in sorted_rotations:
                continue

            rotated_mesh = original_mesh.copy()
            rotated_mesh.apply_transform(np.vstack([np.hstack([R, [[0], [0], [0]]]), [0, 0, 0, 1]]))
            
            voxel_grid = rotated_mesh.voxelized(pitch=VOXEL_PITCH)
            voxel_mesh = voxel_grid.as_boxes()
            voxel_coords = voxel_grid.points

            mesh_width = voxel_mesh.bounds[1][0] - voxel_mesh.bounds[0][0]
            mesh_depth = voxel_mesh.bounds[1][1] - voxel_mesh.bounds[0][1]
            mesh_height = voxel_mesh.bounds[1][2] - voxel_mesh.bounds[0][2]

            for z in range(0, BIN_HEIGHT - int(mesh_height) + 1, STEP_SIZE):
                for x in range(0, BIN_WIDTH - int(mesh_width) + 1, STEP_SIZE):
                    for y in range(0, BIN_DEPTH - int(mesh_depth) + 1, STEP_SIZE):
                        translation = np.array([x, y, z]) - voxel_mesh.bounds[0]

                        if voxel_occupied(voxel_coords, translation, occupancy_grid):
                            continue

                        if z < best_z and z < SMALL_Z_THRESHOLD:
                            best_z = z
                            best_placement = {
                                "mesh": voxel_mesh,
                                "original_mesh": rotated_mesh,
                                "voxel_coords": voxel_coords,
                                "translation": tuple(translation),
                                "width": mesh_width,
                                "depth": mesh_depth,
                                "height": mesh_height,
                                "id": f"{mesh_id}_rot{rot_idx}",
                                "rotation_idx": rot_idx,
                                "x": x,
                                "y": y,
                                "z": z,
                                "object_id": item.id,
                            }

        if best_placement:
            placements.append(best_placement)
            mark_voxels(best_placement["voxel_coords"], np.array(best_placement["translation"]), occupancy_grid)
            print(f"✅ Placed mesh without support: {mesh_id}")
        else:
            unpacked_meshes.append(mesh_info)
            print(f"❌ Could not place mesh at all: {mesh_id}")

    return placements, occupancy_grid, unpacked_meshes

# -------------------- Plotting --------------------

def plot_occupancy_grid_per_object(placements, BIN_WIDTH, BIN_DEPTH, BIN_HEIGHT, SMALL_Z_THRESHOLD, pitch=VOXEL_PITCH, title="Voxel Grid per Object"):
    """
    Plot the voxelized mesh for each object using its own color.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for placement in placements:
        voxel_coords = placement.get("voxel_coords")
        translation = np.array(placement["translation"])
        color = np.random.rand(3,)  # random color per object

        # Translate voxel coordinates
        translated_voxels = voxel_coords + translation

        # Draw each voxel as a bar3d cube
        for point in translated_voxels:
            x, y, z = point
            ax.bar3d(x, y, z, pitch, pitch, pitch, color=color, alpha=0.4, edgecolor='k', linewidth=0.3)

    
        # Plot bin frame (wireframe box)
    def draw_bin(ax, width, depth, height, SMALL_Z_THRESHOLD):
        # Define corners of the bin
        corners = np.array([
            [0, 0, 0],
            [width, 0, 0],
            [width, depth, 0],
            [0, depth, 0],
            [0, 0, height],
            [width, 0, height],
            [width, depth, height],
            [0, depth, height]
        ])
        
        # Define the 12 edges of the box using index pairs
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        for start, end in edges:
            ax.plot(*zip(corners[start], corners[end]), color='black', linewidth=1.0)

        # Define middle band corners (same XY as bottom but Z=SMALL_Z_THRESHOLD)
        band_corners = np.array([
            [0, 0, SMALL_Z_THRESHOLD],
            [width, 0, SMALL_Z_THRESHOLD],
            [width, depth, SMALL_Z_THRESHOLD],
            [0, depth, SMALL_Z_THRESHOLD]
        ])

        # Draw the wrapping line (a rectangle)
        for i in range(4):
            start = band_corners[i]
            end = band_corners[(i + 1) % 4]
            ax.plot(*zip(start, end), color='red', linestyle='--', linewidth=1.0)

    # Draw the bin
    draw_bin(ax, BIN_WIDTH, BIN_DEPTH, BIN_HEIGHT, SMALL_Z_THRESHOLD)

    ax.set_xlim(0, BIN_HEIGHT)
    ax.set_ylim(0, BIN_HEIGHT)
    ax.set_zlim(0, BIN_HEIGHT)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

def plot_original_meshes(placements, BIN_WIDTH, BIN_DEPTH, BIN_HEIGHT, SMALL_Z_THRESHOLD, title="Original Meshes in Bin"):
    """
    Plot the original (non-voxelized) meshes in their placed positions.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for placement in placements:
        mesh = placement["original_mesh"]
        translation = np.array(placement["translation"])
        color = np.random.rand(3,)

        # Apply translation to vertices
        translated_vertices = mesh.vertices + translation
        faces = mesh.faces
        face_triangles = translated_vertices[faces]

        # Plot the faces
        mesh_collection = Poly3DCollection(face_triangles, facecolors=color, edgecolors='k', linewidths=0.3, alpha=0.6)
        ax.add_collection3d(mesh_collection)

        # Plot bin frame (wireframe box)
    def draw_bin(ax, width, depth, height, SMALL_Z_THRESHOLD):
        # Define corners of the bin
        corners = np.array([
            [0, 0, 0],
            [width, 0, 0],
            [width, depth, 0],
            [0, depth, 0],
            [0, 0, height],
            [width, 0, height],
            [width, depth, height],
            [0, depth, height]
        ])
        
        # Define the 12 edges of the box using index pairs
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        for start, end in edges:
            ax.plot(*zip(corners[start], corners[end]), color='black', linewidth=1.0)

        # Define middle band corners (same XY as bottom but Z=SMALL_Z_THRESHOLD)
        band_corners = np.array([
            [0, 0, SMALL_Z_THRESHOLD],
            [width, 0, SMALL_Z_THRESHOLD],
            [width, depth, SMALL_Z_THRESHOLD],
            [0, depth, SMALL_Z_THRESHOLD]
        ])

        # Draw the wrapping line (a rectangle)
        for i in range(4):
            start = band_corners[i]
            end = band_corners[(i + 1) % 4]
            ax.plot(*zip(start, end), color='red', linestyle='--', linewidth=1.0)

    # Draw the bin
    draw_bin(ax, BIN_WIDTH, BIN_DEPTH, BIN_HEIGHT, SMALL_Z_THRESHOLD)

    ax.set_xlim(0, BIN_HEIGHT)
    ax.set_ylim(0, BIN_HEIGHT)
    ax.set_zlim(0, BIN_HEIGHT)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

# -------------------- Main --------------------
@dataclass
class Item:
    id: int
    type: str
    mesh_info: any

type_to_mesh = {}
type_int_to_str = {
    0: "can",
    1: "cube",
    2: "eggs",
    3: "milk",
    4: "toilet_paper",
    5: "wine"
}

def pack_no_weights(items):
    items = sorted(items, key=lambda m: m.mesh_info["voxel_mesh"].volume if m.mesh_info["voxel_mesh"].volume is not None else 0, reverse=True)
    print(items)

    rotations = generate_rotations()

    all_placements = []

    small_placements, small_occupancy_grid, unpacked_meshes = pack_meshes(items,GRID_W1,GRID_D1,GRID_H1,BIN_WIDTH1,BIN_DEPTH1,BIN_HEIGHT1,SMALL_Z_THRESHOLD1)
    print("PLACEMENTS SMALL:",small_placements)
    if len(small_placements) == len(items):
        print("✅ All objects fit in the SMALL bin!")
        all_placements = [('small', p) for p in small_placements]
    else:
        print("🔁 Not all objects fit in the small bin. Using LARGE bin...")
        large_placements, large_occupancy_grid, unpacked_meshes = pack_meshes(items,GRID_W2,GRID_D2,GRID_H2,BIN_WIDTH2,BIN_DEPTH2,BIN_HEIGHT2,SMALL_Z_THRESHOLD2)
        print("PLACEMENTS LARGE:",large_placements)
        if len(large_placements) == len(items):
            print("✅ All objects fit in the LARGE bin!")
            all_placements = [('large', p) for p in large_placements]
        else:
            print("Objects need to be split into two bins")
            small_placements, small_occupancy_grid, unpacked_meshes = pack_meshes(unpacked_meshes,GRID_W1,GRID_D1,GRID_H1,BIN_WIDTH1,BIN_DEPTH1,BIN_HEIGHT1,SMALL_Z_THRESHOLD1)
            print("PLACEMENTS SMALL:",small_placements)
            all_placements = [('large', p) for p in large_placements] + [('small', p) for p in small_placements]

    
    result = []
    for box, placement in all_placements:
        width = BIN_WIDTH2 if box == 'large' else BIN_WIDTH1
        depth = BIN_DEPTH2 if box == 'large' else BIN_DEPTH1
        height = BIN_HEIGHT2 if box == 'large' else BIN_HEIGHT1

        # convert back to meters
        x = (placement["translation"][0] - width / 2) / 1000
        y = (placement["translation"][1] - depth / 2) / 1000
        z = (placement["translation"][2] - height / 2) / 1000
        rot_idx = placement["rotation_idx"]
        item_id = placement["object_id"]
        quat = trimesh.transformations.quaternion_from_matrix(rotations[rot_idx])
        pose = Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0]))
        result.append(PackItem(item_id=item_id, box_id=(1 if box == 'large' else 0), end_pose=pose))

    return result


def load_meshes():
    stl_files = [os.path.join(MESH_FOLDER, f) for f in os.listdir(MESH_FOLDER) if f.endswith('.stl')]
    for fp in stl_files:
        mesh = load_and_voxelize_mesh(fp)
        voxel_areas, bb_areas = compare_voxels_rotations(mesh)
        sorted_rotations = compare_180_area_differences(voxel_areas,bb_areas)
        mesh["rotations"] = sorted_rotations
        file_name, _ = os.path.splitext(os.path.basename(fp))
        type_to_mesh[file_name] = mesh
        

def generate_packing_plan(req):
    '''
    Service callback function to process Detection3DArray and return a custom response.
    'req' is an object of type ProcessDetectionsRequest, which contains 'detections_input'.
    '''
    rospy.loginfo("Received %d 3D detections to process.", len(req.detections.detections))

    # Initialize your custom response
    response = GeneratePackingPlanResponse()

    items = [Item(det.results[0].id, type_int_to_str[det.results[0].id // 10000], type_to_mesh[type_int_to_str[det.results[0].id // 10000]]) for det in req.detections.detections]
    result = pack_no_weights(items)

    response.plan.items = result

    rospy.loginfo("Processing complete. Sending response.")
    return response

def main():
    '''
    Initializes the ROS node and advertises the service.
    '''
    load_meshes()
    rospy.init_node('packer')
    service_name = 'generate_packing_plan'
    rospy.Service(service_name, GeneratePackingPlan, generate_packing_plan)
    rospy.loginfo("Service '%s' is ready.", service_name)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
