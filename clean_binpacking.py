import pybullet as p
import pybullet_data
import os
import time
import random
import trimesh 
import numpy as np
import math
from stl import mesh
import open3d as o3d
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from PIL import Image
from trimesh.creation import box




# === Bin Configuration ===
#BIN_SIZE = 1.27
#BIN_LENGTH = 1.27  # along X-axis
#BIN_WIDTH = 0.88   # along Y-axis
WALL_THICKNESS = 0.02
WALL_HEIGHT = 0.63
MAX_HEIGHT = 2
STEP = 0.1
Z_STEP = 0.5 #25
MESH_SCALE_DEFAULT = 10 #0.01  # Default uniform scale for meshes
MARGIN = 0.001
SMALL_Z_THRESHOLD = 0.63*1.5


# Define candidate Euler angle rotations
ROTATION_CANDIDATES = [
    [0, 0, 0],
    [0, 0, math.pi / 2],
    [0, math.pi / 2, 0],
    [math.pi / 2, 0, 0],
    [0, math.pi / 2, math.pi / 2],
    [math.pi / 2, 0, math.pi / 2],
    [math.pi / 2, math.pi / 2, 0],
]

def euler_to_rotation_matrix(euler):
    """Convert Euler angles to a rotation matrix."""
    x, y, z = euler
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(x), -math.sin(x)],
                   [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0],
                   [math.sin(z), math.cos(z), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def get_xy_surface_area(bounds):
    dims = np.array(bounds[1]) - np.array(bounds[0])
    return dims[0] * dims[1], dims

def is_stable(dims):
    """Check if object is stable: height not too large relative to base."""
    height = dims[2]
    return height <= max(dims[0], dims[1]) * 1.5

def preferred_rotations(mesh_path):
    """Return sorted list of stable rotations minimizing XY surface area."""
    preferred = []
    surface_areas = []

    obj = mesh.Mesh.from_file(mesh_path)
    original_vectors = obj.vectors.reshape(-1, 3)

    for euler in ROTATION_CANDIDATES:
        R = euler_to_rotation_matrix(euler)
        rotated_vectors = np.dot(original_vectors, R.T)
        min_corner = rotated_vectors.min(axis=0)
        max_corner = rotated_vectors.max(axis=0)
        area, dims = get_xy_surface_area([min_corner, max_corner])

        if is_stable(dims):
            surface_areas.append((area, euler))

    surface_areas.sort(key=lambda x: x[0])
    return [euler for area, euler in surface_areas]


def rotate_mesh(original_mesh, rotation_matrix):
    new_mesh = mesh.Mesh(np.copy(original_mesh.data))
    for i in range(len(new_mesh.vectors)):
        new_mesh.vectors[i] = np.dot(new_mesh.vectors[i], rotation_matrix.T)
    return new_mesh



def euler_to_matrix(euler):
    return trimesh.transformations.euler_matrix(*euler, axes='sxyz')

def pybullet_stability_with_viz(stl_path, mesh_scale=10.0, settle_time=2.0):
    """Visualize and check if a mesh is stable in PyBullet."""
    mesh = trimesh.load_mesh(stl_path)
    mesh.apply_scale(mesh_scale)

    # Save the mesh as a temporary .obj file (PyBullet needs .obj for visual/col shapes)
    temp_obj_path = "/tmp/temp_mesh.obj"
    mesh.export(temp_obj_path)

    # Start PyBullet in GUI mode
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=temp_obj_path,
        meshScale=[1, 1, 1],
        rgbaColor=[0.5, 0.5, 0.9, 1]
    )
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=temp_obj_path,
        meshScale=[1, 1, 1]
    )

    if visual_shape_id < 0 or collision_shape_id < 0:
        print("❌ Failed to create shapes for", stl_path)
        p.disconnect()
        return False

    body_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0, 0, 0.5]
    )

    # Let it settle
    for _ in range(int(settle_time * 240)):
        p.stepSimulation()
        #time.sleep(1 / 240.0)

    final_pos, _ = p.getBasePositionAndOrientation(body_id)
    stable = final_pos[2] < 1.05

    #print(f"Stability result for {os.path.basename(stl_path)}: {'Stable' if stable else 'Unstable'}")
    time.sleep(2)  # Pause to view final position before disconnecting
    p.disconnect()
    return stable


VOXEL_PITCH = 3
ROTATION_CANDIDATES = [
    [0, 0, 0],  # No rotation

    # 90-degree rotations
    [np.pi / 2, 0, 0],  # 90° rotation around X-axis
    [0, np.pi / 2, 0],  # 90° rotation around Y-axis
    #[0, 0, np.pi / 2],  # 90° rotation around Z-axis

    # 180-degree rotations
    [np.pi, 0, 0],  # 180° rotation around X-axis
    [0, np.pi, 0],  # 180° rotation around Y-axis
    #[0, 0, np.pi],  # 180° rotation around Z-axis

    # 270-degree rotations
    [3 * np.pi / 2, 0, 0],  # 270° rotation around X-axis
    [0, 3 * np.pi / 2, 0],  # 270° rotation around Y-axis
    #[0, 0, 3 * np.pi / 2],  # 270° rotation around Z-axis
]


def euler_to_matrix(euler):
    return trimesh.transformations.euler_matrix(*euler, axes='sxyz')


def quaternion_diff(q1, q2):
    """Returns the angular difference (in radians) between two quaternions."""
    # Convert both to rotation matrices
    R1 = np.array(p.getMatrixFromQuaternion(q1)).reshape(3, 3)
    R2 = np.array(p.getMatrixFromQuaternion(q2)).reshape(3, 3)
    
    # Relative rotation matrix
    R_rel = R2 @ R1.T
    angle = np.arccos((np.trace(R_rel) - 1) / 2.0)
    return angle


def pybullet_stability_check_rotations(stl_path, mesh_scale=10.0, settle_time=5.0):
    stable_rotations = []

    for i, euler in enumerate(ROTATION_CANDIDATES):
        mesh = trimesh.load_mesh(stl_path)
        mesh.apply_scale(mesh_scale)

        # Apply rotation
        R = euler_to_matrix(euler)
        mesh.apply_transform(R)

        # Export rotated mesh as temporary OBJ
        temp_obj_path = f"/tmp/temp_mesh_rot_{i}.obj"
        mesh.export(temp_obj_path)

        # Start PyBullet GUI
        p.connect(p.DIRECT)  # Use p.GUI if you want to visualize
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        visual_id = p.createVisualShape(p.GEOM_MESH, fileName=temp_obj_path, meshScale=[1,1,1])
        collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=temp_obj_path, meshScale=[1,1,1])

        if visual_id < 0 or collision_id < 0:
            #print(f"❌ Rotation {euler} failed shape creation")
            p.disconnect()
            continue

        body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=[0, 0, 1]
        )

        initial_pos, initial_orn = p.getBasePositionAndOrientation(body_id)
        initial_aabb = p.getAABB(body_id)

        # Let it settle
        for _ in range(int(settle_time * 240)):
            p.stepSimulation()
            #time.sleep(1 / 240.0)

        #final_pos, _ = p.getBasePositionAndOrientation(body_id)
        #stable = final_pos[2] < 1.05

        # After letting the object settle
        #for _ in range(int(settle_time * 240)):
        #    p.stepSimulation()

        # Record initial orientation and position
        #initial_pos, initial_orn = p.getBasePositionAndOrientation(body_id)
        #initial_aabb = p.getAABB(body_id)

        # Apply a small force in the +X direction at the center of mass
        p.applyExternalForce(
            objectUniqueId=body_id,
            linkIndex=-1,
            forceObj=[50, 50, 0],  # Small force in Newtons
            posObj=initial_pos,  # Apply at center of mass
            flags=p.WORLD_FRAME
        )

        # Simulate briefly to see if it tips
        for _ in range(2400):  # 1 second at 240 Hz
            p.stepSimulation()

        # Get final position and orientation
        final_pos, final_orn = p.getBasePositionAndOrientation(body_id)

        # Check for tipping
        # You can check either for significant Z movement or change in orientation
        z_diff = abs(final_pos[2] - initial_pos[2])
        angle_diff = quaternion_diff(initial_orn, final_orn)
        #tipped = angle_diff > np.radians(50) 
        #tipped = z_diff > 0.005 

        final_aabb = p.getAABB(body_id)
        initial_z_height = initial_aabb[1][2] - initial_aabb[0][2]
        final_z_height = final_aabb[1][2] - final_aabb[0][2]
        z_diff = abs(final_z_height - initial_z_height) 
        #z_diff = abs(final_aabb[1][2] - initial_aabb[1][2])  # Compare top Z of AABB
        tipped = z_diff > 0.2

        #print(initial_z_height,final_z_height)

        if tipped:
            i=0
            #print(f"⚠️ Tipped under small force: {euler}")
        else:
            #print(f"✅ Did not tip under small force: {euler}")
            stable_rotations.append(euler)

        #time.sleep(2)  # Pause to view final position before disconnecting
        p.disconnect()

    return stable_rotations

def xy_footprint_area(mesh):
    """
    Computes the XY convex hull area of the mesh projected onto the XY plane.
    This is a better estimate of the real contact area than the bounding box.
    """
    # Project vertices onto XY plane
    xy = mesh.vertices[:, :2]

    # Compute 2D convex hull
    hull = ConvexHull(xy)
    hull_points = xy[hull.vertices]

    # Area of convex hull polygon
    polygon = Polygon(hull_points)
    return polygon.area


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale.'''
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    centers = np.mean(limits, axis=1)
    max_range = np.max(limits[:, 1] - limits[:, 0]) / 2

    for center, set_lim in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        set_lim(center - max_range, center + max_range)



def compare_voxels_rotations(mesh_file,pitch=VOXEL_PITCH):
    """
    Compares the surface contact area and bounding box area for different Euler rotations.
    """
    mesh = trimesh.load(mesh_file)
    #voxelized_mesh = mesh.voxelized(pitch=pitch).as_boxes()
    voxel_grid = mesh.voxelized(pitch=pitch)
    voxel_coords = voxel_grid.points  # (N, 3) array of voxel centers

    surface_areas = []
    bb_areas = []

    num_rotations = len(ROTATION_CANDIDATES)
    grid_size = int(np.ceil(np.sqrt(num_rotations)))
    fig = plt.figure(figsize=(grid_size * 4, grid_size * 4))

    for idx, euler in enumerate(ROTATION_CANDIDATES):
        ax = fig.add_subplot(grid_size, grid_size, idx + 1, projection='3d')

        # Convert Euler to rotation matrix
        rotation_matrix = trimesh.transformations.euler_matrix(*euler, axes='sxyz')[:3, :3]

        # Rotate voxel coordinates
        #voxel_coords = voxelized_mesh.centroids
        rotated_voxels_real = np.dot(voxel_coords, rotation_matrix.T)
        #rotated_voxels_real = np.dot(voxelized_mesh, rotation_matrix.T)
        rotated_voxels = np.round(rotated_voxels_real / VOXEL_PITCH).astype(int)

        # Find bottom layer
        min_z = np.min(rotated_voxels[:, 2])
        bottom_voxels = rotated_voxels[rotated_voxels[:, 2] == min_z]

        # Compute surface area on XY
        area = len(bottom_voxels) * (VOXEL_PITCH ** 2)
        surface_areas.append(area)

        # Compute bounding box area
        min_x, max_x = rotated_voxels_real[:, 0].min(), rotated_voxels_real[:, 0].max()
        min_y, max_y = rotated_voxels_real[:, 1].min(), rotated_voxels_real[:, 1].max()
        bb_area = (max_x - min_x) * (max_y - min_y)
        bb_areas.append(bb_area)

        # Zip and sort by surface area (descending)
        rotation_area_pairs = list(zip(ROTATION_CANDIDATES, surface_areas))
        rotation_area_pairs.sort(key=lambda x: x[1], reverse=True)  # largest to smallest

        # Unzip if you want sorted lists separately
        sorted_rotations, sorted_areas = zip(*rotation_area_pairs)

        # Plot
        #ax.scatter(rotated_voxels_real[:, 0], rotated_voxels_real[:, 1], rotated_voxels_real[:, 2], s=2)
        #ax.set_title(f"Euler: {np.round(euler, 2)}\nArea: {area:.4f}, BB: {bb_area:.4f}")
        #ax.set_box_aspect([1, 1, 1])

    #plt.tight_layout()
    #plt.show()

    return surface_areas, bb_areas, sorted_rotations





# === Create Bin Walls ===
'''
wall_half_len = BIN_SIZE 
walls = [
    ([wall_half_len, WALL_THICKNESS, WALL_HEIGHT], [0, wall_half_len, WALL_HEIGHT]), #[BIN_SIZE, WALL_THICKNESS, WALL_HEIGHT], [0, BIN_SIZE, WALL_HEIGHT]),
    ([wall_half_len, WALL_THICKNESS, WALL_HEIGHT], [0, -wall_half_len, WALL_HEIGHT]), #[BIN_SIZE, WALL_THICKNESS, WALL_HEIGHT], [0, -BIN_SIZE, WALL_HEIGHT]),
    ([WALL_THICKNESS, wall_half_len, WALL_HEIGHT], [wall_half_len, 0, WALL_HEIGHT]), #[WALL_THICKNESS, BIN_SIZE, WALL_HEIGHT], [BIN_SIZE, 0, WALL_HEIGHT]),
    ([WALL_THICKNESS, wall_half_len, WALL_HEIGHT], [-wall_half_len, 0, WALL_HEIGHT]), #[WALL_THICKNESS, BIN_SIZE, WALL_HEIGHT], [-BIN_SIZE, 0, WALL_HEIGHT]),
]
for size, pos in walls:
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=pos)
'''
'''
def create_shakable_bin():
    wall_height = WALL_HEIGHT
    wall_thickness = WALL_THICKNESS
    half_size = BIN_SIZE
    base_thickness = 0.02

    collision_shapes = []
    positions = []

    # Base
    collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size, half_size, base_thickness]))
    positions.append([0, 0, base_thickness])

    # +Y Wall
    collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size, wall_thickness, wall_height]))
    positions.append([0, half_size + wall_thickness, wall_height])

    # -Y Wall
    collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size, wall_thickness, wall_height]))
    positions.append([0, -half_size - wall_thickness, wall_height])

    # +X Wall
    collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, half_size, wall_height]))
    positions.append([half_size + wall_thickness, 0, wall_height])

    # -X Wall
    collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, half_size, wall_height]))
    positions.append([-half_size - wall_thickness, 0, wall_height])

    bin_id = p.createMultiBody(
        baseMass=0.0,  # set to >0 for dynamic; 0 for kinematic
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0],
        linkMasses=[0] * len(collision_shapes),
        linkCollisionShapeIndices=collision_shapes,
        linkVisualShapeIndices=[-1] * len(collision_shapes),
        linkPositions=positions,
        linkOrientations=[[0, 0, 0, 1]] * len(collision_shapes),
        linkInertialFramePositions=[[0, 0, 0]] * len(collision_shapes),
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * len(collision_shapes),
        linkParentIndices=[0] * len(collision_shapes),
        linkJointTypes=[p.JOINT_FIXED] * len(collision_shapes),
        linkJointAxis=[[0, 0, 0]] * len(collision_shapes),
    )
    return bin_id
'''

'''
def create_shakable_bin():
    wall_height = WALL_HEIGHT / 2
    wall_thickness = WALL_THICKNESS
    base_thickness = 0.02

    half_length = BIN_LENGTH / 2
    half_width = BIN_WIDTH / 2

    collision_shapes = []
    positions = []

    # Base
    collision_shapes.append(
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_length, half_width, base_thickness])
    )
    positions.append([0, 0, base_thickness])

    # +Y Wall (back wall)
    collision_shapes.append(
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_length, wall_thickness, wall_height])
    )
    positions.append([0, half_width + wall_thickness, wall_height])

    # -Y Wall (front wall)
    collision_shapes.append(
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_length, wall_thickness, wall_height])
    )
    positions.append([0, -half_width - wall_thickness, wall_height])

    # +X Wall (right wall)
    collision_shapes.append(
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, half_width, wall_height])
    )
    positions.append([half_length + wall_thickness, 0, wall_height])

    # -X Wall (left wall)
    collision_shapes.append(
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, half_width, wall_height])
    )
    positions.append([-half_length - wall_thickness, 0, wall_height])

    bin_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0],
        linkMasses=[0] * len(collision_shapes),
        linkCollisionShapeIndices=collision_shapes,
        linkVisualShapeIndices=[-1] * len(collision_shapes),
        linkPositions=positions,
        linkOrientations=[[0, 0, 0, 1]] * len(collision_shapes),
        linkInertialFramePositions=[[0, 0, 0]] * len(collision_shapes),
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * len(collision_shapes),
        linkParentIndices=[0] * len(collision_shapes),
        linkJointTypes=[p.JOINT_FIXED] * len(collision_shapes),
        linkJointAxis=[[0, 0, 0]] * len(collision_shapes),
    )

    return bin_id
'''

def create_shakable_bin():
    wall_height = WALL_HEIGHT / 2
    wall_thickness = WALL_THICKNESS
    base_thickness = 0.02

    half_length = BIN_LENGTH / 2
    half_width = BIN_WIDTH / 2

    collision_shapes = []
    positions = []

    color = [0.8, 0.2, 0.2, 1]

    collision_shapes = []
    visual_shapes = []
    positions = []

    # Base
    half_base = [half_length, half_width, base_thickness]
    collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_base))
    visual_shapes.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_base, rgbaColor=color))
    positions.append([0, 0, base_thickness])

    # +Y Wall (back wall)
    half_wall_y = [half_length, wall_thickness, wall_height]
    collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_wall_y))
    visual_shapes.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_wall_y, rgbaColor=color))
    positions.append([0, half_width + wall_thickness, wall_height])

    # -Y Wall (front wall)
    collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_wall_y))
    visual_shapes.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_wall_y, rgbaColor=color))
    positions.append([0, -half_width - wall_thickness, wall_height])

    # +X Wall (right wall)
    half_wall_x = [wall_thickness, half_width, wall_height]
    collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_wall_x))
    visual_shapes.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_wall_x, rgbaColor=color))
    positions.append([half_length + wall_thickness, 0, wall_height])

    # -X Wall (left wall)
    collision_shapes.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half_wall_x))
    visual_shapes.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half_wall_x, rgbaColor=color))
    positions.append([-half_length - wall_thickness, 0, wall_height])

    # Create the bin
    bin_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0],
        linkMasses=[0.0] * len(collision_shapes),
        linkCollisionShapeIndices=collision_shapes,
        linkVisualShapeIndices=visual_shapes,
        linkPositions=positions,
        linkOrientations=[[0, 0, 0, 1]] * len(collision_shapes),
        linkInertialFramePositions=[[0, 0, 0]] * len(collision_shapes),
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * len(collision_shapes),
        linkParentIndices=[0] * len(collision_shapes),
        linkJointTypes=[p.JOINT_FIXED] * len(collision_shapes),
        linkJointAxis=[[0, 0, 0]] * len(collision_shapes),
    )


def get_bin_min_max(bin_id):
    # Bin size and wall information
    half_size = BIN_SIZE
    wall_height = WALL_HEIGHT
    wall_thickness = WALL_THICKNESS
    base_thickness = 0.02
    
    # Base position (central, Z = base_thickness)
    base_min_x = -half_size
    base_max_x = half_size
    base_min_y = -half_size
    base_max_y = half_size
    base_min_z = 0  # Floor level
    base_max_z = base_thickness  # Just the thickness of the base
    
    # Wall extents
    # +Y Wall
    wall_plus_y_min_x = -half_size
    wall_plus_y_max_x = half_size
    wall_plus_y_min_y = half_size
    wall_plus_y_max_y = half_size + wall_thickness
    wall_plus_y_min_z = base_max_z
    wall_plus_y_max_z = wall_plus_y_min_z + wall_height

    # -Y Wall
    wall_minus_y_min_x = -half_size
    wall_minus_y_max_x = half_size
    wall_minus_y_min_y = -half_size - wall_thickness
    wall_minus_y_max_y = -half_size
    wall_minus_y_min_z = base_max_z
    wall_minus_y_max_z = wall_minus_y_min_z + wall_height

    # +X Wall
    wall_plus_x_min_x = half_size
    wall_plus_x_max_x = half_size + wall_thickness
    wall_plus_x_min_y = -half_size
    wall_plus_x_max_y = half_size
    wall_plus_x_min_z = base_max_z
    wall_plus_x_max_z = wall_plus_x_min_z + wall_height

    # -X Wall
    wall_minus_x_min_x = -half_size - wall_thickness
    wall_minus_x_max_x = -half_size
    wall_minus_x_min_y = -half_size
    wall_minus_x_max_y = half_size
    wall_minus_x_min_z = base_max_z
    wall_minus_x_max_z = wall_minus_x_min_z + wall_height

    # Combine all min and max values for X, Y, and Z coordinates
    min_x = min(base_min_x, wall_plus_x_min_x, wall_minus_x_min_x)
    max_x = max(base_max_x, wall_plus_x_max_x, wall_minus_x_max_x)
    
    min_y = min(base_min_y, wall_plus_y_min_y, wall_minus_y_min_y)
    max_y = max(base_max_y, wall_plus_y_max_y, wall_minus_y_max_y)
    
    min_z = min(base_min_z, wall_plus_y_min_z, wall_minus_y_min_z, wall_plus_x_min_z, wall_minus_x_min_z)
    max_z = max(base_max_z, wall_plus_y_max_z, wall_minus_y_max_z, wall_plus_x_max_z, wall_minus_x_max_z)
    
    return (min_x, max_x), (min_y, max_y), (min_z, max_z)

# === Box Shake Function ===
def shake_box(object_id):
    # Apply random forces and/or torques to simulate shaking
    random_force = [random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(0, 0)]  # X and Y forces
    random_torque = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]  # Random torque
    p.applyExternalForce(object_id, linkIndex=-1, forceObj=random_force, posObj=[0, 0, 0], flags=p.WORLD_FRAME)
    p.applyExternalTorque(object_id, linkIndex=-1, torqueObj=random_torque, flags=p.WORLD_FRAME)

def shake_bin(duration=20.0, strength=500):
    start = time.time()
    direction = 1
    print("Start Shake")
    while time.time() - start < duration:
        shift = direction * strength
        p.applyExternalForce(objectUniqueId=-1, linkIndex=-1,
                             forceObj=[shift, 0, 0], posObj=[0, 0, 0],
                             flags=p.WORLD_FRAME)
        direction *= -1
        p.stepSimulation()
        #time.sleep(1 / 240)
    print("Stop Shake")

def shake_bin_rigidbody(bin_id, duration=2.0, strength=100):
    start = time.time()
    direction = 1
    while time.time() - start < duration:
        shift = direction * strength
        p.resetBaseVelocity(bin_id, linearVelocity=[shift, 0, 0])
        direction *= -1
        p.stepSimulation()
        #time.sleep(1 / 240)
    # Stop motion
    p.resetBaseVelocity(bin_id, linearVelocity=[0, 0, 0])

# === Utility Functions ===
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

def get_xy_footprint(mesh_path, scale, rotation):
    # Load and rotate mesh
    mesh = trimesh.load_mesh(mesh_path)
    mesh.apply_transform(rotation)
    aabb = mesh.bounds  # min and max corners
    width = aabb[1][0] - aabb[0][0]
    depth = aabb[1][1] - aabb[0][1]
    return width * depth, mesh

def is_stable(object_id, position, orientation, duration=2.0, tolerance=0.001):
    # Set the object's position and orientation
    p.resetBasePositionAndOrientation(object_id, position, orientation)
    
    # Get the initial position and orientation
    initial_position, initial_orientation = p.getBasePositionAndOrientation(object_id)
    
    # Simulate for the specified duration
    start_time = time.time()
    while time.time() - start_time < duration:
        p.stepSimulation()
        #time.sleep(1 / 240)
    
    # Get the final position and orientation
    final_position, final_orientation = p.getBasePositionAndOrientation(object_id)
    
    # Check if the position and orientation have changed
    position_diff = np.linalg.norm(np.array(final_position) - np.array(initial_position))
    orientation_diff = np.linalg.norm(np.array(final_orientation) - np.array(initial_orientation))
    
    # If both have not changed significantly, the object is stable
    return position_diff < tolerance and orientation_diff < tolerance


def is_stable_orientation(mesh, transform):
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(transform)

    # Get the lowest Z face (support face)
    face_normals = mesh_copy.face_normals
    z_aligned_faces = np.where(np.isclose(face_normals[:, 2], 1.0, atol=1e-3))[0]

    if len(z_aligned_faces) == 0:
        return False

    # Center of mass must be over support face
    com = mesh_copy.center_mass
    aabb = mesh_copy.bounds
    min_z = aabb[0][2]

    # If CoM is near the bottom (within small tolerance), assume stable
    return (com[2] - min_z) < 0.1  # 1cm tolerance

def is_physically_stable_in_pybullet(mesh_path, scale, transform, center_offset):
    mesh = trimesh.load_mesh(mesh_path)
    mesh.apply_transform(transform)
    mesh.export("temp_check.stl")

    col_id = p.createCollisionShape(p.GEOM_MESH, fileName="temp_check.stl", meshScale=[scale]*3)
    vis_id = p.createVisualShape(p.GEOM_MESH, fileName="temp_check.stl", meshScale=[scale]*3)
    body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id,
                                baseVisualShapeIndex=vis_id,
                                basePosition=[0, 0, 0.05])  # small drop

    for _ in range(120):
        p.stepSimulation()

    _, orientation = p.getBasePositionAndOrientation(body_id)
    euler = p.getEulerFromQuaternion(orientation)
    p.removeBody(body_id)

    return abs(euler[0]) < 0.1 and abs(euler[1]) < 0.1

def get_best_stable_orientation(mesh_path, scale):
    mesh = trimesh.load_mesh(mesh_path)
    rotations = generate_rotations()
    best_transform = None
    min_area = float('inf')
    best_mesh = None

    for rot in rotations:
        transform = np.eye(4)
        transform[:3, :3] = rot
        if not is_stable_orientation(mesh, transform):
            continue
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(transform)
        area = mesh_copy.bounds[1][0] - mesh_copy.bounds[0][0]
        area *= mesh_copy.bounds[1][1] - mesh_copy.bounds[0][1]

        if area < min_area:
            temp_path = "temp_candidate.stl"
            mesh_copy.export(temp_path)
            half_extents, center_offset = get_mesh_half_extents(temp_path, scale)
            if is_physically_stable_in_pybullet(mesh_path, scale, transform, center_offset):
                min_area = area
                best_transform = transform
                best_mesh = mesh_copy

    if best_transform is None:
        return None, None, None
    return best_transform, best_mesh, center_offset

def get_stable_orientations(mesh_path, scale=0.01, sim_time=0.5, threshold=0.01):
    stable_orientations = []
    mesh = trimesh.load_mesh(mesh_path)
    rotations = generate_rotations()

    for R in rotations:
        rotated_mesh = mesh.copy()
        rotated_mesh.apply_transform(np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1])))

        # Save rotated mesh temporarily
        temp_path = "temp_rotated.obj"
        rotated_mesh.export(temp_path)

        # Load into PyBullet
        col_id = p.createCollisionShape(p.GEOM_MESH, fileName=temp_path, meshScale=[scale]*3)
        vis_id = p.createVisualShape(p.GEOM_MESH, fileName=temp_path, meshScale=[scale]*3)
        body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id,
                                    baseVisualShapeIndex=vis_id, basePosition=[0, 0, 0.1])

        # Simulate physics
        for _ in range(int(sim_time * 240)):
            p.stepSimulation()
            #time.sleep(1 / 240)

        # Check base velocity and orientation
        lin_vel, ang_vel = p.getBaseVelocity(body_id)
        if np.linalg.norm(lin_vel) < threshold and np.linalg.norm(ang_vel) < threshold:
            stable_orientations.append(R)

        p.removeBody(body_id)

    return stable_orientations

'''
def frange(start, stop, step):
    while start < stop:
        yield round(start, 5)
        start += step
'''

def frange(start, stop, step):
    i = start
    while (step > 0 and i < stop) or (step < 0 and i > stop):
        yield round(i, 10)  # Avoid floating point precision issues
        i += step

'''


def frange(start, stop, step):
    if step == 0:
        raise ValueError("step must not be zero")
    if (stop - start) * step < 0:
        return  # no values to generate
    while (step > 0 and start < stop) or (step < 0 and start > stop):
        yield round(start, 10)
        start += step
'''

def get_mesh_half_extents(mesh_path, scale):
    col_id = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis_id = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3)
    temp_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id)
    aabb_min, aabb_max = p.getAABB(temp_id)
    center = [(min_val + max_val) / 2 for min_val, max_val in zip(aabb_min, aabb_max)]
    half_extents = [(max_val - min_val) / 2 for min_val, max_val in zip(aabb_min, aabb_max)]
    p.removeBody(temp_id)
    return half_extents, center

'''
def does_overlap(pos, size, placed_objects):
    x1_min, x1_max = pos[0] - size[0], pos[0] + size[0]
    y1_min, y1_max = pos[1] - size[1], pos[1] + size[1]
    z1_min, z1_max = pos[2] - size[2], pos[2] + size[2]
    for obj_pos, obj_size in placed_objects:
        x2_min, x2_max = obj_pos[0] - obj_size[0], obj_pos[0] + obj_size[0]
        y2_min, y2_max = obj_pos[1] - obj_size[1], obj_pos[1] + obj_size[1]
        z2_min, z2_max = obj_pos[2] - obj_size[2], obj_pos[2] + obj_size[2]
        if not (x1_max <= x2_min or x1_min >= x2_max or
                y1_max <= y2_min or y1_min >= y2_max or
                z1_max <= z2_min or z1_min >= z2_max):
            return True
    return False
'''
def does_overlap(pos, mesh_path, scale, center_offset, placed_ids):
    corrected_pos = [pos[i] - center_offset[i] for i in range(3)]
    
    # Create a temporary body for the candidate mesh
    col_id = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[scale]*3,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )
    temp_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        basePosition=corrected_pos
    )

    # Check for collisions with all placed objects
    for pid in placed_ids:
        contact_points = p.getClosestPoints(bodyA=temp_id, bodyB=pid, distance=0)
        if contact_points:
            p.removeBody(temp_id)
            return True

    # No collision detected
    p.removeBody(temp_id)
    return False


def spawn_mesh(position, rot, mesh_path, scale, center_offset):
    corrected_position = [position[i] - center_offset[i] for i in range(3)]
    # Convert rotation to quaternion
    if isinstance(rot, list):  # Euler
        quat = p.getQuaternionFromEuler(rot)
    else:  # Rotation matrix
        quat = R.from_matrix(rot).as_quat()  # [x, y, z, w]

    color = np.append(np.random.rand(3,), 1.0).tolist()  
    col_id = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3)
    vis_id = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3, rgbaColor=color)
    return p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id,
                             baseVisualShapeIndex=vis_id, basePosition=corrected_position,
                             baseOrientation=quat)  # Rotate 90 degrees around X


'''
def pack_meshes(mesh_list, mesh_scale=MESH_SCALE_DEFAULT):
    placed = []
    for mesh_path in mesh_list:
        half_extents, center_offset = get_mesh_half_extents(mesh_path, mesh_scale)

        x_min = -BIN_SIZE + WALL_THICKNESS + half_extents[0] + MARGIN
        x_max =  BIN_SIZE - WALL_THICKNESS - half_extents[0] - MARGIN
        y_min = -BIN_SIZE + WALL_THICKNESS + half_extents[1] + MARGIN
        y_max =  BIN_SIZE - WALL_THICKNESS - half_extents[1] - MARGIN

        placed_successfully = False
        for z in frange(half_extents[2], MAX_HEIGHT - half_extents[2], Z_STEP):
            for x in frange(x_min, x_max, STEP):
                for y in frange(y_min, y_max, STEP):
                    pos = [x,y,z] #[x - center_offset[0], y - center_offset[1], z - center_offset[2]] #[x, y, z]
                    SHRINK = 0.0001  # or smaller
                    shrunk_size = [s - SHRINK for s in half_extents]
                    if not does_overlap(pos, shrunk_size, placed):
                    #if not does_overlap(pos, half_extents, placed):
                        spawn_mesh(pos, mesh_path, mesh_scale, center_offset)
                        #spawn_mesh(pos, mesh_path, mesh_scale)
                        placed.append((pos, half_extents))
                        placed_successfully = True
                        break
                if placed_successfully:
                    break
            if placed_successfully:
                break

        if not placed_successfully:
            print(f"Unable to place mesh: {os.path.basename(mesh_path)}")
'''
def check_collision(body_id, placed_ids, min_dist=0.001):
    """Check for actual mesh-to-mesh collision using PyBullet."""
    for other_id in placed_ids:
        points = p.getClosestPoints(bodyA=body_id, bodyB=other_id, distance=min_dist)
        if points:
            return True  # Collision detected
    return False


def mesh_collision_check(pos, rot, mesh_path, scale, center_offset, placed_ids):
    corrected_pos = [pos[i] - center_offset[i] for i in range(3)]

    # Convert rotation to quaternion
    if isinstance(rot, list):  # Euler
        quat = p.getQuaternionFromEuler(rot)
    else:  # Rotation matrix
        quat = R.from_matrix(rot).as_quat()  # [x, y, z, w]

    col_id = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[scale]*3,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )
    temp_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        basePosition=corrected_pos,
        baseOrientation=quat
    )

    for pid in placed_ids:
        contacts = p.getClosestPoints(bodyA=temp_id, bodyB=pid, distance=0)
        if contacts:
            p.removeBody(temp_id)
            return True
    p.removeBody(temp_id)

    # Check bounds
    half_extents, center = get_mesh_half_extents(mesh_path, scale)

    #print(pos[0],half_extents[0])

    # Get object AABB
    x_min = pos[0] - half_extents[0]
    x_max = pos[0] + half_extents[0]
    y_min = pos[1] - half_extents[1]
    y_max = pos[1] + half_extents[1]
    z_min = pos[2] - half_extents[2]
    z_max = pos[2] + half_extents[2]

    # Get container bounds once
    #INTERIOR_X_MIN = -BIN_SIZE + WALL_THICKNESS + MARGIN
    #INTERIOR_X_MAX =  BIN_SIZE - WALL_THICKNESS - MARGIN
    #INTERIOR_Y_MIN = -BIN_SIZE + WALL_THICKNESS + MARGIN
    #INTERIOR_Y_MAX =  BIN_SIZE - WALL_THICKNESS - MARGIN

    INTERIOR_X_MIN = -BIN_LENGTH / 2 + WALL_THICKNESS + MARGIN
    INTERIOR_X_MAX =  BIN_LENGTH / 2 - WALL_THICKNESS - MARGIN
    INTERIOR_Y_MIN = -BIN_WIDTH / 2 + WALL_THICKNESS + MARGIN
    INTERIOR_Y_MAX =  BIN_WIDTH / 2 - WALL_THICKNESS - MARGIN
    INTERIOR_Z_MIN = 0 + MARGIN
    INTERIOR_Z_MAX = WALL_HEIGHT - MARGIN

    # Compare for out-of-bounds
    if (x_min < INTERIOR_X_MIN - MARGIN or x_max > INTERIOR_X_MAX + MARGIN or
        y_min < INTERIOR_Y_MIN - MARGIN or y_max > INTERIOR_Y_MAX + MARGIN): # or
        #z_min < INTERIOR_Z_MIN or z_max > INTERIOR_Z_MAX):
        return True

    return False


def try_place_mesh(position, mesh_path, scale, placed_ids, center_offset):
    """Spawn mesh and check collision with existing ones using full geometry."""
    corrected_pos = [position[i] - center_offset[i] for i in range(3)]
    col_id = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3)
    vis_id = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3)
    body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id,
                                baseVisualShapeIndex=vis_id, basePosition=corrected_pos)

    if check_collision(body_id, placed_ids):
        p.removeBody(body_id)
        return None
    return body_id


def pack_meshes(mesh_list, mesh_scale=MESH_SCALE_DEFAULT):
    """Try to pack all meshes into the bin using full collision checks."""
    placed_ids = []
    placed_meshes = []
    not_placed_ids = []
    volumes = []
    stack_count = {}
    i = 0
    for mesh_path, volume, sorted_rotations in mesh_list:
        half_extents, center_offset = get_mesh_half_extents(mesh_path, mesh_scale)
        rotations = generate_rotations()

        #x_min = -BIN_SIZE + WALL_THICKNESS + half_extents[0] + MARGIN
        #x_max =  BIN_SIZE - WALL_THICKNESS - half_extents[0] - MARGIN
        #y_min = -BIN_SIZE + WALL_THICKNESS + half_extents[1] + MARGIN
        #y_max =  BIN_SIZE - WALL_THICKNESS - half_extents[1] - MARGIN

        x_min = -BIN_LENGTH / 2 + WALL_THICKNESS + half_extents[0] + MARGIN
        x_max =  BIN_LENGTH / 2 - WALL_THICKNESS - half_extents[0] - MARGIN
        y_min = -BIN_WIDTH / 2 + WALL_THICKNESS + half_extents[1] + MARGIN
        y_max =  BIN_WIDTH / 2 - WALL_THICKNESS - half_extents[1] - MARGIN        

        original_mesh = trimesh.load_mesh(mesh_path)
        placed_successfully = False
        scale_place = 0.01
        for rot in sorted_rotations:
            #rot = [0,0,0]
            for z in frange(half_extents[2], MAX_HEIGHT - half_extents[2], Z_STEP):
                layer_filled = False
                if placed_successfully:
                    break 
                for x in frange(x_min, x_max, STEP):
                    if placed_successfully:
                        break 
                    for y in frange(y_min, y_max, STEP):
                        #print("ROT:", rot)
                        pos = [x, y, z]
                        #print("Start")
                        #print(f"Testing pos: ({x}, {y}, {z})")
                        if z < SMALL_Z_THRESHOLD and not mesh_collision_check(pos, rot, mesh_path, mesh_scale, center_offset, placed_ids):
                            #print("ENTER")
                            last_valid_pos = pos
                            for y_off in frange(0, 2 * half_extents[1], scale_place):
                                candidate = [x, y - y_off, z]
                                if not mesh_collision_check(candidate, rot, mesh_path, mesh_scale, center_offset, placed_ids):
                                    last_valid_pos = candidate
                                else:
                                    break
                            # Try backtracking in X from updated Y
                            final_pos = last_valid_pos
                            for x_off in frange(0, 2 * half_extents[0], scale_place):
                                candidate = [x - x_off, final_pos[1], z]
                                if not mesh_collision_check(candidate, rot, mesh_path, mesh_scale, center_offset, placed_ids):
                                    final_pos = candidate
                                else:
                                    break
                            #print("✅ Final placement at:", final_pos)
                            new_id = spawn_mesh(final_pos, rot, mesh_path, mesh_scale, center_offset)
                            placed_ids.append(new_id)
                            placed_meshes.append(mesh_path)
                            volumes.append(compute_stl_volume(mesh_path, 10))
                            layer_filled = True
                            placed_successfully = True
                            for _ in range(1200):  # 1 second at 240Hz
                                p.stepSimulation()
                            break
                        if placed_successfully:
                            break
                    if placed_successfully:
                        break
                if placed_successfully:
                    break
            if placed_successfully:
                break

        if not placed_successfully:
            print(f"❌ Unable to place mesh: {os.path.basename(mesh_path)}")
            not_placed_ids.append(1)

        #capture_image(i)
        # Render image from camera
        width_px, height_px = 640, 480
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.1, 0, 2],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width_px/height_px, nearVal=0.1, farVal=10
        )
        #p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])
        img_arr = p.getCameraImage(
            width=width_px,
            height=height_px,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Extract and save RGB image
        rgb_array = img_arr[2]
        image = Image.fromarray(rgb_array)
        os.makedirs("run_images", exist_ok=True)
        image.save(f"run_images{i}.png")
        i = i + 1


    max_z = 0
    total_volume = 0.0
    volume_inside_height = 0.0

    for i, obj_id in enumerate(placed_ids):
        pos, rot_obj = p.getBasePositionAndOrientation(obj_id)
        x, y, z = pos

        max_z = max(max_z, z)

        # Add to total volume
        total_volume += volumes[i]

        #print(placed_meshes[i]); 
        #print(rot_obj)
        volume_inside = compute_clipped_volume_transformed(placed_meshes[i], pos, rot_obj, WALL_HEIGHT)
        volume_inside_height = volume_inside + volume_inside_height
        #print(volume_inside_height)

        # Check if the top of the object is below box height
        #top_z = z + object_heights[i]/2
        #if top_z < height:
        #    volume_inside_height += volumes[i]
        #print(volume_inside_height)

    print("Total Volume:",total_volume); print("Inside Volume:", volume_inside_height)
    
    # Render image from camera
    width_px, height_px = 640, 480
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.1, 0, 2],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 0, 1]
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width_px/height_px, nearVal=0.1, farVal=10
    )
    #p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])
    img_arr = p.getCameraImage(
        width=width_px,
        height=height_px,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    # Extract and save RGB image
    rgb_array = img_arr[2]
    image = Image.fromarray(rgb_array)
    os.makedirs("run_images", exist_ok=True)
    image.save(f"run_images_final.png")

    # Let simulation settle
    for _ in range(1200):  # 1 second at 240Hz
        p.stepSimulation()
        #time.sleep(1.0 / 240.0)
    all_objects_in_bin(placed_ids)

    return placed_ids, not_placed_ids


def compute_stl_volume(stl_path, scale):
    your_mesh = trimesh.load_mesh(stl_path)
    return your_mesh.volume * (scale ** 3)  # Apply cubic scaling

def stl_bounding_box(stl_path):
    # Load STL file
    your_mesh = trimesh.load_mesh(stl_path)
    bounds = your_mesh.bounds  # shape: (2, 3), rows are [min, max]
    dimensions = bounds[1] - bounds[0]  # max - min per axis
    return dimensions
'''
def compute_clipped_volume_transformed(stl_path, position, rotation_matrix, bin_height):
    # Load STL as mesh
    mesh = trimesh.load_mesh(stl_path)
    
    # Construct transformation matrix (4x4)
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    
    # Apply transformation to mesh
    mesh.apply_transform(transform)
    
    # Define clipping plane (z = bin_height)
    plane_origin = [0, 0, bin_height]
    plane_normal = [0, 0, 1]

    # Slice mesh below the bin height
    clipped = mesh.slice_plane(plane_origin, plane_normal)

    if clipped is None or clipped.volume is None:
        return 0.0
    
    return clipped.volume
'''

from trimesh.creation import box
from scipy.spatial.transform import Rotation as R
import trimesh
import numpy as np

def compute_clipped_volume_transformed(stl_path, position, rot, bin_height, dz=1e-3, scale_factor=10.0):
    # Load the mesh and apply scaling
    mesh = trimesh.load_mesh(stl_path)
    mesh.apply_scale(scale_factor)  # Apply the scaling transformation

    # Convert Euler angles to a rotation matrix
    #rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()
    rotation = R.from_quat(rot)
    rotation_matrix = rotation.as_matrix()
    #print(rotation_matrix)

    # Create 4x4 transformation matrix for position and rotation
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    mesh.apply_transform(transform)

    # Apply clipping and slicing
    z_min = mesh.bounds[0][2]
    z_max = min(mesh.bounds[1][2], bin_height)

    if z_min >= z_max:
        return 0.0  # Entire mesh is above bin height

    # Slice mesh horizontally and accumulate volume
    z_slices = np.arange(z_min, z_max, dz)
    volume = 0.0

    for z in z_slices:
        section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is None:
            continue
        try:
            section_2D, _ = section.to_2D()
            paths = section_2D.polygons_full
            area = sum(p.area for p in paths)
            volume += area * dz #* 1000
        except Exception as e:
            print(f"Sectioning failed at z={z}: {e}")
            continue

    return volume


def create_box(base_position, length, width, height, wall_thickness):
    half_length = length / 2
    half_width = width / 2
    wall_height = height / 2
    base_z = base_position[2]

    color=[0.8, 0.2, 0.2, 1]
    base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_length, half_width, wall_thickness])
    base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_length, half_width, wall_thickness], rgbaColor=color)
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=base_col,
        baseVisualShapeIndex=base_vis,
        basePosition=[base_position[0], base_position[1], base_z - wall_thickness]
    )

    side_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, half_width, wall_height])
    side_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness, half_width, wall_height], rgbaColor=color)
    side_wall_positions = [
        (base_position[0] - half_length + wall_thickness, base_position[1], base_z + wall_height),
        (base_position[0] + half_length - wall_thickness, base_position[1], base_z + wall_height)
    ]
    for pos in side_wall_positions:
        p.createMultiBody(0, side_col, side_vis, basePosition=pos)

    fb_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_length, wall_thickness, wall_height])
    fb_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_length, wall_thickness, wall_height], rgbaColor=color)
    front_back_positions = [
        (base_position[0], base_position[1] - half_width + wall_thickness, base_z + wall_height),
        (base_position[0], base_position[1] + half_width - wall_thickness, base_z + wall_height)
    ]
    for pos in front_back_positions:
        p.createMultiBody(0, fb_col, fb_vis, basePosition=pos)


def small_pack_meshes_randomly(mesh_list, mesh_scale=MESH_SCALE_DEFAULT):
    """Try to pack all meshes into the bin using full collision checks."""
    # Configuration
    length = 0.88
    width = 1.27
    height = 0.63
    wall_thickness = 0.02
    scale = 10
    base_z = 0

    # Run 20 fast iterations
    best_result = None
    results = []

    run = 0
    for run in range(10):
        random.shuffle(mesh_list)
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        bin_id = create_shakable_bin()

        placed_objects = []
        placed_ids = []
        placed_meshes = []
        not_placed_ids = []
        object_heights = []
        volumes = []
        results = []
        stack_count = {}
        i = 0

        for mesh_path, volume, sorted_rotations in mesh_list:
            half_extents, center_offset = get_mesh_half_extents(mesh_path, mesh_scale)
            rotations = generate_rotations()

            x_min = -BIN_LENGTH / 2 + WALL_THICKNESS + half_extents[0] + MARGIN
            x_max =  BIN_LENGTH / 2 - WALL_THICKNESS - half_extents[0] - MARGIN
            y_min = -BIN_WIDTH / 2 + WALL_THICKNESS + half_extents[1] + MARGIN
            y_max =  BIN_WIDTH / 2 - WALL_THICKNESS - half_extents[1] - MARGIN        

            original_mesh = trimesh.load_mesh(mesh_path)
            placed_successfully = False
            scale_place = 0.01
            for rot in sorted_rotations:
                #rot = [0,0,0]
                for z in frange(half_extents[2], MAX_HEIGHT - half_extents[2], Z_STEP):
                    layer_filled = False
                    if placed_successfully:
                        break 
                    for x in frange(x_min, x_max, STEP):
                        if placed_successfully:
                            break 
                        for y in frange(y_min, y_max, STEP):
                            #print("ROT:", rot)
                            pos = [x, y, z]
                            #print("Start")
                            #print(f"Testing pos: ({x}, {y}, {z})")
                            if z < SMALL_Z_THRESHOLD and not mesh_collision_check(pos, rot, mesh_path, mesh_scale, center_offset, placed_ids):
                                #print("ENTER")
                                last_valid_pos = pos
                                for y_off in frange(0, 2 * half_extents[1], scale_place):
                                    candidate = [x, y - y_off, z]
                                    if not mesh_collision_check(candidate, rot, mesh_path, mesh_scale, center_offset, placed_ids):
                                        last_valid_pos = candidate
                                    else:
                                        break
                                # Try backtracking in X from updated Y
                                final_pos = last_valid_pos
                                for x_off in frange(0, 2 * half_extents[0], scale_place):
                                    candidate = [x - x_off, final_pos[1], z]
                                    if not mesh_collision_check(candidate, rot, mesh_path, mesh_scale, center_offset, placed_ids):
                                        final_pos = candidate
                                    else:
                                        break
                                #print("✅ Final placement at:", final_pos)
                                new_id = spawn_mesh(final_pos, rot, mesh_path, mesh_scale, center_offset)
                                placed_ids.append(new_id)
                                placed_meshes.append(mesh_path)
                                volumes.append(compute_stl_volume(mesh_path, 10))
                                dimensions = stl_bounding_box(mesh_path) * 10
                                object_heights.append(dimensions[2])
                                layer_filled = True
                                placed_successfully = True
                                for _ in range(1200):  # 1 second at 240Hz
                                    p.stepSimulation()
                                break
                            if placed_successfully:
                                break
                        if placed_successfully:
                            break
                    if placed_successfully:
                        break
                if placed_successfully:
                    break

            if not placed_successfully:
                #print(f"❌ Unable to place mesh: {os.path.basename(mesh_path)}")
                not_placed_ids.append(1)

            #capture_image(i)
            #i = i + 1

        # Let simulation settle
        for _ in range(1200):  # 1 second at 240Hz
            p.stepSimulation()
            #time.sleep(1.0 / 240.0)
        #all_objects_in_bin(placed_ids)

        max_z = 0
        total_volume = 0.0
        volume_inside_height = 0.0

        for i, obj_id in enumerate(placed_ids):
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            x, y, z = pos

            max_z = max(max_z, z)

            # Add to total volume
            total_volume += volumes[i]

            #print(placed_meshes[i]); print(rot)
            volume_inside = compute_clipped_volume_transformed(placed_meshes[i], pos, rot, WALL_HEIGHT)
            volume_inside_height = volume_inside + volume_inside_height
            #print(volume_inside_height)

            # Check if the top of the object is below box height
            #top_z = z + object_heights[i]/2
            #if top_z < height:
            #    volume_inside_height += volumes[i]
            #print(volume_inside_height)

        results.append({
            "run": run + 1,
            "max_z": round(max_z, 3),
            "total_volume": total_volume,
            "inside_volume": volume_inside_height,
            "objects_placed": len(placed_ids)
        })

        current_result = {
            "run": run + 1,
            "max_z": round(max_z, 3),
            "total_volume": total_volume,
            "inside_volume": volume_inside_height,
            "objects_placed": len(placed_ids)
        }
        results.append(current_result)

        # Update best result if needed
        if (best_result is None or 
            current_result["objects_placed"] > best_result["objects_placed"] or
            (current_result["objects_placed"] == best_result["objects_placed"])):
            best_result = current_result


        # Render image from camera
        width_px, height_px = 640, 480
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.1, 0, 2],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width_px/height_px, nearVal=0.1, farVal=10
        )
        #p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])
        img_arr = p.getCameraImage(
            width=width_px,
            height=height_px,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Extract and save RGB image
        rgb_array = img_arr[2]
        image = Image.fromarray(rgb_array)
        os.makedirs("run_images", exist_ok=True)
        image.save(f"run_images/run_{run+1:02d}.png")
        image.save(f"run_images/best_run_{best_result['run']:02d}.png")

        p.disconnect()

    # Output results
    for result in results:
        print(f"Run {result['run']} | Placed: {result['objects_placed']} | Max Z: {result['max_z']}")

    print("\nBest Result:")
    print(f"Run {best_result['run']} | Placed: {result['objects_placed']} | Total Volume: {best_result['total_volume']} | Inside Volume: {best_result['inside_volume']}")

    return best_result['run'],result['objects_placed'],best_result['max_z']


def large_pack_meshes_randomly(mesh_list, mesh_scale=MESH_SCALE_DEFAULT):
    """Try to pack all meshes into the bin using full collision checks."""
    # Configuration
    length = 1.27
    width = 1.27
    height = 0.63
    wall_thickness = 0.02
    scale = 10
    base_z = 0

    # Run 20 fast iterations
    best_result = None
    results = []

    run = 0
    for run in range(10):
        random.shuffle(mesh_list)
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        bin_id = create_shakable_bin()

        placed_objects = []
        placed_ids = []
        placed_meshes = []
        not_placed_ids = []
        object_heights = []
        volumes = []
        results = []
        stack_count = {}
        i = 0

        for mesh_path, volume, sorted_rotations in mesh_list:
            half_extents, center_offset = get_mesh_half_extents(mesh_path, mesh_scale)
            rotations = generate_rotations()

            x_min = -BIN_LENGTH / 2 + WALL_THICKNESS + half_extents[0] + MARGIN
            x_max =  BIN_LENGTH / 2 - WALL_THICKNESS - half_extents[0] - MARGIN
            y_min = -BIN_WIDTH / 2 + WALL_THICKNESS + half_extents[1] + MARGIN
            y_max =  BIN_WIDTH / 2 - WALL_THICKNESS - half_extents[1] - MARGIN        

            original_mesh = trimesh.load_mesh(mesh_path)
            placed_successfully = False
            scale_place = 0.01
            for rot in sorted_rotations:
                #rot = [0,0,0]
                for z in frange(half_extents[2], MAX_HEIGHT - half_extents[2], Z_STEP):
                    layer_filled = False
                    if placed_successfully:
                        break 
                    for x in frange(x_min, x_max, STEP):
                        if placed_successfully:
                            break 
                        for y in frange(y_min, y_max, STEP):
                            #print("ROT:", rot)
                            pos = [x, y, z]
                            #print("Start")
                            #print(f"Testing pos: ({x}, {y}, {z})")
                            if z < SMALL_Z_THRESHOLD and not mesh_collision_check(pos, rot, mesh_path, mesh_scale, center_offset, placed_ids):
                                #print("ENTER")
                                last_valid_pos = pos
                                for y_off in frange(0, 2 * half_extents[1], scale_place):
                                    candidate = [x, y - y_off, z]
                                    if not mesh_collision_check(candidate, rot, mesh_path, mesh_scale, center_offset, placed_ids):
                                        last_valid_pos = candidate
                                    else:
                                        break
                                # Try backtracking in X from updated Y
                                final_pos = last_valid_pos
                                for x_off in frange(0, 2 * half_extents[0], scale_place):
                                    candidate = [x - x_off, final_pos[1], z]
                                    if not mesh_collision_check(candidate, rot, mesh_path, mesh_scale, center_offset, placed_ids):
                                        final_pos = candidate
                                    else:
                                        break
                                #print("✅ Final placement at:", final_pos)
                                new_id = spawn_mesh(final_pos, rot, mesh_path, mesh_scale, center_offset)
                                placed_ids.append(new_id)
                                placed_meshes.append(mesh_path)
                                volumes.append(compute_stl_volume(mesh_path, 10))
                                dimensions = stl_bounding_box(mesh_path) * 10
                                object_heights.append(dimensions[2])
                                layer_filled = True
                                placed_successfully = True
                                for _ in range(1200):  # 1 second at 240Hz
                                    p.stepSimulation()
                                break
                            if placed_successfully:
                                break
                        if placed_successfully:
                            break
                    if placed_successfully:
                        break
                if placed_successfully:
                    break

            if not placed_successfully:
                #print(f"❌ Unable to place mesh: {os.path.basename(mesh_path)}")
                not_placed_ids.append(1)

            #capture_image(i)
            #i = i + 1

        # Let simulation settle
        for _ in range(1200):  # 1 second at 240Hz
            p.stepSimulation()
            #time.sleep(1.0 / 240.0)
        #all_objects_in_bin(placed_ids)

        max_z = 0
        total_volume = 0.0
        volume_inside_height = 0.0

        for i, obj_id in enumerate(placed_ids):
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            x, y, z = pos

            max_z = max(max_z, z)

            # Add to total volume
            total_volume += volumes[i]

            volume_inside = compute_clipped_volume_transformed(placed_meshes[i], pos, rot, WALL_HEIGHT)
            volume_inside_height = volume_inside + volume_inside_height

            # Check if the top of the object is below box height
            #top_z = z + object_heights[i]/2
            #if top_z < height:
            #    volume_inside_height += volumes[i]

        results.append({
            "run": run + 1,
            "max_z": round(max_z, 3),
            "total_volume": total_volume,
            "inside_volume": volume_inside_height,
            "objects_placed": len(placed_ids)
        })

        current_result = {
            "run": run + 1,
            "max_z": round(max_z, 3),
            "total_volume": total_volume,
            "inside_volume": volume_inside_height,
            "objects_placed": len(placed_ids)
        }
        results.append(current_result)

        # Update best result if needed
        if (best_result is None or 
            current_result["objects_placed"] > best_result["objects_placed"] or
            (current_result["objects_placed"] == best_result["objects_placed"])):
            best_result = current_result


        # Render image from camera
        width_px, height_px = 640, 480
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.1, 0, 2],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width_px/height_px, nearVal=0.1, farVal=10
        )
        #p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])
        img_arr = p.getCameraImage(
            width=width_px,
            height=height_px,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Extract and save RGB image
        rgb_array = img_arr[2]
        image = Image.fromarray(rgb_array)
        os.makedirs("run_images", exist_ok=True)
        image.save(f"run_images/run_{run+1:02d}.png")
        image.save(f"run_images/best_run_{best_result['run']:02d}.png")

        p.disconnect()

    # Output results
    for result in results:
        print(f"Run {result['run']} | Placed: {result['objects_placed']}")

    print("\nBest Result:")
    print(f"Run {best_result['run']} | Placed: {result['objects_placed']} | Total Volume: {best_result['total_volume']} | Inside Volume: {best_result['inside_volume']}")

    return best_result['run'],result['objects_placed'],best_result['max_z']



def pack_meshes_min_xy(mesh_list, mesh_scale=MESH_SCALE_DEFAULT):
    placed_ids = []
    rotations = generate_rotations()
    for mesh_path in mesh_list:
        best_rotation = None
        min_area = float('inf')
        best_mesh = None
        best_extents = None
        best_center_offset = None

        # Evaluate each rotation
        for rot in rotations:
            transform = np.eye(4)
            transform[:3, :3] = rot
            area, mesh = get_xy_footprint(mesh_path, mesh_scale, transform)
            if area < min_area:
                min_area = area
                best_rotation = transform
                best_mesh = mesh

        # Export best mesh as temp STL to load in PyBullet
        temp_path = "temp_rotated_mesh.stl"
        best_mesh.export(temp_path)
        half_extents, center_offset = get_mesh_half_extents(temp_path, mesh_scale)

        x_min = -BIN_SIZE + WALL_THICKNESS + half_extents[0] + MARGIN
        x_max =  BIN_SIZE - WALL_THICKNESS - half_extents[0] - MARGIN
        y_min = -BIN_SIZE + WALL_THICKNESS + half_extents[1] + MARGIN
        y_max =  BIN_SIZE - WALL_THICKNESS - half_extents[1] - MARGIN

        placed_successfully = False
        for x in frange(x_min, x_max, STEP):
            for y in frange(y_min, y_max, STEP):
                z = half_extents[2]  # base layer
                pos = [x, y, z]
                if not mesh_collision_check(pos, temp_path, mesh_scale, center_offset, placed_ids):
                    new_id = spawn_mesh(pos, temp_path, mesh_scale, center_offset)
                    placed_ids.append(new_id)
                    placed_successfully = True
                    break
            if placed_successfully:
                break

        if not placed_successfully:
            print(f"❌ Unable to place mesh: {os.path.basename(mesh_path)}")

def pack_meshes_with_stability(mesh_list, mesh_scale=MESH_SCALE_DEFAULT):
    placed_ids = []
    temp_mesh_path = "temp_rotated_mesh.stl"

    for mesh_path in mesh_list:
        print(f"\n🔍 Trying to place: {os.path.basename(mesh_path)}")
        mesh = trimesh.load_mesh(mesh_path)
        rotations = generate_rotations()

        best_transform = None
        best_area = float('inf')
        best_center_offset = None

        for i, rot in enumerate(rotations):
            transform = np.eye(4)
            transform[:3, :3] = rot

            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(transform)

            # Check static stability: base is flat + COM near bottom
            z_faces = np.where(mesh_copy.face_normals[:, 2] > 0.50)[0]
            com = mesh_copy.center_mass
            min_z = mesh_copy.bounds[0][2]
            if len(z_faces) == 0:
                continue
            if (com[2] - min_z) > 0.05:
                continue

            # Export mesh in this orientation
            mesh_copy.export(temp_mesh_path)

            # Check dynamic stability
            half_extents, center_offset = get_mesh_half_extents(temp_mesh_path, mesh_scale)
            if not is_physically_stable_in_pybullet(temp_mesh_path, mesh_scale, np.eye(4), center_offset):
                continue

            # Calculate XY area for sorting
            bounds = mesh_copy.bounds
            area = (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1])

            if area < best_area:
                best_area = area
                best_transform = transform
                best_center_offset = center_offset

        if best_transform is None:
            print(f"❌ No stable orientation found for {os.path.basename(mesh_path)}")
            continue

        # Export best mesh version
        mesh.apply_transform(best_transform)
        mesh.export(temp_mesh_path)
        half_extents, center_offset = get_mesh_half_extents(temp_mesh_path, mesh_scale)

        x_min = -BIN_SIZE + WALL_THICKNESS + half_extents[0] + MARGIN
        x_max =  BIN_SIZE - WALL_THICKNESS - half_extents[0] - MARGIN
        y_min = -BIN_SIZE + WALL_THICKNESS + half_extents[1] + MARGIN
        y_max =  BIN_SIZE - WALL_THICKNESS - half_extents[1] - MARGIN

        placed = False
        for x in frange(x_min, x_max, STEP):
            for y in frange(y_min, y_max, STEP):
                z = half_extents[2]
                pos = [x, y, z]
                if not mesh_collision_check(pos, temp_mesh_path, mesh_scale, center_offset, placed_ids):
                    obj_id = spawn_mesh(pos, temp_mesh_path, mesh_scale, center_offset)
                    placed_ids.append(obj_id)
                    print(f"✅ Placed at: {pos}")
                    placed = True
                    break
            if placed:
                break

        if not placed:
            print(f"❌ Unable to place mesh: {os.path.basename(mesh_path)}")

def load_stl_model(path, scale=10.0):
    mesh = trimesh.load_mesh(path)
    mesh.apply_scale(scale)
    return mesh

def all_objects_in_bin(object_ids):
    x_bounds = [-BIN_LENGTH / 2, BIN_LENGTH / 2]
    y_bounds = [-BIN_WIDTH / 2, BIN_WIDTH / 2]
    z_bounds = [0, MAX_HEIGHT]

    for obj_id in object_ids:
        aabb_min, aabb_max = p.getAABB(obj_id)

        if (aabb_min[0] < x_bounds[0] or aabb_max[0] > x_bounds[1] or
            aabb_min[1] < y_bounds[0] or aabb_max[1] > y_bounds[1] or
            aabb_min[2] < z_bounds[0] or aabb_max[2] > z_bounds[1]):
            print(f"❌ Object {obj_id} out of bounds")
            return False

    print("✅ All objects are within the bin.")
    return True


def sort_rotations(mesh,stable_rots):
    # List to hold (rotation, area) pairs
    rotation_area_list = []

    for rot in stable_rots:
        # Create mesh copy
        m = mesh.copy()

        # Get transform
        if isinstance(rot, list):  # Euler
            T = trimesh.transformations.euler_matrix(*rot, axes='sxyz')
        else:  # Rotation matrix
            T = np.eye(4)
            T[:3, :3] = rot

        m.apply_transform(T)

        # Compute area
        area = xy_footprint_area(m)

        # Store rotation and its area
        rotation_area_list.append((rot, area))

    # Sort the list by area
    rotation_area_list.sort(key=lambda x: x[1])  # sort by area
    #print(rotation_area_list)

    return rotation_area_list




# === Load STL Meshes ===
mesh_dir = "new_mesh/set1"
mesh_files = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith(".stl")]
# Load meshes and compute volume
mesh_info = []
for f in mesh_files:
    #print(f)
    stable_rots = pybullet_stability_check_rotations(f)
    #print("Stable rotations:", stable_rots)
    #mesh = load_stl_model(f, scale=10)
    mesh = trimesh.load_mesh(f)
    area_rots = sort_rotations(mesh,stable_rots)
    surface_areas, bb_areas, sorted_rotations = compare_voxels_rotations(f,pitch=VOXEL_PITCH)
    #mesh.apply_scale(100)
    volume = mesh.volume  # Automatically computed by trimesh
    mesh_info.append((f,volume,sorted_rotations))
    #print(mesh_info)

# Sort meshes by volume (descending for packing larger ones first)
sorted_meshes = sorted(mesh_info, key=lambda x: x[1], reverse=True) # Larger -> Smaller
#sorted_meshes = sorted(mesh_info, key=lambda x: x[1]) # Smaller -> Larger
#sorted_mesh_files = [f for f,_,_ in sorted_meshes]
#print(sorted_mesh_files)


#test = mesh_files[4]
#stable_rots = pybullet_stability_check_rotations(test)
#area_rots = sort_rotations(mesh,stable_rots)
#print("AREA ROT:", area_rots)
#mesh = trimesh.load_mesh(test)
def plot_rotations(mesh):
    # Plot setup
    fig = plt.figure(figsize=(18, 8))
    cols = 3
    rows = int(np.ceil(len(ROTATION_CANDIDATES) / cols))

    for i, euler in enumerate(ROTATION_CANDIDATES):
        rotated_mesh = mesh.copy()
        R = euler_to_matrix(euler)
        rotated_mesh.apply_transform(R)

        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.add_collection3d(Poly3DCollection(rotated_mesh.triangles, alpha=0.5, facecolor='gray'))
        ax.auto_scale_xyz(*rotated_mesh.bounds.T)
        set_axes_equal(ax) 
        label = "Stable" if euler in stable_rots else "Unstable"
        ax.set_title(f"{label}\n{np.round(euler, 2)}")
        #ax.set_axis_off()

    plt.tight_layout()
    plt.show()

#plot_rotations(mesh)


# Camera setup
camera_target_position = [0, 0, 0]
camera_distance = 4.0
camera_yaw = 0
camera_pitch = -89.9 #60
camera_roll = 0
up_axis_index = 2
aspect_ratio = 1.0

view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=camera_target_position,
    distance=camera_distance,
    yaw=camera_yaw,
    pitch=camera_pitch,
    roll=camera_roll,
    upAxisIndex=up_axis_index
)

projection_matrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=aspect_ratio,
    nearVal=0.01,
    farVal=100.0
)

width, height = 640, 480

def capture_image(step_number):
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        shadow=True,
        lightDirection=[1, 1, 1]
    )
    rgb_array = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    image = Image.fromarray(rgb_array)
    image.save(f"packing_step_{step_number}.png")


def run_random():
    start_time = time.time()
    run, placed, max_z = small_pack_meshes_randomly(sorted_meshes)
    if placed != len(sorted_meshes):
        run, placed, max_z = large_pack_meshes_randomly(sorted_meshes)
        print("Packed everything in LARGER box")
        if placed != len(sorted_meshes):
            print("Packed everything in LARGER box")
    else:
        print("Packed everything in SMALLER box")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("TIME:",elapsed_time)


BIN_LENGTH = 0.88 #1.27  # along X-axis
BIN_WIDTH = 1.27 #0.88   # along Y-axis
def run_main():
    # === PyBullet Setup ===
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # === Run the Packing ===
    # === Bin Configuration ===
    #BIN_SIZE = 1.27
    BIN_LENGTH = 0.88 #1.27  # along X-axis
    BIN_WIDTH = 1.27 #0.88   # along Y-axis

    start_time = time.time()
    bin_id = create_shakable_bin()
    #for i in range(10):
    placed_ids, not_placed_ids = pack_meshes(sorted_meshes)
    print(len(mesh_files),len(placed_ids),len(not_placed_ids))

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    if (len(mesh_files) != len(placed_ids)):
        p.disconnect()
        # === PyBullet Setup ===
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        BIN_LENGTH = 1.27  # along X-axis
        BIN_WIDTH = 1.27   # along Y-axis
        bin_id = create_shakable_bin()
        placed_ids, not_placed_ids = pack_meshes(sorted_meshes)
        print(len(mesh_files),len(placed_ids),len(not_placed_ids))

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("TIME:",elapsed_time)
    else:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("TIME:",elapsed_time)



    #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    #pack_meshes_with_stability(sorted_mesh_files)
    #shake_bin_rigidbody(bin_id)


    # === After Packing, Start Box Shake ===
    # Periodically apply the shake effect after packing
    #shaking_interval = 5  # Shake every 5 seconds
    #last_shake_time = time.time()

    # === Camera & Simulation Loop ===
    p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])
    while True:
        p.stepSimulation()
        #time.sleep(1 / 240)

#run_random(); print("DONE RAN")
run_main()
















'''
import pybullet as p
import pybullet_data
import time
import os

# === Simulation setup ===
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")

# === Bin configuration ===
BIN_SIZE = 1.27
WALL_THICKNESS = 0.02
WALL_HEIGHT = 0.63
MAX_HEIGHT = 0.5
STEP = 0.015
Z_STEP = 0.01
MESH_SCALE = 0.01  # Uniform scale for all meshes

# === Create bin walls ===
walls = [
    ([BIN_SIZE, WALL_THICKNESS, WALL_HEIGHT], [0, BIN_SIZE, WALL_HEIGHT]),
    ([BIN_SIZE, WALL_THICKNESS, WALL_HEIGHT], [0, -BIN_SIZE, WALL_HEIGHT]),
    ([WALL_THICKNESS, BIN_SIZE, WALL_HEIGHT], [BIN_SIZE, 0, WALL_HEIGHT]),
    ([WALL_THICKNESS, BIN_SIZE, WALL_HEIGHT], [-BIN_SIZE, 0, WALL_HEIGHT]),
]
for size, pos in walls:
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=pos)

# === Load STL meshes ===
mesh_dir = "our_mesh/set1"
mesh_files = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith(".stl")]

# === Helpers ===
def frange(start, stop, step):
    while start < stop:
        yield start
        start += step

def get_mesh_half_extents(mesh_path, scale=1.0):
    col_id = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3)
    vis_id = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3)
    temp_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id,
                                  baseVisualShapeIndex=vis_id, basePosition=[0, 0, 0])
    aabb_min, aabb_max = p.getAABB(temp_body)
    p.removeBody(temp_body)
    half_extents = [(max_val - min_val) / 2 for min_val, max_val in zip(aabb_min, aabb_max)]
    return half_extents

def does_overlap(pos, size, placed):
    x1_min, x1_max = pos[0] - size[0], pos[0] + size[0]
    y1_min, y1_max = pos[1] - size[1], pos[1] + size[1]
    z1_min, z1_max = pos[2] - size[2], pos[2] + size[2]
    for obj_pos, obj_size in placed:
        x2_min, x2_max = obj_pos[0] - obj_size[0], obj_pos[0] + obj_size[0]
        y2_min, y2_max = obj_pos[1] - obj_size[1], obj_pos[1] + obj_size[1]
        z2_min, z2_max = obj_pos[2] - obj_size[2], obj_pos[2] + obj_size[2]
        if not (x1_max <= x2_min or x1_min >= x2_max or
                y1_max <= y2_min or y1_min >= y2_max or
                z1_max <= z2_min or z1_min >= z2_max):
            return True
    return False

def spawn_mesh(position, mesh_path, scale=1.0):
    col_id = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3)
    vis_id = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id,
                             baseVisualShapeIndex=vis_id, basePosition=position)

# === Main bin packing function ===
def pack_meshes(mesh_list):
    placed = []

    for mesh_path in mesh_list:
        half_extents = get_mesh_half_extents(mesh_path, MESH_SCALE)

        # Adjust bounds to stay inside the bin walls + add margin
        margin = 0.001
        x_min = -BIN_SIZE + WALL_THICKNESS + half_extents[0] + margin
        x_max =  BIN_SIZE - WALL_THICKNESS - half_extents[0] - margin
        y_min = -BIN_SIZE + WALL_THICKNESS + half_extents[1] + margin
        y_max =  BIN_SIZE - WALL_THICKNESS - half_extents[1] - margin

        placed_successfully = False
        for z in frange(half_extents[2], MAX_HEIGHT - half_extents[2], Z_STEP):
            for x in frange(x_min, x_max, STEP):
                for y in frange(y_min, y_max, STEP):
                    pos = [x, y, z]
                    if not does_overlap(pos, half_extents, placed):
                        spawn_mesh(pos, mesh_path, MESH_SCALE)
                        placed.append((pos, half_extents))
                        placed_successfully = True
                        break
                if placed_successfully:
                    break
            if placed_successfully:
                break
        if not placed_successfully:
            print(f"Unable to place mesh: {mesh_path}")

# === Run the packing ===
pack_meshes(mesh_files)  # Repeat to increase number

# === Camera and simulation loop ===
p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])
while True:
    p.stepSimulation()
    time.sleep(1 / 240)
'''



'''
# Create bin walls
walls = [
    ([BIN_SIZE, WALL_THICKNESS, WALL_HEIGHT], [0, BIN_SIZE, WALL_HEIGHT]),
    ([BIN_SIZE, WALL_THICKNESS, WALL_HEIGHT], [0, -BIN_SIZE, WALL_HEIGHT]),
    ([WALL_THICKNESS, BIN_SIZE, WALL_HEIGHT], [BIN_SIZE, 0, WALL_HEIGHT]),
    ([WALL_THICKNESS, BIN_SIZE, WALL_HEIGHT], [-BIN_SIZE, 0, WALL_HEIGHT]),
]
for size, pos in walls:
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=pos)

# Mesh-based object spawning
def spawn_static_mesh(position, mesh_scale, stl_path):
    if not os.path.exists(stl_path):
        print(f"STL not found: {stl_path}")
        return None
    col_id = p.createCollisionShape(p.GEOM_MESH, fileName=stl_path, meshScale=mesh_scale)
    vis_id = p.createVisualShape(p.GEOM_MESH, fileName=stl_path, meshScale=mesh_scale)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id,
                             baseVisualShapeIndex=vis_id, basePosition=position)

# Overlap check
def does_overlap(pos, size, placed_objects):
    x1_min, x1_max = pos[0] - size[0], pos[0] + size[0]
    y1_min, y1_max = pos[1] - size[1], pos[1] + size[1]
    z1_min, z1_max = pos[2] - size[2], pos[2] + size[2]

    for obj_pos, obj_size in placed_objects:
        x2_min, x2_max = obj_pos[0] - obj_size[0], obj_pos[0] + obj_size[0]
        y2_min, y2_max = obj_pos[1] - obj_size[1], obj_pos[1] + obj_size[1]
        z2_min, z2_max = obj_pos[2] - obj_size[2], obj_pos[2] + obj_size[2]

        if not (x1_max <= x2_min or x1_min >= x2_max or
                y1_max <= y2_min or y1_min >= y2_max or
                z1_max <= z2_min or z1_min >= z2_max):
            return True
    return False

# Stability check (optional)
def is_stable(pos, size, placed_objects, min_support_ratio=0.5):
    x_min, x_max = pos[0] - size[0], pos[0] + size[0]
    y_min, y_max = pos[1] - size[1], pos[1] + size[1]
    z_bottom = pos[2] - size[2]
    support_area = 0.0
    box_area = (x_max - x_min) * (y_max - y_min)

    for obj_pos, obj_size in placed_objects:
        z_top = obj_pos[2] + obj_size[2]
        if abs(z_top - z_bottom) > 0.005:
            continue

        x2_min, x2_max = obj_pos[0] - obj_size[0], obj_pos[0] + obj_size[0]
        y2_min, y2_max = obj_pos[1] - obj_size[1], obj_pos[1] + obj_size[1]

        overlap_x = max(0, min(x_max, x2_max) - max(x_min, x2_min))
        overlap_y = max(0, min(y_max, y2_max) - max(y_min, y2_min))
        support_area += overlap_x * overlap_y

    return support_area >= min_support_ratio * box_area

# Float range generator
def frange(start, stop, step):
    while start < stop:
        yield start
        start += step

# 3D Greedy bin packing with STL meshes
def greedy_bin_packing_3d(num_boxes, min_support_ratio=0.5):
    placed_objects = []

    for i in range(num_boxes):
        size = [
            random.uniform(0.03, 0.045),
            random.uniform(0.03, 0.045),
            random.uniform(0.025, 0.035)
        ]

        mesh_index = i % len(mesh_files)
        mesh_path = mesh_files[mesh_index]
        scale_factor = 0.0001
        mesh_scale = [s * scale_factor for s in size]  # STL unit-size mesh assumed

        placed = False
        for z in frange(size[2], MAX_HEIGHT - size[2], Z_STEP):
            for x in frange(-BIN_SIZE + size[0], BIN_SIZE - size[0], PLACEMENT_STEP):
                for y in frange(-BIN_SIZE + size[1], BIN_SIZE - size[1], PLACEMENT_STEP):
                    pos = [x, y, z]
                    if not does_overlap(pos, size, placed_objects):
                        if abs(z - size[2]) < 1e-3 or is_stable(pos, size, placed_objects, min_support_ratio):
                            spawn_static_mesh(pos, mesh_scale, mesh_path)
                            placed_objects.append((pos, size))
                            placed = True
                            break
                if placed:
                    break
            if placed:
                break

        if not placed:
            print(f"Box {i+1}: No available space!")

# Run simulation
greedy_bin_packing_3d(num_boxes=30, min_support_ratio=0.4)

# Top-down camera
p.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=0,
    cameraPitch=-89.9,
    cameraTargetPosition=[0, 0, 0]
)

# Simulation loop
while True:
    p.stepSimulation()
    time.sleep(1 / 240)
'''


'''
import pybullet as p
import pybullet_data
import random
import time
import os 

# Setup PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)  # Gravity enabled

# Bin dimensions
BIN_SIZE = 0.25
WALL_THICKNESS = 0.02
WALL_HEIGHT = 0.1#2
MAX_HEIGHT = 0.5
Z_STEP = 0.01  # Height scan step

# Use flat plane as the floor at z = 0
p.loadURDF("plane.urdf")

# Create bin walls
walls = [
    ([BIN_SIZE, WALL_THICKNESS, WALL_HEIGHT], [0, BIN_SIZE, WALL_HEIGHT]),
    ([BIN_SIZE, WALL_THICKNESS, WALL_HEIGHT], [0, -BIN_SIZE, WALL_HEIGHT]),
    ([WALL_THICKNESS, BIN_SIZE, WALL_HEIGHT], [BIN_SIZE, 0, WALL_HEIGHT]),
    ([WALL_THICKNESS, BIN_SIZE, WALL_HEIGHT], [-BIN_SIZE, 0, WALL_HEIGHT]),
]

for size, pos in walls:
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=pos)

# Function to spawn a box and freeze it
def spawn_static_box(position, size):
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=shape, basePosition=position)
    
    # Freeze the box so it doesn't fall or react to gravity
    p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])
    p.changeDynamics(body_id, -1, mass=0)  # Make static

    return body_id

def spawn_static_mesh(position, mesh_scale=[1,1,1], stl_path):
    if not os.path.exists(stl_path):
        print(f"Error: STL file '{stl_path}' not found.")
        return None

    col_id = p.createCollisionShape(p.GEOM_MESH, fileName=stl_path, meshScale=mesh_scale)
    vis_id = p.createVisualShape(p.GEOM_MESH, fileName=stl_path, meshScale=mesh_scale)

    body_id = p.createMultiBody(
        baseMass=0,  # static
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=position
    )
    return body_id

# Check for overlap
def does_overlap(pos, size, placed_objects):
    x1_min, x1_max = pos[0] - size[0], pos[0] + size[0]
    y1_min, y1_max = pos[1] - size[1], pos[1] + size[1]
    z1_min, z1_max = pos[2] - size[2], pos[2] + size[2]

    for obj_pos, obj_size in placed_objects:
        x2_min, x2_max = obj_pos[0] - obj_size[0], obj_pos[0] + obj_size[0]
        y2_min, y2_max = obj_pos[1] - obj_size[1], obj_pos[1] + obj_size[1]
        z2_min, z2_max = obj_pos[2] - obj_size[2], obj_pos[2] + obj_size[2]

        if not (x1_max <= x2_min or x1_min >= x2_max or 
                y1_max <= y2_min or y1_min >= y2_max or
                z1_max <= z2_min or z1_min >= z2_max):
            return True
    return False

# Partial support check (optional)
def is_stable(pos, size, placed_objects, min_support_ratio=0.5):
    x_min, x_max = pos[0] - size[0], pos[0] + size[0]
    y_min, y_max = pos[1] - size[1], pos[1] + size[1]
    z_bottom = pos[2] - size[2]
    support_area = 0.0
    box_area = (x_max - x_min) * (y_max - y_min)

    for obj_pos, obj_size in placed_objects:
        z_top = obj_pos[2] + obj_size[2]
        if abs(z_top - z_bottom) > 0.005:
            continue

        x2_min, x2_max = obj_pos[0] - obj_size[0], obj_pos[0] + obj_size[0]
        y2_min, y2_max = obj_pos[1] - obj_size[1], obj_pos[1] + obj_size[1]

        overlap_x = max(0, min(x_max, x2_max) - max(x_min, x2_min))
        overlap_y = max(0, min(y_max, y2_max) - max(y_min, y2_min))
        support_area += overlap_x * overlap_y

    return support_area >= min_support_ratio * box_area

# Float range
def frange(start, stop, step):
    while start < stop:
        yield start
        start += step

# Greedy placement
def greedy_bin_packing(num_boxes, min_support_ratio):
    placed_objects = []
    step = 0.01  # granularity of scan

    for i in range(num_boxes):
        size = [
            random.uniform(0.03, 0.06),  # x half-extent
            random.uniform(0.03, 0.06),  # y half-extent
            random.uniform(0.02, 0.04) #0.025  # z half-extent
        ]
        z = size[2]  # box sits on floor at z=0

        placed = False
        for z in frange(size[2], MAX_HEIGHT - size[2], Z_STEP):
            for x in frange(-BIN_SIZE + size[0], BIN_SIZE - size[0], step):
                for y in frange(-BIN_SIZE + size[1], BIN_SIZE - size[1], step):
                    pos = [x, y, z]
                    if not does_overlap(pos, size, placed_objects): # and is_stable(pos, size, placed_objects, min_support_ratio):
                        spawn_static_box(pos, size)
                        placed_objects.append((pos, size))
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break

        if not placed:
            print(f"Box {i+1}: No available space!")

# Run it
greedy_bin_packing(num_boxes=30, min_support_ratio=0.5)

mesh_index = i % len(mesh_files)  # cycle through meshes
mesh_path = mesh_files[mesh_index]
mesh_scale = [size[0]*2, size[1]*2, size[2]*2]  # STL unit cube scaling
spawn_static_mesh(pos, mesh_scale, mesh_path)


# Top-down camera
p.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=0,
    cameraPitch=-89.9,
    cameraTargetPosition=[0, 0, 0]
)

# Run simulation
while True:
    p.stepSimulation()
    time.sleep(1 / 240)
'''

'''
import pybullet as p
import pybullet_data
import random
import time

# Connect to the PyBullet GUI
p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load plane and create the bin using walls and a base
plane_id = p.loadURDF("plane.urdf")

bin_size = 0.5  # Size of the bin
wall_thickness = 0.02
wall_height = 0.2

# Create the bin (base and walls)
bin_base = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[bin_size, bin_size, wall_thickness]),
    basePosition=[0, 0, wall_thickness]
)

# Define the walls for the bin
walls = [
    ([bin_size, wall_thickness, wall_height], [0, bin_size, wall_height + wall_thickness]),
    ([bin_size, wall_thickness, wall_height], [0, -bin_size, wall_height + wall_thickness]),
    ([wall_thickness, bin_size, wall_height], [bin_size, 0, wall_height + wall_thickness]),
    ([wall_thickness, bin_size, wall_height], [-bin_size, 0, wall_height + wall_thickness]),
]

for size, pos in walls:
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=pos)

# Function to spawn a box
def spawn_box(pos, size=(0.05, 0.05, 0.05)):
    colShape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colShape, basePosition=pos)
    return body

# Function to check if the box fits at the given position
def fits_at_position(objects, box_size, pos):
    """Check if the box can be placed at the given position without overlapping any other objects."""
    # Check overlap with previously placed objects
    for obj_pos, obj_size in objects:
        if (pos[0] < obj_pos[0] + obj_size[0] and
            pos[0] + box_size[0] > obj_pos[0] and
            pos[1] < obj_pos[1] + obj_size[1] and
            pos[1] + box_size[1] > obj_pos[1]):
            return False
    return True

# Function to place boxes in rows starting from the bottom-left corner
def place_boxes(objects, num_objects, box_size):
    placed_objects = []
    pos_x = -bin_size + box_size[0]  # Start from the left edge
    pos_y = -bin_size + box_size[1]  # Start from the bottom edge

    for i in range(num_objects):
        # Find a spot to place the box
        if fits_at_position(placed_objects, box_size, [pos_x, pos_y, 0.5]):
            # Spawn the box at the position
            spawn_box([pos_x, pos_y, 0.5], box_size)
            placed_objects.append(([pos_x, pos_y, 0.5], box_size))
        
        # Move to the next position in the row
        pos_x += box_size[0]  # Move horizontally
        
        # If we reach the end of the row, move to the next row
        if pos_x + box_size[0] > bin_size:
            pos_x = -bin_size + box_size[0]  # Reset to the left edge
            pos_y += box_size[1]  # Move up vertically
            
            # If we reach the top, stop placing boxes
            if pos_y + box_size[1] > bin_size:
                print(f"Unable to place box {i+1}. No available space left!")
                break

    return placed_objects

# Number of objects to place
num_objects = 10
box_size = (random.uniform(0.05, 0.1), random.uniform(0.05, 0.1), 0.05)  # Random box sizes

# Start placing boxes from the corner
placed_objects = place_boxes([], num_objects, box_size)

# Set top-down camera
p.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=0,
    cameraPitch=-89.9,
    cameraTargetPosition=[0, 0, 0]
)

# Run the simulation
while True:
    p.stepSimulation()
    time.sleep(1/240)

'''

'''
import pybullet as p
import pybullet_data
import time
import random

# Connect to GUI
p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load plane and bin (use a simple box as bin)
plane_id = p.loadURDF("plane.urdf")

# Create a bin using 4 walls and a base
bin_size = 0.5
wall_thickness = 0.02
wall_height = 0.2

def create_wall(pos, size):
    return p.createCollisionShape(p.GEOM_BOX, halfExtents=size), pos

bin_base = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[bin_size, bin_size, wall_thickness]),
    basePosition=[0, 0, wall_thickness]
)

# Walls
walls = [
    ([bin_size, wall_thickness, wall_height], [0, bin_size, wall_height + wall_thickness]),
    ([bin_size, wall_thickness, wall_height], [0, -bin_size, wall_height + wall_thickness]),
    ([wall_thickness, bin_size, wall_height], [bin_size, 0, wall_height + wall_thickness]),
    ([wall_thickness, bin_size, wall_height], [-bin_size, 0, wall_height + wall_thickness]),
]

for size, pos in walls:
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=pos)

# Function to spawn a box
def spawn_box(pos, size=(0.05, 0.05, 0.05)):
    colShape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colShape, basePosition=pos)
    return body

# Drop random boxes inside the bin
for i in range(10):
    x = random.uniform(-0.4, 0.4)
    y = random.uniform(-0.4, 0.4)
    spawn_box(pos=[x, y, 0.5])
    time.sleep(0.1)

# Set top-down camera
p.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=0,
    cameraPitch=-89.9,
    cameraTargetPosition=[0, 0, 0]
)

# Run the simulation
while True:
    p.stepSimulation()
    time.sleep(1/240)
'''