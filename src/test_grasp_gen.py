#!/usr/bin/env python3

import rospy
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from robot_manipulation.srv import GenerateGraspCandidates, GenerateGraspCandidatesRequest, GenerateGraspCandidatesResponse
import struct
from numpy_ros import to_numpy, to_message
import sys
import copy
import moveit_commander
import moveit_msgs.msg
from math import pi
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from moveit_commander.conversions import pose_to_list
import time
from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import RobotState, Constraints, PositionConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from interbotix_xs_modules.arm import InterbotixManipulatorXS


OBJECT_POSE_GLOBAL = np.array([[1, 0, 0, 0.3],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0.07],
                               [0, 0, 0, 1]])

class MoveItInterface:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander(robot_description="wx250s/robot_description", ns="wx250s")
        self.scene = moveit_commander.PlanningSceneInterface(ns="wx250s")
        self.move_group = moveit_commander.MoveGroupCommander("interbotix_arm", robot_description="wx250s/robot_description", ns="wx250s", wait_for_servers=10.0)
        self.display_trajectory_publisher = rospy.Publisher("move_group/display_planned_path",
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)
    
    def display_trajectory(self, plan):
        trajectory = moveit_msgs.msg.DisplayTrajectory()
        trajectory.trajectory_start = self.robot.get_current_state()
        trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(trajectory)


# Function to convert Open3D PointCloud to ROS PointCloud2
def convert_open3d_to_ros_pointcloud2(open3d_cloud, frame_id="base_link"):
    """
    Convert an Open3D PointCloud object to a ROS PointCloud2 message.

    Args:
        open3d_cloud (open3d.geometry.PointCloud): The Open3D point cloud.
        frame_id (str): The frame ID for the ROS message header.

    Returns:
        sensor_msgs.msg.PointCloud2: The ROS PointCloud2 message.
    """
    # Check if the Open3D point cloud is empty
    if not open3d_cloud.has_points():
        rospy.logwarn("Open3D point cloud is empty.")
        return None

    # Get points from the Open3D point cloud
    points = np.asarray(open3d_cloud.points)

    # Define the fields for the PointCloud2 message (x, y, z)
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]

    # Correct calculation for point_step based on datatype size
    datatype_sizes = {
        PointField.INT8: 1, PointField.UINT8: 1,
        PointField.INT16: 2, PointField.UINT16: 2,
        PointField.INT32: 4, PointField.UINT32: 4,
        PointField.FLOAT32: 4, PointField.FLOAT64: 8
    }
    point_step = sum([datatype_sizes[f.datatype] * f.count for f in fields])

    # Create the PointCloud2 message
    ros_cloud = PointCloud2()
    ros_cloud.header.stamp = rospy.Time.now()
    ros_cloud.header.frame_id = frame_id
    ros_cloud.height = 1  # Unorganized point cloud
    ros_cloud.width = len(points) # Number of points
    ros_cloud.fields = fields
    ros_cloud.is_bigendian = False # Assuming little-endian system
    ros_cloud.point_step = point_step
    ros_cloud.row_step = ros_cloud.point_step * ros_cloud.width
    ros_cloud.is_dense = True # Assuming no invalid points

    # Pack the point data into a byte array
    buffer = [struct.pack('fff', x, y, z) for x, y, z in points]
    ros_cloud.data = b''.join(buffer)

    return ros_cloud

def main():
    # only used for gripper
    rospy.init_node("test_grasp_gen")
    moveit = MoveItInterface()

    # --- Parameters ---
    # Replace with the actual path to your STL file
    stl_file_path = rospy.get_param('~stl_file', 'path/to/your/model.stl')
    # Replace with the desired number of points to sample from the mesh
    point_density_m2 = rospy.get_param('~point_density', 100000)
    # Replace with the name of the ROS service you want to call
    service_name = rospy.get_param('~service_name', '/your_service_name')
    # Replace with the actual type of the ROS service
    # Example: from your_package.srv import YourServiceType
    # For this example, we'll use std_srvs/Trigger as a placeholder
    service_type = GenerateGraspCandidates # Replace with your actual service type

    rospy.loginfo(f"Loading STL file: {stl_file_path}")

    # --- Load STL file using Open3D ---
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    num_points_to_sample = int(point_density_m2 * mesh.get_surface_area())

    # --- Convert mesh to point cloud ---
    rospy.loginfo(f"Sampling {num_points_to_sample} points from the mesh.")
    point_cloud_o3d = mesh.sample_points_uniformly(number_of_points=num_points_to_sample)

    # --- Convert Open3D point cloud to ROS PointCloud2 ---
    rospy.loginfo("Converting Open3D point cloud to ROS PointCloud2.")
    ros_point_cloud = convert_open3d_to_ros_pointcloud2(point_cloud_o3d)

    if ros_point_cloud is None:
        rospy.logerr("Failed to convert Open3D point cloud to ROS PointCloud2.")
        return

    rospy.loginfo("PointCloud2 message created successfully.")

    # --- Call the ROS service ---
    rospy.loginfo(f"Waiting for service '{service_name}'...")

    # Wait for the service to be available
    rospy.wait_for_service(service_name)
    rospy.loginfo(f"Service '{service_name}' is available.")

    # Create a service proxy
    service_proxy = rospy.ServiceProxy(service_name, service_type)

    # Create the service request
    # NOTE: The request structure depends on your service type.
    # If your service takes a PointCloud2, the request might look like this:
    # request = YourServiceTypeRequest()
    # request.cloud = ros_point_cloud
    # If using std_srvs/Trigger (as in this example), the request is empty:
    request = GenerateGraspCandidatesRequest(ros_point_cloud) # Replace with your actual request type

    # Call the service
    rospy.loginfo(f"Calling service '{service_name}'...")
    # NOTE: If your service takes a PointCloud2, pass it as an argument:
    # response = service_proxy(ros_point_cloud) # If service takes PointCloud2 directly
    # If using a request object:
    response: GenerateGraspCandidatesResponse = service_proxy(request) # Replace with your actual service call

    # bot.gripper.open()

    constraints = Constraints()
    position_constraint = PositionConstraint()
    bounding_volume = BoundingVolume()
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [1.0, 0.6, 0.7]
    box_pose = Pose()
    box_pose.position.x = 0.25
    box_pose.position.y = 0
    box_pose.position.z = 0.38
    bounding_volume.primitives = [box]
    bounding_volume.primitive_poses = [box_pose]
    position_constraint.link_name = "wx250s/ee_gripper_link"
    position_constraint.constraint_region = bounding_volume
    position_constraint.target_point_offset.x = 0
    position_constraint.target_point_offset.y = 0
    position_constraint.target_point_offset.z = 0
    position_constraint.weight = 1.0
    position_constraint.header.frame_id = 'wx250s/base_link'
    constraints.name = "workspace"
    constraints.position_constraints = [position_constraint]
    moveit.move_group.set_path_constraints(constraints)

    for candidate in sorted(response.candidates.candidates, key=lambda x: x.score, reverse=True):
        approach = to_numpy(candidate.approach)
        binormal = to_numpy(candidate.binormal)
        axis = to_numpy(candidate.axis)

        gripper_rot = np.column_stack([approach, binormal, axis])
        gripper_t = to_numpy(candidate.position)
        pose = np.block([[gripper_rot, gripper_t.reshape((-1, 1))], [np.zeros(3), 1]])
        pose_global = OBJECT_POSE_GLOBAL @ pose
        pose_approach = OBJECT_POSE_GLOBAL @ np.block([[np.eye(3), (-0.05 * approach).reshape((-1, 1))], [np.zeros(3), 1]]) @ pose

        pose_approach_ros = to_message(Pose, pose_approach)
        pose_global_ros = to_message(Pose, pose_global)
        waypoints = [pose_approach_ros, pose_global_ros]

        success_approach, approach_plan, plan_time, code = moveit.move_group.plan(pose_approach_ros)
        if not success_approach:
            continue

        traj_point = approach_plan.joint_trajectory.points[len(approach_plan.joint_trajectory.points) - 1]
        joint_state = JointState()
        joint_state.header = approach_plan.joint_trajectory.header
        joint_state.name = approach_plan.joint_trajectory.joint_names
        joint_state.position = traj_point.positions
        joint_state.velocity = traj_point.velocities
        joint_state.effort = traj_point.effort
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = joint_state
        moveit.move_group.set_start_state(moveit_robot_state)
        success_final, final_plan, plan_time, code = moveit.move_group.plan(pose_global_ros)
        if not success_final:
            continue

        moveit.move_group.execute(approach_plan)
        moveit.move_group.stop()

        moveit.move_group.set_start_state(moveit.robot.get_current_state())
        success_final, final_plan, plan_time, code = moveit.move_group.plan(pose_global_ros)
        if not success_final:
            print("doom")
            break

        moveit.move_group.execute(final_plan)
        moveit.move_group.stop()
        # bot.gripper.close()

        break



if __name__ == '__main__':
    main()