from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import PlanningScene, AllowedCollisionEntry
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningSceneRequest
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.msg import PlanningScene, AllowedCollisionMatrix, AllowedCollisionEntry
import rospy
import sys
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from dataclasses import dataclass
from robot_manipulation.srv import ActivePerception, ActivePerceptionRequest, GenerateGraspCandidates, GenerateGraspCandidatesRequest, GenerateGraspCandidatesResponse
import time
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import struct
from pathlib import Path
from typing import List, Dict
import open3d as o3d
import os.path
from vision_msgs.msg import Detection3DArray, BoundingBox3D
from numpy_ros import to_numpy, to_message
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from std_msgs.msg import Header
from interbotix_xs_modules.gripper import InterbotixGripperXS
import copy


@dataclass
class MotionGoal:
    pose: Pose
    take_snapshot: bool


GOALS = [
    MotionGoal(pose=Pose(position=Point(x=0.0, y=0.0, z=0.35), orientation=Quaternion(
        x=0, y=0.428, z=0, w=0.904)), take_snapshot=True),
    MotionGoal(pose=Pose(position=Point(x=0.15, y=0.0, z=0.35), orientation=Quaternion(
        x=0, y=0.6427876, z=0, w=0.7660444)), take_snapshot=True),
    MotionGoal(pose=Pose(position=Point(x=0.047, y=0.227, z=0.144), orientation=Quaternion(
        x=0.07129, y=0.1788946, z=-0.3632623, w=0.9115673)), take_snapshot=True),
    MotionGoal(pose=Pose(position=Point(x=0.047, y=-0.227, z=0.144), orientation=Quaternion(
        x=-0.07129, y=0.1788946, z=0.3632623, w=0.9115673)), take_snapshot=True),
]

SNAPSHOT_SERVICE = "/active_perception/take_snapshot"
GRASP_SERVICE = "/grasp_gen_service/generate_grasp_candidates"
OBJECT_REG_TOPIC = "/object_reg/detections"
BOX_TOPIC = "/box_detector/detections"


class MoveItInterface:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander(
            robot_description="wx250s/robot_description", ns="wx250s")
        self.scene = moveit_commander.PlanningSceneInterface(ns="wx250s")
        self.move_group = moveit_commander.MoveGroupCommander(
            "interbotix_arm", robot_description="wx250s/robot_description", ns="wx250s", wait_for_servers=10.0)
        self.gripper_group = moveit_commander.MoveGroupCommander(
            "interbotix_gripper", robot_description="wx250s/robot_description", ns="wx250s", wait_for_servers=10.0
        )
        self.display_trajectory_publisher = rospy.Publisher("wx250s/move_group/display_planned_path",
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)
        self.interbotix = InterbotixGripperXS(
            "wx250s", "gripper", init_node=False)

    def display_trajectory(self, start_state, plan):
        trajectory = moveit_msgs.msg.DisplayTrajectory()
        trajectory.trajectory_start = start_state
        trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(trajectory)


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
    ros_cloud.width = len(points)  # Number of points
    ros_cloud.fields = fields
    ros_cloud.is_bigendian = False  # Assuming little-endian system
    ros_cloud.point_step = point_step
    ros_cloud.row_step = ros_cloud.point_step * ros_cloud.width
    ros_cloud.is_dense = True  # Assuming no invalid points

    # Pack the point data into a byte array
    buffer = [struct.pack('fff', x, y, z) for x, y, z in points]
    ros_cloud.data = b''.join(buffer)

    return ros_cloud


@dataclass
class ObjectModel:
    id: int
    point_cloud: PointCloud2
    mesh_path: str


def load_objects(path: Path) -> Dict[int, ObjectModel]:
    models = {}
    point_density_m2 = 100_000

    name_to_id = {
        "cube": 0,
        "milk": 1,
        "wine": 2,
        "eggs": 3,
        "toilet_paper": 4,
        "can": 5,
    }

    for filename in os.listdir(path):
        if not filename.lower().endswith(".stl"):
            continue

        file_path = os.path.join(path, filename)
        obj_id = name_to_id.get(filename[:-4])
        if obj_id is None:
            continue

        mesh = o3d.io.read_triangle_mesh(file_path)
        num_points_to_sample = int(
            point_density_m2 * mesh.get_surface_area())
        point_cloud_o3d = mesh.sample_points_uniformly(
            number_of_points=num_points_to_sample)
        ros_point_cloud = convert_open3d_to_ros_pointcloud2(
            point_cloud_o3d)

        models[obj_id] = ObjectModel(
            id=obj_id, point_cloud=ros_point_cloud, mesh_path=file_path)

    return models


def add_pole_obstacles(moveit: MoveItInterface):
    box_name = "pole1"
    moveit.scene.remove_world_object(box_name)
    box_pose = PoseStamped()
    box_pose.header.frame_id = "wx250s/base_link"
    box_pose.pose.position.x = 0.35
    box_pose.pose.position.y = -0.18
    box_pose.pose.position.z = 0.5
    box_pose.pose.orientation.w = 1.0  # No rotation
    box_dimensions = (0.05, 0.05, 1.0)
    moveit.scene.add_box(box_name, box_pose, size=box_dimensions)

    box_name = "pole2"
    moveit.scene.remove_world_object(box_name)
    box_pose = PoseStamped()
    box_pose.header.frame_id = "wx250s/base_link"
    box_pose.pose.position.x = 0.35
    box_pose.pose.position.y = 0.20
    box_pose.pose.position.z = 0.5
    box_pose.pose.orientation.w = 1.0  # No rotation
    box_dimensions = (0.05, 0.05, 1.0)
    moveit.scene.add_box(box_name, box_pose, size=box_dimensions)


def add_attached_camera(moveit: MoveItInterface):
    attach_link = "wx250s/ar_tag_link"
    object_name = "attached_camera"
    moveit.scene.remove_attached_object(attach_link, object_name)

    relative_pose = Pose(position=Point(x=0.01, y=0.0, z=0.01),
                         orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
    stamped = PoseStamped(header=Header(
        frame_id=attach_link), pose=relative_pose)

    touch_links = [attach_link,
                   "wx250s/gripper_bar_link", "wx250s/gripper_link"]

    moveit.scene.attach_box(attach_link, object_name, pose=stamped,
                            size=(0.03, 0.11, 0.04), touch_links=touch_links)


def add_ground_obstacle(moveit: MoveItInterface):
    box_name = "ground"
    moveit.scene.remove_world_object(box_name)
    box_pose = PoseStamped()
    box_pose.header.frame_id = "wx250s/base_link"
    box_pose.pose.position.x = 0.0
    box_pose.pose.position.y = 0.0
    box_pose.pose.position.z = -0.025
    box_pose.pose.orientation.w = 1.0  # No rotation
    box_dimensions = (1.0, 0.5, 0.05)
    moveit.scene.add_box(box_name, box_pose, size=box_dimensions)


def add_detected_objects(moveit: MoveItInterface, objects: Detection3DArray, models: Dict[int, ObjectModel]):
    for detection in objects.detections:
        assert len(detection.results) == 1
        result = detection.results[0]

        pose_stamped = PoseStamped()
        pose_stamped.header = objects.header
        pose_stamped.pose = result.pose.pose

        model_id = result.id // 10_000
        object_id = result.id % 10_000

        model_data = models[model_id]

        moveit.scene.add_mesh(f"object_{object_id}", pose_stamped,
                              filename=model_data.mesh_path, size=(1, 1, 1))


def scan_area(moveit: MoveItInterface, take_snapshot: rospy.ServiceProxy):
    box_name = "objects_area"
    moveit.scene.remove_world_object(box_name)

    box_pose = PoseStamped()
    box_pose.header.frame_id = "wx250s/base_link"

    # Position the center of the box
    box_pose.pose.position.x = 0.3
    box_pose.pose.position.y = 0.0
    box_pose.pose.position.z = 0.1
    box_pose.pose.orientation.w = 1.0  # No rotation

    box_dimensions = (0.3, 0.3, 0.2)

    moveit.scene.add_box(box_name, box_pose, size=box_dimensions)

    for i, goal in enumerate(GOALS):
        rospy.loginfo(f"Planning goal {i}.")
        moveit.move_group.set_pose_target(goal.pose)
        success, plan, plan_time, code = moveit.move_group.plan()
        if not success:
            rospy.logerr(f"Failed to plan to pose {i}.")
            continue

        current_state = moveit.robot.get_current_state()
        moveit.display_trajectory(current_state, plan)
        answer = input("Trajectory okay [y/N]? ")
        if answer.lower() == "y":
            moveit.move_group.execute(plan)
            moveit.move_group.stop()
        else:
            continue

        if goal.take_snapshot:
            try:
                take_snapshot(ActivePerceptionRequest())
                rospy.loginfo("Snapshot taken.")
            except rospy.ServiceException as e:
                rospy.logwarn(f"Failed to take snapshot! {e}")

    moveit.scene.remove_world_object(box_name)


def open_gripper(moveit: MoveItInterface) -> bool:
    moveit.gripper_group.set_named_target("Open")
    return moveit.gripper_group.go(wait=True)


def close_gripper(moveit: MoveItInterface, width: float) -> bool:
    width = max(min(width - 0.01, 0.035), 0.014)
    moveit.gripper_group.set_joint_value_target({"left_finger": width})
    return moveit.gripper_group.go(wait=True)


def get_acm():
    # Set up service to get the current planning scene
    service_timeout = 5.0
    _get_planning_scene = rospy.ServiceProxy(
        "wx250s/get_planning_scene", GetPlanningScene)
    _get_planning_scene.wait_for_service(service_timeout)
    request = GetPlanningScene()
    request.components = 0  # Get just the Allowed Collision Matrix
    scene = _get_planning_scene.call(request).scene
    return scene.allowed_collision_matrix


def set_acm_for_object(acm, obj, other=None, allowed=False):
    """Updates the MoveIt PlanningScene using the AllowedCollisionMatrix to ignore collisions for an object"""
    if other is None:
        set_acm_default_for_object(acm, obj, allowed)
        return

    other_idx = acm.entry_names.index(other)
    if obj not in acm.entry_names:
        acm.entry_names.append(obj)
        for entry in acm.entry_values:
            entry.enabled.append(allowed)
        acm.entry_values.append(AllowedCollisionEntry(
            enabled=[allowed for i in range(len(acm.entry_names))]))
        acm.entry_values[-1].enabled[other_idx] = allowed
    else:
        obj_idx = acm.entry_names.index(obj)
        acm.entry_values[obj_idx].enabled[other_idx] = allowed
        acm.entry_values[other_idx].enabled[obj_idx] = allowed


def set_acm_default_for_object(acm, obj, allowed=False):
    if obj not in acm.default_entry_names:
        acm.default_entry_names.append(obj)
        acm.default_entry_values.append(allowed)
    else:
        idx = acm.default_entry_names.index(obj)
        acm.default_entry_values[idx] = allowed


def apply_acm(psi, acm):
    req = ApplyPlanningSceneRequest()
    req.scene.allowed_collision_matrix = acm
    req.scene.is_diff = True
    req.scene.robot_state.is_diff = True
    psi._apply_planning_scene_diff.call(req)


def attach_object_to_ee(moveit: MoveItInterface, object_id):
    gripper_links = moveit.robot.get_link_names(group="interbotix_gripper")
    attach_link = moveit.move_group.get_end_effector_link()
    rospy.loginfo(f"Will attach object to link: '{attach_link}'")

    touch_links = list(gripper_links)

    rospy.loginfo(
        f"Attaching '{object_id}' to '{attach_link}' with touch_links: {touch_links}")

    moveit.move_group.attach_object(
        object_id, attach_link, touch_links=touch_links)


def detach_object_from_ee(moveit: MoveItInterface, object_id):
    attach_link = moveit.move_group.get_end_effector_link()
    rospy.loginfo(f"Will detach object from link: '{attach_link}'")
    moveit.move_group.detach_object(object_id)


def bbox_volume(bbox: BoundingBox3D) -> float:
    return bbox.size.x * bbox.size.y * bbox.size.z


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return v1.dot(v2) / (np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2)))


def main():
    rospy.init_node("active_perception_runner")
    moveit = MoveItInterface()

    moveit.move_group.allow_replanning(True)
    moveit.move_group.set_goal_tolerance(0.01)
    moveit.move_group.set_num_planning_attempts(10)
    moveit.move_group.set_planning_time(3.0)
    moveit.move_group.set_max_velocity_scaling_factor(0.5)

    moveit.gripper_group.set_goal_tolerance(0.001)
    moveit.gripper_group.set_max_velocity_scaling_factor(1.0)

    add_pole_obstacles(moveit)
    add_attached_camera(moveit)
    add_ground_obstacle(moveit)

    try:
        rospy.wait_for_service(SNAPSHOT_SERVICE, timeout=5.0)
    except rospy.ROSException as e:
        rospy.logerr(f"Service '{SNAPSHOT_SERVICE}' not available: {e}")
        return
    take_snapshot = rospy.ServiceProxy(
        SNAPSHOT_SERVICE, ActivePerception, persistent=True)

    try:
        rospy.wait_for_service(GRASP_SERVICE, timeout=5.0)
    except rospy.ROSException as e:
        rospy.logerr(f"Service '{GRASP_SERVICE}' not available: {e}")
        return
    generate_grasps = rospy.ServiceProxy(
        GRASP_SERVICE, GenerateGraspCandidates, persistent=True)

    models = load_objects(
        "/home/group3/interbotix_ws/src/robot_manipulation/resources/meshes")

    open_gripper(moveit)

    scan_area(moveit, take_snapshot)

    rospy.loginfo("Waiting for detections")
    # technically there is a race condition here lol
    detections: Detection3DArray = rospy.wait_for_message(
        OBJECT_REG_TOPIC, Detection3DArray, rospy.Duration(30.0))

    rospy.loginfo("Adding object models")
    add_detected_objects(moveit, detections, models)
    rospy.loginfo("Object models added")

    current_state = moveit.robot.get_current_state()

    eef_link = moveit.move_group.get_end_effector_link()
    current_pose_stamped = moveit.move_group.get_current_pose(eef_link)
    current_pose = current_pose_stamped.pose
    current_pose_np = to_numpy(current_pose, homogeneous=True)
    facing_direction = current_pose_np[:3, 0]

    for detection in detections.detections:
        assert len(detection.results) == 1
        result = detection.results[0]

        model_id = result.id // 10_000
        object_id = result.id % 10_000

        if model_id != 5:
            continue

        object_name = f"object_{object_id}"

        object_pose = to_numpy(result.pose.pose, homogeneous=True)

        model = models[model_id]
        request = GenerateGraspCandidatesRequest(model.point_cloud)
        response: GenerateGraspCandidatesResponse = generate_grasps(request)

        for candidate in sorted(response.candidates.candidates, key=lambda x: cosine_similarity(to_numpy(x.approach), facing_direction), reverse=True):
            print(candidate.width)

            approach: np.ndarray = to_numpy(candidate.approach)
            binormal: np.ndarray = to_numpy(candidate.binormal)
            axis: np.ndarray = to_numpy(candidate.axis)

            if axis.dot(np.array([0, 0, 1])) < 0:
                binormal *= -1
                axis *= -1

            gripper_rot = np.column_stack([approach, binormal, axis])
            gripper_t = to_numpy(candidate.position)
            grasp_pose_local = np.block(
                [[gripper_rot, gripper_t.reshape((-1, 1))], [np.zeros(3), 1]])

            pose_approach = object_pose @ np.block(
                [[np.eye(3), (-0.05 * approach).reshape((-1, 1))], [np.zeros(3), 1]]) @ grasp_pose_local

            pose_global = object_pose @ np.block(
                [[np.eye(3), (0.03 * approach).reshape((-1, 1))], [np.zeros(3), 1]]) @ grasp_pose_local

            pose_approach_ros = to_message(Pose, pose_approach)
            pose_global_ros = to_message(Pose, pose_global)

            moveit.move_group.set_start_state(current_state)
            moveit.move_group.set_pose_target(pose_approach_ros)
            success_approach, approach_plan, plan_time, code = moveit.move_group.plan()
            if not success_approach:
                continue

            traj_point = approach_plan.joint_trajectory.points[len(
                approach_plan.joint_trajectory.points) - 1]
            joint_state = JointState()
            joint_state.header = approach_plan.joint_trajectory.header
            joint_state.name = approach_plan.joint_trajectory.joint_names
            joint_state.position = traj_point.positions
            joint_state.velocity = traj_point.velocities
            joint_state.effort = traj_point.effort
            moveit_robot_state = RobotState()
            moveit_robot_state.joint_state = joint_state

            moveit.move_group.set_start_state(moveit_robot_state)
            moveit.move_group.set_pose_target(pose_global_ros)
            success_final, final_plan, plan_time, code = moveit.move_group.plan()
            if not success_final:
                continue

            moveit.display_trajectory(current_state, approach_plan)
            answer = input("Trajectory okay [y/N]? ")
            if answer.lower() != "y":
                rospy.loginfo("Trajectory not okay. Trying others...")
                continue

            moveit.move_group.execute(approach_plan)
            moveit.move_group.stop()

            new_current = moveit.robot.get_current_state()
            moveit.move_group.set_start_state(new_current)
            moveit.move_group.set_pose_target(pose_global_ros)
            success_final, final_plan, plan_time, code = moveit.move_group.plan()
            if not success_final:
                rospy.logwarn(
                    "Failed to successfully plan to final state. wtf?")
                break

            moveit.display_trajectory(new_current, final_plan)
            answer = input("Trajectory okay [y/N]? ")
            if answer.lower() != "y":
                rospy.loginfo("Trajectory not okay. Trying others...")
                continue

            moveit.move_group.execute(final_plan)
            moveit.move_group.stop()

            acm = get_acm()
            set_acm_for_object(
                acm, object_name, other="wx250s/left_finger_link", allowed=True)
            set_acm_for_object(
                acm, object_name, other="wx250s/right_finger_link", allowed=True)
            apply_acm(moveit.scene, acm)

            close_gripper(moveit, candidate.width)
            attach_object_to_ee(moveit, object_name)

            boxes: Detection3DArray = rospy.wait_for_message(
                BOX_TOPIC, Detection3DArray, rospy.Duration(1.0))

            sorted_boxes = sorted(
                boxes.detections, key=lambda x: bbox_volume(x.bbox), reverse=True)

            if len(sorted_boxes) == 0:
                rospy.logwarn("No boxes found!")
                break

            box: BoundingBox3D = sorted_boxes[0].bbox
            eef_link = moveit.move_group.get_end_effector_link()
            target_position_xyz = [
                box.center.position.x, box.center.position.y, box.center.position.z + 0.1]

            moveit.move_group.set_start_state(moveit.robot.get_current_state())
            moveit.move_group.set_position_target(
                target_position_xyz, end_effector_link=eef_link)
            success_drop, drop_plan, plan_time, code = moveit.move_group.plan()
            if not success_drop:
                rospy.logwarn("Failed drop plan")
                break

            moveit.move_group.execute(drop_plan)

            detach_object_from_ee(moveit, object_name)
            open_gripper(moveit)
            moveit.scene.remove_world_object(object_name)

            return


if __name__ == '__main__':
    main()
