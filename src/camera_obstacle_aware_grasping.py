#!/usr/bin/env python3
#THis is the most basic arcitecture I can give you for the path planning
#Injects detected objects into plan
#Uses MoveIt to plan a path and prevent collision
#I hope this helps, please forgive me, let me know how well it works for you
#Error with interbotix have no idea how to fix
import rospy
import numpy as np
#fix call for WidowX Controller
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import PointCloud2
# import sensor_msgs.point_cloud2 as pc2  # optional for real point cloud processing

def detect_objects_from_pointcloud(pointcloud_msg):
    #Simulate object detection (replace this with your point cloud clustering output).
    #Return list of object dicts with 'label' and 'position' keys.
    return [
        {"label": "can", "position": {"x": 0.2, "y": -0.1, "z": 0.05}},
        {"label": "milk", "position": {"x": 0.15, "y": 0.0, "z": 0.05}},  # TARGET
        {"label": "eggs", "position": {"x": 0.3, "y": 0.05, "z": 0.05}}
    ]

def add_obstacle(scene, name, position, size=(0.05, 0.05, 0.1)):
    #Add a box-shaped obstacle to MoveIt planning scene.
    pose = PoseStamped()
    pose.header.frame_id = "wx250s/base_link"
    pose.pose.position.x = position['x']
    pose.pose.position.y = position['y']
    pose.pose.position.z = position['z']
    pose.pose.orientation.w = 1.0
    scene.add_box(name, pose, size=size)

def setup_scene_with_obstacles(scene, objects, target_label):
    #Clear and re-add all non-target objects to MoveIt scene as obstacles.
    rospy.sleep(1.0)
    scene.remove_world_object()
    for i, obj in enumerate(objects):
        if obj["label"] != target_label:
            add_obstacle(scene, f"obstacle_{i}", obj["position"])

def plan_and_execute(bot, move_group, target_position):
    #Plan and execute a collision-free motion to the target pose.
    pose_target = Pose()
    pose_target.position.x = target_position['x']
    pose_target.position.y = target_position['y']
    pose_target.position.z = target_position['z']
    pose_target.orientation.w = 1.0
    move_group.set_pose_target(pose_target)

    success, plan, _, _ = move_group.plan()
    if success:
        rospy.loginfo("Plan successful. Executing...")
        move_group.execute(plan, wait=True)
        bot.gripper.close()
    else:
        rospy.logwarn("Path planning failed.")

def main():
    rospy.init_node("camera_obstacle_aware_grasping")

    bot = InterbotixManipulatorXS("wx250s", moving_time=1.5, accel_time=0.75)
    move_group = MoveGroupCommander("interbotix_arm")
    scene = PlanningSceneInterface(ns="wx250s")

    rospy.sleep(2.0)
    bot.arm.go_to_sleep_pose()
    bot.gripper.open()

    #Detect Objects
    dummy_pc_msg = PointCloud2()  # Placeholder, replace with real data
    detected_objects = detect_objects_from_pointcloud(dummy_pc_msg)

    #Identify Target
    target_label = "milk"
    target_obj = next((obj for obj in detected_objects if obj["label"] == target_label), None)

    if not target_obj:
        rospy.logwarn("Target object not found.")
        return

    #Add Obstacles
    setup_scene_with_obstacles(scene, detected_objects, target_label)

    #Plan to Target
    plan_and_execute(bot, move_group, target_obj["position"])

    #Return to Sleep
    bot.arm.go_to_sleep_pose()

if __name__ == '__main__':
    main()
#Make it executable:
#chmod +x camera_obstacle_aware_grasping.py
#Run with ROS:
#rosrun your_package_name camera_obstacle_aware_grasping.py
