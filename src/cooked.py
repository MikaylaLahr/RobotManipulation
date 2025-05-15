import rospy
import numpy as np
import math
import cv2
import pyrealsense2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Pose
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesisWithPose
from robot_manipulation.srv import GeneratePackingPlanRequest, GeneratePackingPlan, GeneratePackingPlanResponse
from robot_manipulation.msg import PackingPlan, PackItem
from typing import List
from numpy_ros import to_numpy, to_message

BOX_TOPIC = "/box_detector/detections"
OBJECT_TOPIC = "/object_reg/detections"


np.float = float


class Listener:
    def __init__(self):
        self.box_sub = rospy.Subscriber(
            BOX_TOPIC, Detection3DArray, self.callback)
        self.object_sub = rospy.Subscriber(
            OBJECT_TOPIC, Detection3DArray, self.callback_2
        )
        self.object_sub = rospy.Subscriber
        self.box_data = None
        self.object_data = None

    def callback(self, boxes: Detection3DArray):
        self.box_data = boxes

    def callback_2(self, objects: Detection3DArray):
        self.object_data = objects


def detection_volume(detection: Detection3D) -> float:
    return detection.bbox.size.x * detection.bbox.size.y * detection.bbox.size.z


def within_bbox(bbox: BoundingBox3D, point: Point) -> bool:
    half_x = bbox.size.x / 2
    half_y = bbox.size.y / 2
    half_z = bbox.size.z / 2
    return (abs(point.x - bbox.center.position.x) < half_x and abs(point.y - bbox.center.position.y) < half_y and abs(point.z - bbox.center.position.z) < half_z)


def move_to_pose(bot, point):
    bot.gripper.open()
    x, y, z = point.x + 0.02, point.y, point.z
    bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.2, pitch=1.00)
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(x=x, y=y, z=z, pitch=1.00)
    rospy.sleep(0.5)
    bot.gripper.close()
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.2, pitch=1.00)
    rospy.sleep(0.5)


def pick_up_object(bot, point: Point):
    point_np = np.array([point.x, point.y, point.z])
    approach_vec = point_np.copy()
    z_angle = -1.00  # radians
    approach_vec[2] = np.linalg.norm(point_np) * np.tan(z_angle)
    approach_vec /= np.linalg.norm(approach_vec)

    approach_point = point_np - 0.07 * approach_vec

    approach_vec_only_plane = approach_vec.copy()
    approach_vec_only_plane[2] = 0
    yaw_angle = np.arctan2(
        approach_vec_only_plane[1], approach_vec_only_plane[0])

    bot.gripper.open()
    bot.arm.set_ee_pose_components(
        x=approach_point[0], y=approach_point[1], z=approach_point[2], roll=0.0, pitch=-z_angle, yaw=yaw_angle)
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(
        x=point_np[0], y=point_np[1], z=point_np[2], roll=0.0, pitch=-z_angle, yaw=yaw_angle)
    rospy.sleep(0.5)
    bot.gripper.close()
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(
        x=point_np[0], y=point_np[1], z=point_np[2] + 0.2, roll=0.0, pitch=-z_angle, yaw=yaw_angle)
    rospy.sleep(0.5)


def place_in_bin(bot, bin_point):
    x, y, z = bin_point.x, bin_point.y, bin_point.z
    bot.arm.set_ee_pose_components(
        x=x, y=y, z=z + 0.2, roll=0.0, pitch=1.00, yaw=0.0)
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(
        x=x, y=y, z=z + 0.05, roll=0.0, pitch=1.00, yaw=0.0)
    rospy.sleep(0.5)
    bot.gripper.open()
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(
        x=x, y=y, z=z + 0.2, roll=0.0, pitch=1.00, yaw=0.0)
    rospy.sleep(0.5)


def main():
    rospy.init_node("interbotix_pick_place_to_bin")
    bot = InterbotixManipulatorXS(
        "wx250s", moving_time=1.5, accel_time=0.75, init_node=False)
    listener = Listener()

    get_plan = rospy.ServiceProxy(
        'generate_packing_plan', GeneratePackingPlan)

    while not rospy.is_shutdown():
        if listener.box_data is None or listener.object_data is None:
            rospy.sleep(0.5)
            continue

        data = listener.box_data
        objects = listener.object_data
        break

    rospy.loginfo(f"{len(data.detections)} boxes detected")
    sorted_detections: List[Detection3D] = sorted(
        data.detections, key=detection_volume)
    for detection in sorted_detections:
        assert len(detection.results) == 1
        result = detection.results[0]
        rospy.loginfo(f"The next largest box is box {result.id}")

    filtered_objects: List[Detection3D] = [
        det for det in objects.detections if 0.125 < det.bbox.center.position.x < 0.35 and all(not within_bbox(det_box.bbox, det.bbox.center.position) for det_box in sorted_detections)]
    rospy.loginfo(f"{len(filtered_objects)} objects found")

    if len(filtered_objects) == 0 or len(sorted_detections) != 2:
        rospy.logwarn("No objects or not the right number of boxes!")
        return

    biggest_box = sorted_detections[1]
    smallest_box = sorted_detections[0]

    cube_type_mesh_id = 1
    object_detections = Detection3DArray()

    # remove invalid detections
    filtered_objects = [
        obj for obj in filtered_objects
        if obj.results and (obj.results[0].id//10000) != 6
    ]

    for i, object in enumerate(filtered_objects):

        hyp = ObjectHypothesisWithPose()
        hyp.id = (object.results[0].id//10000) * 10000 + i
        object.results = [hyp]
    object_detections.detections = filtered_objects

    plan: GeneratePackingPlanResponse = get_plan(
        GeneratePackingPlanRequest(detections=object_detections))

    print(object_detections)
    rospy.loginfo("Plan generated")

    smallest_box.bbox.center.position.y += 0.015
    smallest_box.bbox.center.position.x += 0.01

    biggest_box.bbox.center.position.x += 0.02
    biggest_box.bbox.center.position.y += 0.01

    for item in plan.plan.items:
        item: PackItem
        idx = item.item_id % 10000
        object = filtered_objects[idx]

        box = biggest_box if item.box_id == 1 else smallest_box
        # print(f"\n --- \n box {box}")
        end_pose_np = to_numpy(item.end_pose, homogeneous=True)
        box_pose_np = to_numpy(box.bbox.center, homogeneous=True)
        print(end_pose_np, box_pose_np)
        end_pose_global = end_pose_np @ box_pose_np
        print(end_pose_global)
        message: Pose = to_message(Pose, end_pose_global)

        object.bbox.center.position.x += 0.01
        object.bbox.center.position.x += 0.005
        pick_up_object(bot, object.bbox.center.position)
        place_in_bin(bot, message.position)
        bot.arm.go_to_sleep_pose()


if __name__ == "__main__":
    main()
