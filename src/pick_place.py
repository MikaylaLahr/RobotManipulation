#!/usr/bin/env python3

import rospy
import numpy as np
import math
import cv2
import pyrealsense2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import tf2_ros
from tf2_geometry_msgs import do_transform_point

np.float = float

class ImageListener:
    def __init__(self):
        self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.color_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.info_callback)
        self.cv_image = None
        self.dp_image = None
        self.color_info = None

    def color_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Color image conversion error: {e}")

    def depth_callback(self, msg):
        try:
            self.dp_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr(f"Depth image conversion error: {e}")

    def info_callback(self, msg):
        self.color_info = msg

def get_intrinsics(info):
    intrinsics = pyrealsense2.intrinsics()
    intrinsics.width = info.width
    intrinsics.height = info.height
    intrinsics.ppx = info.K[2]
    intrinsics.ppy = info.K[5]
    intrinsics.fx = info.K[0]
    intrinsics.fy = info.K[4]
    intrinsics.model = pyrealsense2.distortion.none
    intrinsics.coeffs = list(info.D)
    return intrinsics

def extract_all_objects(hsv_img, color_bounds):
    object_pixels = []
    for color, bounds in color_bounds.items():
        for low, high in bounds:
            mask = cv2.inRange(hsv_img, low, high)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                M = cv2.moments(contour)
                if abs(M["m00"]) < 1e-5:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                object_pixels.append((cx, cy))
    return object_pixels

def extract_red_bins(hsv_img, red_bounds):
    bins = []
    for low, high in red_bounds:
        mask = cv2.inRange(hsv_img, low, high)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            M = cv2.moments(contour)
            if abs(M["m00"]) < 1e-5:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bins.append((cx, cy))
    return bins

def get_transformed_point(cx, cy, depth_img, intrinsics, buffer):
    depth = depth_img[cy, cx]
    if depth == 0 or math.isnan(depth):
        return None
    x, y, z = pyrealsense2.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth * 0.001)
    point = PointStamped()
    point.header.frame_id = "camera_color_optical_frame"
    point.header.stamp = rospy.Time.now()
    point.point.x = x
    point.point.y = y
    point.point.z = z
    try:
        transform = buffer.lookup_transform('wx250s/base_link', 'camera_color_optical_frame', rospy.Time(0))
        return do_transform_point(point, transform).point
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return None

def move_to_pose(bot, point):
    x, y, z = point.x + 0.025, point.y, point.z
    bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.2, pitch=1.57)
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(x=x, y=y, z=z, pitch=1.57)
    rospy.sleep(0.5)
    bot.gripper.close()
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.2, pitch=1.57)
    rospy.sleep(0.5)

def place_in_bin(bot, bin_point):
    x, y, z = bin_point.x + 0.025, bin_point.y, bin_point.z
    bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.2, roll=0.0, pitch=1.57, yaw=0.0)
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(x=x, y=y, z=z, roll=0.0, pitch=1.57, yaw=0.0)
    rospy.sleep(0.5)
    bot.gripper.open()
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.2, roll=0.0, pitch=1.57, yaw=0.0)
    rospy.sleep(0.5)

def main():
    rospy.init_node("interbotix_pick_place_to_bin")
    bot = InterbotixManipulatorXS("wx250s", moving_time=1.5, accel_time=0.75)
    listener = ImageListener()
    buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(buffer)
    rospy.sleep(2.0)

    color_bounds = {
        'red': [(np.array([0, 100, 50]), np.array([10, 255, 255])), (np.array([160, 100, 50]), np.array([180, 255, 255]))],
        'blue': [(np.array([100, 150, 50]), np.array([140, 255, 255]))],
        'green': [(np.array([40, 100, 40]), np.array([90, 255, 255]))],
        'yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
        'pink': [(np.array([145, 100, 100]), np.array([165, 255, 255]))],
        'gray': [(np.array([0, 0, 40]), np.array([180, 50, 130]))]
    }

    while not rospy.is_shutdown():
        if listener.cv_image is None or listener.dp_image is None or listener.color_info is None:
            rospy.sleep(0.5)
            continue

        hsv_img = cv2.cvtColor(listener.cv_image, cv2.COLOR_BGR2HSV)
        intrinsics = get_intrinsics(listener.color_info)

        object_pixels = extract_all_objects(hsv_img, {k: v for k, v in color_bounds.items() if k != 'red'})
        bin_pixels = extract_red_bins(hsv_img, color_bounds['red'])

        if not object_pixels or not bin_pixels:
            rospy.loginfo("Waiting for objects and bins...")
            rospy.sleep(1.0)
            continue

        bin_pixels = sorted(bin_pixels, key=lambda x: x[1])  # sort bins by y (screen row), assuming lower bin is lower on screen

        for obj_cx, obj_cy in object_pixels:
            obj_point = get_transformed_point(obj_cx, obj_cy, listener.dp_image, intrinsics, buffer)

            for bin_cx, bin_cy in bin_pixels:
                bin_point = get_transformed_point(bin_cx, bin_cy, listener.dp_image, intrinsics, buffer)
                if obj_point and bin_point:
                    move_to_pose(bot, obj_point)
                    place_in_bin(bot, bin_point)
                    break
            bot.arm.go_to_sleep_pose()

if __name__ == '__main__':
    main()
