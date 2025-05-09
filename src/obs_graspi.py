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

def extract_bins(listener, buffer):
    if listener.cv_image is None or listener.dp_image is None or listener.color_info is None:
        return None

    hsv_img = cv2.cvtColor(listener.cv_image, cv2.COLOR_BGR2HSV)
    red_range = [
        (np.array([0, 100, 50]), np.array([10, 255, 255])),
        (np.array([160, 100, 50]), np.array([180, 255, 255]))
    ]

    intrinsics = pyrealsense2.intrinsics()
    intrinsics.width = listener.color_info.width
    intrinsics.height = listener.color_info.height
    intrinsics.ppx = listener.color_info.K[2]
    intrinsics.ppy = listener.color_info.K[5]
    intrinsics.fx = listener.color_info.K[0]
    intrinsics.fy = listener.color_info.K[4]
    intrinsics.model = pyrealsense2.distortion.none
    intrinsics.coeffs = list(listener.color_info.D)

    for low, high in red_range:
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
            depth = listener.dp_image[cy, cx]
            if depth == 0 or math.isnan(depth):
                continue

            x, y, z = pyrealsense2.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth * 0.001)
            point = PointStamped()
            point.header.frame_id = "camera_color_optical_frame"
            point.header.stamp = rospy.Time.now()
            point.point.x = x
            point.point.y = y
            point.point.z = z

            try:
                transform = buffer.lookup_transform('wx250s/base_link', 'camera_color_optical_frame', rospy.Time(0))
                transformed_point = do_transform_point(point, transform)
                return transformed_point.point
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
    return None

def move_to_pose(bot, point):
    x, y, z = point.x + 0.025, point.y, point.z
    bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.2, pitch=1.57)
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(x=x, y=y, z=z, pitch=1.57)
    rospy.sleep(0.5)
    bot.gripper.open()
    rospy.sleep(0.5)
    bot.arm.set_ee_pose_components(x=x, y=y, z=z + 0.2, pitch=1.57)
    rospy.sleep(0.5)

def main():
    rospy.init_node("interbotix_place_bin")
    bot = InterbotixManipulatorXS("wx250s", moving_time=3.0, accel_time=0.75, init_node=False)
    listener = ImageListener()
    buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(buffer)
    rospy.sleep(2.0)

    while not rospy.is_shutdown():
        point = extract_bins(listener, buffer)
        if not point:
            rospy.loginfo("Waiting for red bins to be detected...")
            rospy.sleep(1.0)
            continue

        rospy.loginfo("Red bin detected. Moving to place...")
        move_to_pose(bot, point)
        bot.arm.go_to_sleep_pose()
        break

if __name__ == '__main__':
    main()
