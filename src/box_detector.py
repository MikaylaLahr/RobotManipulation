import pyrealsense2
import pyrealsense2.pyrealsense2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs import point_cloud2
import cv2
import numpy as np
import message_filters
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
import math
from geometry_msgs.msg import Pose

# Class to receive color and depth images from rostopic
class ImageListener:
    def __init__(self):
        self.bridge = CvBridge()  # Used to convert from ros format to cv

        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.color_info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.color_info_sub], 10, 0.1)
        self.ts.registerCallback(self.collective_callback)
        
        self.detection_pub = rospy.Publisher('/detections', Detection3DArray, queue_size=1)
        self.debug_pub = rospy.Publisher('/debug', Image, queue_size=1)
        self.cloud_pub = rospy.Publisher('/cloud', PointCloud2, queue_size=1)

    def collective_callback(self, color: Image, depth: Image, camera_info: CameraInfo):
        color_image = self.bridge.imgmsg_to_cv2(color, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')

        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Red Masking
        low_mask = np.array([0, 140, 20])
        hi_mask = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, low_mask, hi_mask)
        low_mask = np.array([170, 140, 20])
        hi_mask = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_img, low_mask, hi_mask)
        red_mask = mask1 + mask2

        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_image = color_image.copy()
        detections = []

        for contour in contours_red:
            if cv2.contourArea(contour) < 500:
                continue
            
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-5:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            # Get the 3D position of the object from the depth image (using the center of the contour)
            depth_value = depth_image[center_y, center_x]
            if depth_value == 0 or math.isnan(depth_value):
                continue

            # Create the intrinsics object for 3D conversion
            intrinsics = pyrealsense2.intrinsics()
            intrinsics.width = camera_info.width
            intrinsics.height = camera_info.height
            intrinsics.ppx = camera_info.K[2]
            intrinsics.ppy = camera_info.K[5]
            intrinsics.fx = camera_info.K[0]
            intrinsics.fy = camera_info.K[4]
            intrinsics.model = pyrealsense2.distortion.none
            intrinsics.coeffs = [i for i in camera_info.D]

            # Deproject 2D pixel to 3D point
            # x, y, z = pyrealsense2.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], depth_value * 0.001)
            depth_m = depth_image * 0.001
            points = np.stack(np.indices((intrinsics.height, intrinsics.width)), axis=-1)
            points = points - np.array([intrinsics.ppy, intrinsics.ppx])
            points = points * np.expand_dims(depth_m, axis=-1)
            points = points / np.array([intrinsics.fy, intrinsics.fx])
            points = np.concatenate([points, np.expand_dims(depth_m, axis=-1)], axis=-1)
            points[:, :, [0, 1]] = points[:, :, [1, 0]]  # swap x and y back

            collapsed_points = np.reshape(points, (-1, 3))
            collapsed_colors = np.reshape(color_image, (-1, 3))
            colors = np.floor(collapsed_colors).astype(np.uint32) # nx3 matrix
            colors = colors[:, 2] << 16 | colors[:, 1] << 8 | colors[:, 0]  # colors are bgr, so reversed order
            cloud_data = np.column_stack([collapsed_points, colors.view(dtype=np.float32)])
            
            FIELDS_XYZRGB = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
            ]
            res = point_cloud2.create_cloud(color.header, FIELDS_XYZRGB, cloud_data)
            self.cloud_pub.publish(res)

            # pose = Pose()
            # pose.position.x = x
            # pose.position.y = y
            # pose.position.z = z

            # detection = Detection3D()
            # detection.header.stamp = color.header.stamp
            # detection.header.frame_id = color.header.frame_id  # Set the correct frame
            # detection.bbox.center = pose

            # detections.append(detection)
            
            cv2.drawContours(debug_image, [contour], contourIdx=0, color=(255, 0, 0))
            bounding_rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(bounding_rect)
            box = np.int0(box)
            cv2.drawContours(debug_image, [box], contourIdx=0, color=(255, 0, 0))

        # Publish the detections as a Detection3DArray
        detection_array = Detection3DArray()
        detection_array.header.stamp = color.header.stamp
        detection_array.header.frame_id = color.header.frame_id  # Ensure this frame matches the TF frame
        detection_array.detections = detections
        self.detection_pub.publish(detection_array)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image))


def main():
    rospy.init_node('box_detector_node')
    _ = ImageListener()
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == "__main__":
    main()
