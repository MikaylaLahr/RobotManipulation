import rospy

from interbotix_perception_modules.armtag import InterbotixArmTagInterface
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface
from interbotix_xs_modules.arm import InterbotixManipulatorXS

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

import tf2_ros
from tf2_geometry_msgs import do_transform_point
import geometry_msgs.msg
from sensor_msgs.msg import CameraInfo
import math

import signal

import pyrealsense2

    
transform = None
linktransform = None
opticaltransform = None

# Class to recieve color and depth images from rostopic
class ImageListener:
    def __init__(self):
        self.bridge = CvBridge() #used to convert from ros format to cv
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback) #Raw topics may cause lag, may be worth trying compressed topics
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.color_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.info_callback)
        self.cv_image = None
        self.dp_image = None
        self.color_info = None

    def color_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Process the color image using OpenCV
            #cv2.imshow("Color Image", self.cv_image)
            #cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"Color image conversion error: {e}")

    def depth_callback(self, msg):
        try:
            self.dp_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            #cv2.imshow("Depth Image", cv_image)
            #cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"Depth image conversion error: {e}")

    def info_callback(self, msg):
        self.color_info = msg


def signal_handler(sig, frame): #ensure behavior with ctrl C
    rospy.loginfo("Shutdown initiated with Ctrl+C")
    cv2.destroyAllWindows()
    exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    listener = ImageListener() #Listen for image topic

    #   Initialize the arm module along with the pointcloud and armtag modules
    bot = InterbotixManipulatorXS("wx250s", moving_time=1.5, accel_time=0.75)

    buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(buffer)

    pub = rospy.Publisher('/point', geometry_msgs.msg.PointStamped)
    
    start_area = 0
    while not rospy.is_shutdown():
        if listener.cv_image is None or listener.dp_image is None:
            rospy.loginfo_once("Waiting for frames...")
            # rospy.loginfo_throttle(1, "Waiting for frames...")
            continue

        color_image = listener.cv_image
        depth_image = listener.dp_image
        camera_info = listener.color_info

        # temporary
        listener.cv_image = None

        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Red Masking
        low_mask = np.array([0, 140, 20])
        hi_mask = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, low_mask, hi_mask)
        low_mask = np.array([170, 140, 20])
        hi_mask = np.array([180, 255,255])
        mask2 = cv2.inRange(hsv_img, low_mask, hi_mask)
        red_mask = mask1+mask2

        red_area = np.count_nonzero(red_mask)

        if (start_area != 1):
            start_area = 1
            start_red_area = np.count_nonzero(red_mask)
            #print("##############################################")
            #print("Red Area")
            #print(start_red_area)
            #print("##############################################")

        percent_fill = (red_area/start_red_area)*100

        #print("##############################################")
        print("Percent Fill")
        print(percent_fill)
        #print("##############################################")

        '''
        # Black cube masking
        low_mask = np.array([0, 0, 0])
        hi_mask = np.array([180, 140, 100])
        black_mask = cv2.inRange(hsv_img, low_mask, hi_mask)
        kernel = np.ones((5, 5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)'
        '''

        # Blue cube masking
        low_mask = np.array([100, 150, 50])
        hi_mask = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv_img, low_mask, hi_mask)
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_red = list(sorted(contours_red, key=cv2.contourArea, reverse=True))
        contours_red = contours_red[:1]
        
        for contour in contours_red:
            if cv2.contourArea(contour) < 500:
                continue
            
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-5:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            # Get the 3D position of the object from the depth image (using the center of the contour)
            depth = depth_image[center_y, center_x]
            if depth == 0 or math.isnan(depth):
                continue

            # https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a
            intrinsics = pyrealsense2.intrinsics()
            intrinsics.width = camera_info.width
            intrinsics.height = camera_info.height
            intrinsics.ppx = camera_info.K[2]
            intrinsics.ppy = camera_info.K[5]
            intrinsics.fx = camera_info.K[0]
            intrinsics.fy = camera_info.K[4]
            intrinsics.model = pyrealsense2.distortion.none 
            intrinsics.coeffs = [i for i in camera_info.D]

            x, y, z = pyrealsense2.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], depth * 0.001)
            point = geometry_msgs.msg.PointStamped()
            point.header.frame_id = "camera_color_optical_frame"
            point.header.stamp = rospy.Time.now()
            point.point.x = x
            point.point.y = y
            point.point.z = z

            try:
                transform = buffer.lookup_transform('wx250s/base_link', 'camera_color_optical_frame', rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
        
            transformed_point_red = do_transform_point(point, transform)

            #bot.arm.set_ee_pose_components(x=transformed_point_red.point.x, y=transformed_point_red.point.y, z=transformed_point_red.point.z + 0.2)
            
            pub.publish(point)

            break

        
        '''
        for contour in contours_black:
            if cv2.contourArea(contour) < 500:
                continue
            
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-5:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            # Get the 3D position of the object from the depth image (using the center of the contour)
            depth = depth_image[center_y, center_x]
            if depth == 0 or math.isnan(depth):
                continue

            # https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a
            intrinsics = pyrealsense2.intrinsics()
            intrinsics.width = camera_info.width
            intrinsics.height = camera_info.height
            intrinsics.ppx = camera_info.K[2]
            intrinsics.ppy = camera_info.K[5]
            intrinsics.fx = camera_info.K[0]
            intrinsics.fy = camera_info.K[4]
            intrinsics.model = pyrealsense2.distortion.none 
            intrinsics.coeffs = [i for i in camera_info.D]

            x, y, z = pyrealsense2.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], depth * 0.001)
            point = geometry_msgs.msg.PointStamped()
            point.header.frame_id = "camera_color_optical_frame"
            point.header.stamp = rospy.Time.now()
            point.point.x = x
            point.point.y = y
            point.point.z = z

            try:
                transform = buffer.lookup_transform('wx250s/base_link', 'camera_color_optical_frame', rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
        
            transformed_point = do_transform_point(point, transform)

            bot.arm.set_ee_pose_components(x=transformed_point.point.x, y=transformed_point.point.y, z=transformed_point.point.z + 0.2)
            
            pub.publish(point)'
        '''

        # Initialize a list to store the areas of each blue contour
        blue_areas = []
        blue_contours_with_area = []
        '''
        # Loop through each contour in contours_blue
        for contour in contours_blue:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            blue_areas.append(area)
            blue_contours_with_area.append((contour, area))
        '''

        #blue_areas.sort(key=lambda x: x[1], reverse=True)

        for contour, area in contours_blue: #blue_contours_with_area:               
            if cv2.contourArea(contour) < 500:
                continue
            
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-5:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            # Get the 3D position of the object from the depth image (using the center of the contour)
            depth = depth_image[center_y, center_x]
            if depth == 0 or math.isnan(depth):
                continue

            # https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a
            intrinsics = pyrealsense2.intrinsics()
            intrinsics.width = camera_info.width
            intrinsics.height = camera_info.height
            intrinsics.ppx = camera_info.K[2]
            intrinsics.ppy = camera_info.K[5]
            intrinsics.fx = camera_info.K[0]
            intrinsics.fy = camera_info.K[4]
            intrinsics.model = pyrealsense2.distortion.none 
            intrinsics.coeffs = [i for i in camera_info.D]

            x, y, z = pyrealsense2.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], depth * 0.001)
            point = geometry_msgs.msg.PointStamped()
            point.header.frame_id = "camera_color_optical_frame"
            point.header.stamp = rospy.Time.now()
            point.point.x = x
            point.point.y = y
            point.point.z = z

            try:
                transform = buffer.lookup_transform('wx250s/base_link', 'camera_color_optical_frame', rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
        
            transformed_point = do_transform_point(point, transform)

            bot.arm.set_ee_pose_components(x=transformed_point.point.x, y=transformed_point.point.y, z=transformed_point.point.z + 0.2, pitch=0.5)
            bot.arm.set_ee_pose_components(x=transformed_point.point.x, y=transformed_point.point.y, z=transformed_point.point.z, pitch=0.5)
            bot.gripper.close()
            bot.arm.set_ee_pose_components(x=transformed_point.point.x, y=transformed_point.point.y, z=transformed_point.point.z + 0.2, pitch=0.5)
            bot.arm.set_ee_pose_components(x=transformed_point_red.point.x, y=transformed_point_red.point.y, z=transformed_point_red.point.z + 0.1, pitch=0.5)
            bot.gripper.open()
            
            pub.publish(point)




        #bot.arm.go_to_sleep_pose()


    cv2.destroyAllWindows()




if __name__ == '__main__':
    print("In Code")
    main()