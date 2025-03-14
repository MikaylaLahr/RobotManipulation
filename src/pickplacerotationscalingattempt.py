import rospy

from interbotix_perception_modules.armtag import InterbotixArmTagInterface
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface
from interbotix_xs_modules.arm import InterbotixManipulatorXS

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

import tf2_ros
import geometry_msgs.msg
from tf2_msgs.msg import TFMessage

from scipy.spatial.transform import Rotation as R


import signal

    
transform = None
linktransform = None
opticaltransform = None

#Class to recieve color and depth images from rostopic
class ImageListener:
    def __init__(self):
        self.bridge = CvBridge() #used to convert from ros format to cv
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback) #Raw topics may cause lag, may be worth trying compressed topics
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.cv_image = None
        self.dp_image = None

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


def signal_handler(sig, frame): #ensure behavior with ctrl C
    rospy.loginfo("Shutdown initiated with Ctrl+C")
    cv2.destroyAllWindows()
    exit(0)


def quat_to_rot_matrix(quat):
    """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    # Convert quaternion [x, y, z, w] to a rotation matrix
    r = R.from_quat([quat[0], quat[1], quat[2], quat[3]])  # Pass quaternion as a list
    return r.as_matrix()  # Return the 3x3 rotation matrix

def frame_transform(x, y, z ,transform) :
    point_camera_frame = np.array([x * 0.001, y * 0.001, x])
    translation = np.array([transform.transform.translation.x,
    transform.transform.translation.y,
    transform.transform.translation.z])

    point_base_frame = point_camera_frame + translation  # Adding translation

    # 3. Apply rotation (quaternion to rotation matrix)
    quat = transform.transform.rotation
    q = np.array([quat.x, quat.y, quat.z, quat.w])

    # Convert quaternion to rotation matrix (using scipy)
    rotation_matrix = quat_to_rot_matrix(q)

    # Apply the rotation
    return np.dot(rotation_matrix, point_base_frame)

def main():

    listener = ImageListener() #Listen for image topic

    #   Initialize the arm module along with the pointcloud and armtag modules
    bot = InterbotixManipulatorXS("wx250s", moving_time=1.5, accel_time=0.75)
    pcl = InterbotixPointCloudInterface()

    # Callback function to handle tf_static data
    def tf_static_callback(msg):
        global transform
        global linktransform
        global opticaltransform
        ftransform = None
        for ftransform in msg.transforms: #These are rotation matrices from the tf_static topic, I think some combo should get us where we want?
            if ftransform.header.frame_id == "camera_color_optical_frame" and ftransform.child_frame_id == "wx250s/base_link":
                rospy.loginfo(f"Found static transform: {ftransform}")
                transform = ftransform
            if ftransform.header.frame_id == "camera_link" and ftransform.child_frame_id == "camera_color_frame":
                rospy.loginfo(f"Found static transform: {ftransform}")
                linktransform = ftransform
            if ftransform.header.frame_id == "camera_color_frame" and ftransform.child_frame_id == "camera_color_optical_frame":
                rospy.loginfo(f"Found static transform: {ftransform}")
                opticaltransform = ftransform
        return None

    # Subscribe to tf_static to get the transform once
    rospy.Subscriber("/tf_static", TFMessage, tf_static_callback)

    # Wait for a valid transform from the topic
    rospy.loginfo("Waiting for transform...")


    while transform is None or linktransform is None or opticaltransform is None:
        rospy.sleep(1)  # Sleep while waiting for transform

    rospy.loginfo(f"Got transform: {transform}")


    # set initial arm and gripper pose
    bot.arm.set_ee_pose_components(x=0.3, z=0.2)
    bot.gripper.open()

    # get the cluster positions
    # sort them from max to min 'x' position w.r.t. the 'wx250s/base_link' frame
    success = False
    while not success:
        success, clusters = pcl.get_cluster_positions(ref_frame="wx250s/base_link", sort_axis="x", reverse=True)
        rospy.loginfo((success, clusters))

    print(clusters)

    signal.signal(signal.SIGINT, signal_handler)

    # Start capturing frames from the RealSense camera
    code_running = True
    
    while code_running:

        if listener.cv_image is None or listener.dp_image is None:
            rospy.loginfo("Waiting for new frame...")
            rospy.sleep(0.1)  # Sleep for a while before checking again
            continue

        color_image = listener.cv_image
        depth_image = listener.dp_image
        listener.cv_image = None

        # Color Masking (Red Detection)
        hsv_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Red Masking
        low_mask = np.array([0, 120, 80])
        hi_mask = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, low_mask, hi_mask)

        low_mask = np.array([170, 120, 80])
        hi_mask = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_img, low_mask, hi_mask)
        red_mask = mask1 + mask2

        red_mask_out = color_image.copy()
        red_mask_out[np.where(red_mask == 0)] = 0

        # Brown Masking for the box
        low_mask = np.array([5, 120, 40])
        hi_mask = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv_img, low_mask, hi_mask)

        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #why red mask and not redmask out
        contours_brown, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect and draw contours for red objects
        ###UP TO HERE TESTED
        for contour in contours_red:
            if cv2.contourArea(contour) > 100:
                #This method can also work instead of moment centroid, may be less accurate overall
                '''x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.polylines(color_image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)'''

                M = cv2.moments(contour)
                if M["m00"] != 0:  # Avoid division by zero
                # Compute centroid of the contour
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])

                    # Visualize centroid
                    cv2.circle(color_image, (center_x, center_y), 5, (0, 255, 0), -1)  # Green dot at centroid
                    cv2.imshow("Detected Red Objects", color_image)
                    cv2.waitKey(1)  # Refresh window and handle key events

                    # Get the 3D position of the object from the depth image (using the center of the contour)
                    depth = depth_image[center_y, center_x]
                    if depth != 0:
                        depth_in_meters = depth * 0.001  # Convert to meters
                        # Now, we'll estimate the position of the object in 3D space based on the depth
                        # You may need to calibrate this if necessary

                        #THESE VALUES DO NOT MATCH THOSE OF PICK PLACE ORIGINAL< NEED TO TRANSFORM TO ARM SPACE
                        try:
                            x_position = center_x
                            y_position = center_y
                            z_position = depth_in_meters

                            print("X: ", x_position, "Y: ", y_position, "Z: ", z_position)
                            point = frame_transform(x_position,y_position,z_position, linktransform)
                            point = frame_transform(point[0],point[1],point[2], opticaltransform)
                            point = frame_transform(point[0],point[1],point[2], transform)

                            '''point = geometry_msgs.msg.PointStamped()
                            point.header.stamp = rospy.Time(0)
                            point.header.frame_id = 'camera_color_frame'
                            point.point.x = x_position
                            point.point.y = y_position
                            point.point.z = z_position'''

                            '''point_base = tf_listener.transformPoint('camera_color_frame', point)
                            print("Transformed Point - X: ", point_base.point.x, " Y: ", point_base.point.y, " Z: ", point_base.point.z)'''

                            # The arm moves to the detected object's position
                            '''bot.arm.set_ee_pose_components(x=x_position, y=y_position, z=z_position + 0.05, pitch=0.5)
                            bot.arm.set_ee_pose_components(x=x_position, y=y_position, z=z_position, pitch=0.5)
                            bot.gripper.close()'''
                            print("Point: ", point)
                            bot.arm.set_ee_pose_components(x=-point[0], y=point[1], z=point[2] + 0.05, pitch=0.5)
                            bot.arm.set_ee_pose_components(x=-point[0], y=point[1], z=point[2], pitch=0.5)
                            bot.gripper.close()

                            # Move the arm to the brown box center to place the object
                            '''for brown_contour in contours_brown:
                                if cv2.contourArea(brown_contour) > 100:  # Ensure it's a valid contour
                                    x_brown, y_brown, w_brown, h_brown = cv2.boundingRect(brown_contour)
                                    center_brown_x = x_brown + w_brown // 2
                                    center_brown_y = y_brown + h_brown // 2
                                    # Now use this center as the target for placing the red object
                                    bot.arm.set_ee_pose_components(x=center_brown_x, y=center_brown_y, z=0.2, pitch=0.5)
                                    bot.arm.set_ee_pose_components(x=center_brown_x, y=center_brown_y, z=0.1, pitch=0.5)
                                    bot.gripper.open()
                            '''
                            # Move the arm back to the starting position
                            bot.arm.set_ee_pose_components(x=0.3, z=0.2)
                            bot.gripper.open()
                        except (Exception) as e:
                            print(e)
        code_running = False

        # Show the frame with detected red objects
        red_mask_out = color_image.copy()
        red_mask_out[np.where(red_mask==0)] = 0
        cv2.imshow("Detected Red Objects", red_mask_out)
        cv2.waitKey(1)  # Refresh window and handle key events

    cv2.destroyAllWindows()




if __name__ == '__main__':
    print("In Code")
    main()