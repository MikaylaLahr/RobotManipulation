import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from vision_msgs.msg import Detection3DArray, Detection3D
from std_msgs.msg import Header
import struct
import numpy as np
import open3d
import matplotlib

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZRGB = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
]

def open3d_cloud_to_ros(open3d_cloud, header: Header):
    # Set "fields" and "cloud_data"
    points = np.asarray(open3d_cloud.points)
    colors = np.floor(np.asarray(open3d_cloud.colors) * 255.0).astype(np.uint32) # nx3 matrix
    colors = colors[:, 0] << 16 | colors[:, 1] << 8 | colors[:, 2]  
    cloud_data = np.column_stack([points, colors.view(dtype=np.float32)])
    
    # create ros_cloud
    return point_cloud2.create_cloud(header, FIELDS_XYZRGB, cloud_data)

def ros_cloud_to_open3d(ros_cloud: PointCloud2):
    # Get cloud data from ros_cloud
    field_names = [field.name for field in FIELDS_XYZRGB]
    pc2_cloud = point_cloud2.read_points(ros_cloud, skip_nans=True, field_names=field_names)
    cloud_data = np.array(list(pc2_cloud), dtype=np.float32)

    open3d_cloud = open3d.geometry.PointCloud()
    rgb = np.ascontiguousarray(cloud_data[:, 3]).view(dtype=np.uint8).reshape((-1, 4))
    rgb = rgb[:, 0:3][:, ::-1]

    open3d_cloud.points = open3d.utility.Vector3dVector(cloud_data[:, :3])
    open3d_cloud.colors = open3d.utility.Vector3dVector(rgb / 255.0)

    return open3d_cloud

class BoxDetectorNode:
    def __init__(self):
        self.pc_sub = rospy.Subscriber("/point_cloud", PointCloud2, self.point_cloud_callback)
        self.debug_pub = rospy.Publisher("/debug", PointCloud2, queue_size=1)
        self.detection_pub = rospy.Publisher("/detections", Detection3DArray, queue_size=1)

    def point_cloud_callback(self, msg: PointCloud2):
        pcd = ros_cloud_to_open3d(msg)
        colors = np.asarray(pcd.colors)
        hsv_colors = matplotlib.colors.rgb_to_hsv(colors)
        mask = (hsv_colors[:, 0] < 0.05) | (hsv_colors[:, 0] > 0.94)
        mask &= hsv_colors[:, 1] > 0.54
        mask &= hsv_colors[:, 2] > 0.08

        inlier_pcd = pcd.select_by_index(np.where(mask)[0])

        back = open3d_cloud_to_ros(inlier_pcd, msg.header)

        # Publish the filtered point cloud
        self.debug_pub.publish(back)


def main():
    rospy.init_node("box_detector_node")
    _ = BoxDetectorNode()
    rospy.spin()

if __name__ == "__main__":
    main()
