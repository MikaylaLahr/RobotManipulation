#include <open3d/geometry/BoundingVolume.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/FileFormatIO.h>
#include <open3d/io/ModelIO.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/pipelines/registration/ColoredICP.h>
#include <open3d/pipelines/registration/FastGlobalRegistration.h>
#include <open3d/pipelines/registration/Feature.h>
#include <open3d/pipelines/registration/Registration.h>
#include <open3d/pipelines/registration/RobustKernel.h>
#include <open3d/pipelines/registration/TransformationEstimation.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Pose.h>
#include <vision_msgs/Detection3DArray.h>
#include <vision_msgs/Detection3D.h>
#include <vision_msgs/BoundingBox3D.h>
#include <rviz_visual_tools/rviz_visual_tools.h>
#include <tf2_eigen/tf2_eigen.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <apriltag_ros/AprilTagDetectionArray.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <open3d/Open3D.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <string>
#include <optional>

#include "open3d_conversions/open3d_conversions.h"
#include "robot_manipulation/ActivePerception.h"

class ActivePerceptionService {
public:
    ActivePerceptionService(): nh_("~"), tf_listener_(tf_buffer_) {
        nh_.param<double>("crop_min_x", crop_box_.min_bound_.x(), 0.0);
        nh_.param<double>("crop_min_y", crop_box_.min_bound_.y(), -0.2);
        nh_.param<double>("crop_min_z", crop_box_.min_bound_.z(), 0.0);
        nh_.param<double>("crop_max_x", crop_box_.max_bound_.x(), 0.6);
        nh_.param<double>("crop_max_y", crop_box_.max_bound_.y(), 0.2);
        nh_.param<double>("crop_max_z", crop_box_.max_bound_.z(), 0.15);
        nh_.param<double>("voxel_size", voxel_size_, 0.005);
        nh_.param<std::string>("base_link_frame", base_link_frame_, "wx250s/base_link");

        service_ = nh_.advertiseService(
            "take_snapshot", &ActivePerceptionService::take_snapshot_callback, this);
        debug_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("debug_cloud", 1, true);

        ROS_INFO("Active perception node initialized.");
    }

private:
    bool take_snapshot_callback(robot_manipulation::ActivePerception::Request& req,
        robot_manipulation::ActivePerception::Response& res) {
        sensor_msgs::PointCloud2::ConstPtr pc =
            ros::topic::waitForMessage<sensor_msgs::PointCloud2>(
                "/point_cloud", ros::Duration(1.0));

        if (!pc) {
            ROS_WARN("Timed out waiting for point cloud message.");
            return false;
        }

        apriltag_ros::AprilTagDetectionArrayConstPtr tags =
            ros::topic::waitForMessage<apriltag_ros::AprilTagDetectionArray>(
                "/tag_detections", ros::Duration(1.0));

        if (!tags) {
            ROS_WARN("Timed out waiting for AprilTag detections message.");
            return false;
        }

        if (tags->detections.empty()) {
            ROS_WARN("No tags in frame!");
            return false;
        }

        bool found = false;
        apriltag_ros::AprilTagDetection detection;
        for (const auto& tag: tags->detections) {
            if (tag.id[0] == 29) {
                found = true;
                detection = tag;
            }
        }

        if (!found) {
            ROS_WARN("Correct tag id not found!");
            return false;
        }

        if (pc_so_far.IsEmpty()) {
            // first time around, just assume the camera pose is roughly correct
            // when we get more clouds we will not use the robot's odometry, only the tag to align
            // scans
            geometry_msgs::PoseStamped stamped;
            stamped.header = detection.pose.header;
            stamped.pose = detection.pose.pose.pose;

            tag_pose_in_base_link = tf_buffer_.transform(stamped, base_link_frame_).pose;

            auto transformed = transform_point_cloud(*pc, base_link_frame_);
            if (!transformed) {
                return false;
            }

            sensor_msgs::PointCloud2::ConstPtr pc_transformed =
                boost::make_shared<sensor_msgs::PointCloud2>(std::move(*transformed));

            open3d::geometry::PointCloud o3d_cloud;
            open3d_conversions::rosToOpen3d(pc_transformed, o3d_cloud);

            o3d_cloud = *o3d_cloud.Crop(crop_box_);
            pc_so_far += o3d_cloud;
            pc_so_far = *pc_so_far.VoxelDownSample(voxel_size_);
        } else {
            // now need to reconstruct camera -> base_link transform from detection.pose and
            // tag_pose_in_base_link
            Eigen::Isometry3d eigen_detection_pose;
            tf2::fromMsg(detection.pose.pose.pose, eigen_detection_pose);

            Eigen::Isometry3d eigen_tag_pose_base_link;
            tf2::fromMsg(tag_pose_in_base_link, eigen_tag_pose_base_link);

            auto pc_to_optical = tf_buffer_.lookupTransform(
                detection.pose.header.frame_id, pc->header.frame_id, pc->header.stamp);
            Eigen::Isometry3d eigen_pc_to_optical = tf2::transformToEigen(pc_to_optical);

            Eigen::Isometry3d pc_to_base_link = eigen_tag_pose_base_link
                                                * eigen_detection_pose.inverse()
                                                * eigen_pc_to_optical;

            open3d::geometry::PointCloud o3d_cloud;
            open3d_conversions::rosToOpen3d(pc, o3d_cloud);

            o3d_cloud.Transform(pc_to_base_link.matrix());

            o3d_cloud = *o3d_cloud.Crop(crop_box_);
            pc_so_far += o3d_cloud;
            pc_so_far = *pc_so_far.VoxelDownSample(voxel_size_);
        }

        open3d_conversions::open3dToRos(pc_so_far, res.collected_cloud, base_link_frame_);
        debug_pub_.publish(res.collected_cloud);
        return true;
    }

    std::optional<sensor_msgs::PointCloud2> transform_point_cloud(
        const sensor_msgs::PointCloud2& msg, const std::string& target_frame) {
        sensor_msgs::PointCloud2 cloud_transformed;
        try {
            // Wait for the transform to become available
            geometry_msgs::TransformStamped transformStamped = tf_buffer_.lookupTransform(
                target_frame, msg.header.frame_id, msg.header.stamp, ros::Duration(1.0));

            // Transform the point cloud
            tf2::doTransform(msg, cloud_transformed, transformStamped);
        } catch (tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(3, "Unable to transform point cloud. Reason: %s", ex.what());
            return {};
        }
        return cloud_transformed;
    }

    ros::NodeHandle nh_;
    ros::ServiceServer service_;
    ros::Publisher debug_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string base_link_frame_;

    open3d::geometry::PointCloud pc_so_far;

    geometry_msgs::Pose tag_pose_in_base_link;

    open3d::geometry::AxisAlignedBoundingBox crop_box_;

    double voxel_size_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "active_perception_service");
    ActivePerceptionService loader;
    ros::spin();
    return 0;
}
