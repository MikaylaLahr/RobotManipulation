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
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>

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
    ActivePerceptionService()
        : nh_("~"),
          tf_listener_(tf_buffer_),
          volume_(nh_.param<double>("voxel_size", 0.005), 0.01,
              open3d::pipelines::integration::TSDFVolumeColorType::RGB8) {
        nh_.param<double>("crop_min_x", crop_box_.min_bound_.x(), 0.0);
        nh_.param<double>("crop_min_y", crop_box_.min_bound_.y(), -0.2);
        nh_.param<double>("crop_min_z", crop_box_.min_bound_.z(), 0.0);
        nh_.param<double>("crop_max_x", crop_box_.max_bound_.x(), 0.6);
        nh_.param<double>("crop_max_y", crop_box_.max_bound_.y(), 0.2);
        nh_.param<double>("crop_max_z", crop_box_.max_bound_.z(), 0.15);
        nh_.param<std::string>("base_link_frame", base_link_frame_, "wx250s/base_link");

        service_ = nh_.advertiseService(
            "take_snapshot", &ActivePerceptionService::take_snapshot_callback, this);
        debug_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("debug_cloud", 1, true);

        fetchCameraIntrinsics();

        ROS_INFO("Active perception node initialized.");
    }

private:
    std::shared_ptr<open3d::geometry::Image> convertRosImageToOpen3D(
        const sensor_msgs::ImageConstPtr& ros_img_msg, const std::string& expected_ros_encoding,
        const std::string& desired_encoding, int o3d_channels, int o3d_bytes_per_channel) {
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            // Convert ROS image message to OpenCV CvImage.
            // toCvShare avoids a copy if the encoding matches and the data is not modified.
            cv_ptr = cv_bridge::toCvShare(ros_img_msg, expected_ros_encoding);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s. Target encoding: %s, Image encoding: %s", e.what(),
                expected_ros_encoding.c_str(), ros_img_msg->encoding.c_str());
            return nullptr;
        }

        if (!cv_ptr || !cv_ptr->image.data) {
            ROS_ERROR("Failed to convert ROS image to CV Mat or image data is null.");
            return nullptr;
        }

        // Sanity check for channel consistency
        if (cv_ptr->image.channels() != o3d_channels) {
            ROS_ERROR(
                "Channel mismatch during conversion. CV_Mat channels: %d, Expected Open3D "
                "channels: %d",
                cv_ptr->image.channels(), o3d_channels);
            return nullptr;
        }

        cv_ptr = cv_bridge::cvtColor(cv_ptr, desired_encoding);

        // Create an Open3D image and prepare its dimensions and data type
        auto o3d_image = std::make_shared<open3d::geometry::Image>();
        o3d_image->Prepare(
            cv_ptr->image.cols, cv_ptr->image.rows, o3d_channels, o3d_bytes_per_channel);

        // Calculate the expected data size for verification
        size_t cv_data_size = cv_ptr->image.total() * cv_ptr->image.elemSize();
        size_t o3d_buffer_size = static_cast<size_t>(
            o3d_image->width_ * o3d_image->height_ * o3d_channels * o3d_bytes_per_channel);

        if (cv_data_size != o3d_buffer_size) {
            ROS_ERROR(
                "Data size mismatch during Open3D image preparation. CV Mat data size: %zu, "
                "Expected "
                "O3D buffer size: %zu",
                cv_data_size, o3d_buffer_size);
            return nullptr;
        }

        // Copy data from cv::Mat to Open3D image buffer
        if (cv_ptr->image.isContinuous()) {
            // If cv::Mat data is continuous, a single memcpy can be used
            memcpy(o3d_image->data_.data(), cv_ptr->image.data, cv_data_size);
        } else {
            // If not continuous, copy row by row (less efficient but necessary)
            ROS_WARN_THROTTLE(5.0, "cv::Mat is not continuous for %s image. Copying row by row.",
                (o3d_channels == 1 ? "depth" : "color"));
            for (int i = 0; i < cv_ptr->image.rows; ++i) {
                memcpy(o3d_image->data_.data()
                           + i * cv_ptr->image.cols * o3d_channels * o3d_bytes_per_channel,
                    cv_ptr->image.ptr<uint8_t>(i),  // Get pointer to i-th row
                    static_cast<size_t>(cv_ptr->image.cols * o3d_channels
                                        * o3d_bytes_per_channel));  // Size of one row
            }
        }

        return o3d_image;
    }

    bool fetchCameraIntrinsics() {
        sensor_msgs::CameraInfoConstPtr info_msg =
            ros::topic::waitForMessage<sensor_msgs::CameraInfo>(
                "/camera_info", nh_, ros::Duration(20.0)  // Timeout after 20 seconds
            );

        if (!info_msg) {
            ROS_ERROR_STREAM("Failed to receive camera intrinsics from camera info topic");
            return false;
        }

        // Set Open3D intrinsics from the received CameraInfo message
        pinhole_model_.SetIntrinsics(info_msg->width, info_msg->height,
            info_msg->K[0],  // fx - principal focal length in x
            info_msg->K[4],  // fy - principal focal length in y
            info_msg->K[2],  // cx - principal point x-coordinate
            info_msg->K[5]   // cy - principal point y-coordinate
        );

        // Validate the received intrinsics
        if (pinhole_model_.width_ <= 0 || pinhole_model_.height_ <= 0
            || pinhole_model_.intrinsic_matrix_(0, 0) <= 0) {  // Check if fx is valid
            ROS_ERROR(
                "Invalid camera intrinsics received! fx = %f, fy = %f, cx = %f, cy = %f, w = %d, h "
                "= "
                "%d",
                info_msg->K[0], info_msg->K[4], info_msg->K[2], info_msg->K[5], info_msg->width,
                info_msg->height);
            return false;
        }

        ROS_INFO("Camera intrinsics received and set successfully:");
        return true;
    }

    bool take_snapshot_callback(robot_manipulation::ActivePerception::Request& req,
        robot_manipulation::ActivePerception::Response& res) {
        sensor_msgs::ImageConstPtr depth = ros::topic::waitForMessage<sensor_msgs::Image>(
            "/depth_image", ros::Duration(1.0));

        if (!depth) {
            ROS_WARN("Unable to recieve depth frame");
        }

        sensor_msgs::ImageConstPtr color = ros::topic::waitForMessage<sensor_msgs::Image>(
            "/color_image", ros::Duration(1.0));

        if (!color) {
            ROS_WARN("Unable to recieve color frame");
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

        assert(detection.pose.header.frame_id == color.header.frame_id);

        auto o3d_depth = convertRosImageToOpen3D(depth, sensor_msgs::image_encodings::TYPE_16UC1,
            sensor_msgs::image_encodings::TYPE_16UC1, 1, 2);
        auto o3d_color = convertRosImageToOpen3D(
            color, sensor_msgs::image_encodings::BGR8, sensor_msgs::image_encodings::RGB8, 3, 1);

        if (!o3d_color || !o3d_depth) {
            ROS_WARN("Failed to convert one or both ROS images to Open3D format. Skipping frame.");
            return false;
        }

        auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
            *o3d_color, *o3d_depth, 1000.0, 3.0, false);

        if (first_) {
            // first time around, just assume the camera pose is roughly correct
            // when we get more clouds we will not use the robot's odometry, only the tag to
            // align scans
            geometry_msgs::PoseStamped stamped;
            stamped.header = detection.pose.header;
            stamped.pose = detection.pose.pose.pose;

            tag_pose_in_base_link = tf_buffer_.transform(stamped, base_link_frame_).pose;

            auto camera_base_link = tf_buffer_.lookupTransform(
                detection.pose.header.frame_id, base_link_frame_, detection.pose.header.stamp);
            Eigen::Isometry3d eigen_camera_base_link = tf2::transformToEigen(camera_base_link);

            volume_.Integrate(*rgbd, pinhole_model_, eigen_camera_base_link.matrix());

            first_ = false;
        } else {
            // now need to reconstruct camera -> base_link transform from detection.pose and
            // tag_pose_in_base_link
            Eigen::Isometry3d eigen_detection_pose;
            tf2::fromMsg(detection.pose.pose.pose, eigen_detection_pose);

            Eigen::Isometry3d eigen_tag_pose_base_link;
            tf2::fromMsg(tag_pose_in_base_link, eigen_tag_pose_base_link);

            Eigen::Isometry3d camera_base_link = eigen_detection_pose
                                                 * eigen_tag_pose_base_link.inverse();

            volume_.Integrate(*rgbd, pinhole_model_, camera_base_link.matrix());
        }

        open3d::geometry::PointCloud cloud = *volume_.ExtractPointCloud()->Crop(crop_box_);
        open3d_conversions::open3dToRos(cloud, res.collected_cloud, base_link_frame_);
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

    geometry_msgs::Pose tag_pose_in_base_link;

    open3d::geometry::AxisAlignedBoundingBox crop_box_;

    open3d::pipelines::integration::ScalableTSDFVolume volume_;
    bool first_ = true;

    open3d::camera::PinholeCameraIntrinsic pinhole_model_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "active_perception_service");
    ActivePerceptionService loader;
    ros::spin();
    return 0;
}
