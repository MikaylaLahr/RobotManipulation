#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>  // Needed for TF lookup
#include <tf2_ros/buffer.h>              // Needed for TF lookup

#include <opencv2/opencv.hpp>

// Open3D includes - Ensure Open3D is correctly installed and findable by your build system
#include <open3d/Open3D.h>  // Main Open3D header

// Eigen for matrix manipulations (Open3D uses Eigen)
#include <Eigen/Dense>

#include <string>
#include <vector>
#include <memory>      // For std::unique_ptr
#include <functional>  // For std::bind and std::placeholders

/**
 * @class RealSenseO3DOdometry
 * @brief A ROS node that performs RGBD odometry using a RealSense camera and Open3D.
 *
 * This class subscribes to synchronized RGB and depth image topics, along with camera info,
 * from a RealSense camera. It then uses Open3D's RGBD odometry pipeline to estimate
 * the camera's pose and publishes it as a geometry_msgs::PoseStamped message and
 * optionally as a TF transform.
 */
class RealSenseO3DOdometry {
public:
    /**
     * @brief Constructor for RealSenseO3DOdometry.
     * @param nh ROS NodeHandle.
     * @param pnh ROS Private NodeHandle for accessing parameters.
     */
    RealSenseO3DOdometry(ros::NodeHandle& nh, ros::NodeHandle& pnh);

    /**
     * @brief Initializes the node.
     *
     * Loads parameters, fetches camera intrinsics, and sets up subscribers and publishers.
     * @return True if initialization is successful, false otherwise.
     */
    bool init();

private:
    // ROS-related members
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher pose_pub_;

    // TF related for initial pose
    tf2_ros::Buffer tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>
        MySyncPolicy;
    std::unique_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;

    // Open3D and odometry state members
    std::shared_ptr<open3d::geometry::RGBDImage> prev_rgbd_image_ptr_;
    open3d::camera::PinholeCameraIntrinsic o3d_intrinsics_;
    Eigen::Matrix4d global_pose_ = Eigen::Matrix4d::Identity();

    // Topic names and parameters
    std::string rgb_topic_;
    std::string depth_topic_;
    std::string camera_info_topic_;
    std::string pose_topic_out_;
    std::string odom_frame_id_;  // Frame ID for the odometry origin
    double
        depth_scale_factor_;  // Factor to convert depth image units to meters (e.g., 1000.0 for mm)
    double depth_trunc_value_;     // Maximum depth value in meters to consider for odometry
    std::string camera_frame_id_;  // The frame ID of the camera, used as the child frame in TF

    // Open3D Odometry options
    open3d::pipelines::odometry::OdometryOption odom_option_;

    // Callback for synchronized RGB and Depth images
    void imageCallback(
        const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg);

    // Helper methods
    /**
     * @brief Converts a ROS sensor_msgs::Image to an Open3D Image.
     * @param ros_img_msg Pointer to the ROS image message.
     * @param expected_ros_encoding The expected encoding of the ROS image (e.g.,
     * sensor_msgs::image_encodings::BGR8).
     * @param o3d_channels Number of channels for the Open3D image (e.g., 3 for color, 1 for depth).
     * @param o3d_bytes_per_channel Bytes per channel for the Open3D image (e.g., 1 for 8-bit, 2 for
     * 16-bit).
     * @return A shared pointer to the Open3D image, or nullptr on failure.
     */
    std::shared_ptr<open3d::geometry::Image> convertRosImageToOpen3D(
        const sensor_msgs::ImageConstPtr& ros_img_msg, const std::string& expected_ros_encoding,
        int o3d_channels, int o3d_bytes_per_channel);

    /**
     * @brief Fetches camera intrinsic parameters by waiting for a CameraInfo message.
     * @return True if intrinsics were successfully fetched and are valid, false otherwise.
     */
    bool fetchCameraIntrinsics();

    bool fetchInitialPoseViaTf();
};

RealSenseO3DOdometry::RealSenseO3DOdometry(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh),
      pnh_(pnh),
      rgb_sub_(nh_, "", 1),   // Dummy topic name, will be set in init()
      depth_sub_(nh_, "", 1)  // Dummy topic name, will be set in init()
{
    // Parameters are loaded in init()
}

bool RealSenseO3DOdometry::fetchCameraIntrinsics() {
    ROS_INFO_STREAM("Waiting for camera intrinsics on topic: " << camera_info_topic_ << "...");
    sensor_msgs::CameraInfoConstPtr info_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(
        camera_info_topic_, nh_, ros::Duration(20.0)  // Timeout after 20 seconds
    );

    if (!info_msg) {
        ROS_ERROR_STREAM("Failed to receive camera intrinsics from topic " << camera_info_topic_
                                                                           << " within timeout.");
        return false;
    }

    // Set Open3D intrinsics from the received CameraInfo message
    o3d_intrinsics_.SetIntrinsics(info_msg->width, info_msg->height,
        info_msg->K[0],  // fx - principal focal length in x
        info_msg->K[4],  // fy - principal focal length in y
        info_msg->K[2],  // cx - principal point x-coordinate
        info_msg->K[5]   // cy - principal point y-coordinate
    );

    // Validate the received intrinsics
    if (o3d_intrinsics_.width_ <= 0 || o3d_intrinsics_.height_ <= 0
        || o3d_intrinsics_.intrinsic_matrix_(0, 0) <= 0) {  // Check if fx is valid
        ROS_ERROR(
            "Invalid camera intrinsics received! fx = %f, fy = %f, cx = %f, cy = %f, w = %d, h = "
            "%d",
            info_msg->K[0], info_msg->K[4], info_msg->K[2], info_msg->K[5], info_msg->width,
            info_msg->height);
        return false;
    }

    ROS_INFO("Camera intrinsics received and set successfully:");
    ROS_INFO_STREAM("Width: " << o3d_intrinsics_.width_ << ", Height: " << o3d_intrinsics_.height_);
    ROS_INFO_STREAM("Intrinsic Matrix:\n" << o3d_intrinsics_.intrinsic_matrix_);
    return true;
}

bool RealSenseO3DOdometry::fetchInitialPoseViaTf() {
    geometry_msgs::TransformStamped transform_stamped;
    try {
        ROS_INFO_STREAM("Attempting to look up initial TF transform from '"
                        << odom_frame_id_ << "' to '" << camera_frame_id_ << "'...");
        // Wait for the transform to be available, with a timeout.
        // ros::Time(0) means "latest available"
        transform_stamped = tf_buffer_.lookupTransform(odom_frame_id_, camera_frame_id_,
            ros::Time(0), ros::Duration(10.0));  // 10 second timeout

        Eigen::Vector3d translation(transform_stamped.transform.translation.x,
            transform_stamped.transform.translation.y, transform_stamped.transform.translation.z);
        Eigen::Quaterniond rotation(transform_stamped.transform.rotation.w,
            transform_stamped.transform.rotation.x, transform_stamped.transform.rotation.y,
            transform_stamped.transform.rotation.z);
        rotation.normalize();

        global_pose_ = Eigen::Matrix4d::Identity();
        global_pose_.block<3, 3>(0, 0) = rotation.toRotationMatrix();
        global_pose_.block<3, 1>(0, 3) = translation;

        ROS_INFO_STREAM("Initial pose set from TF transform: \n" << global_pose_);
        return true;

    } catch (tf2::TransformException& ex) {
        ROS_WARN_STREAM("Could not get initial TF transform from '"
                        << odom_frame_id_ << "' to '" << camera_frame_id_ << "': " << ex.what()
                        << ". Defaulting to identity pose.");
        global_pose_ = Eigen::Matrix4d::Identity();  // Default to identity
        ROS_INFO_STREAM("Default initial pose (Identity):\n" << global_pose_);
        return true;  // Still true as we successfully defaulted
    }
    return false;  // Should not be reached due to catch or success
}

bool RealSenseO3DOdometry::init() {
    // Initialize TF listener
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(tf_buffer_);

    // Load ROS parameters
    pnh_.param<std::string>("rgb_topic", rgb_topic_, "/camera/color/image_raw");
    pnh_.param<std::string>(
        "depth_topic", depth_topic_, "/camera/aligned_depth_to_color/image_raw");
    pnh_.param<std::string>("camera_info_topic", camera_info_topic_, "/camera/color/camera_info");
    pnh_.param<std::string>("pose_topic_out", pose_topic_out_, "/realsense_odometry/pose");
    pnh_.param<std::string>("odom_frame_id", odom_frame_id_, "odom");
    pnh_.param<double>(
        "depth_scale_factor", depth_scale_factor_, 1000.0);  // Default for RealSense (mm to m)
    pnh_.param<double>(
        "depth_trunc_value", depth_trunc_value_, 3.0);  // Default max depth 3m for odometry
    pnh_.param<std::string>("camera_frame_id", camera_frame_id_, "camera_odom_link");

    // Load Open3D odometry options from parameters
    std::vector<int> iterations;
    if (pnh_.getParam("odom_iterations_per_pyramid", iterations)) {
        odom_option_.iteration_number_per_pyramid_level_.assign(
            iterations.begin(), iterations.end());
    } else {
        odom_option_.iteration_number_per_pyramid_level_ = {
            10, 5, 3};  // Default iteration counts per pyramid level
    }
    pnh_.param(
        "odom_depth_min", odom_option_.depth_min_, 0.1);  // Minimum depth to consider (meters)
    pnh_.param("odom_depth_max", odom_option_.depth_max_,
        depth_trunc_value_);  // Maximum depth to consider (meters)
    // Example: pnh_.param("odom_depth_diff_max", odom_option_.depth_diff_max_, 0.07); // Max depth
    // difference for correspondence

    // Fetch camera intrinsics. This is a blocking call.
    if (!fetchCameraIntrinsics()) {
        ROS_ERROR("Failed to initialize camera intrinsics. Node will not start image processing.");
        return false;  // Indicate initialization failure
    }

    // Fetch initial pose via TF.
    if (!fetchInitialPoseViaTf()) {
        ROS_ERROR(
            "Failed to set initial pose via TF (this path should ideally not be hit due to "
            "defaulting logic). Node will not start.");
        return false;
    }

    // Setup subscribers for RGB and Depth images using message_filters for synchronization
    rgb_sub_.subscribe(nh_, rgb_topic_, 5);      // Queue size 5
    depth_sub_.subscribe(nh_, depth_topic_, 5);  // Queue size 5

    sync_ = std::make_unique<message_filters::Synchronizer<MySyncPolicy>>(
        MySyncPolicy(10), rgb_sub_, depth_sub_);  // Approx time policy queue size 10
    sync_->registerCallback(std::bind(
        &RealSenseO3DOdometry::imageCallback, this, std::placeholders::_1, std::placeholders::_2));

    // Setup publisher for the estimated pose
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(
        pose_topic_out_, 10);  // Publisher queue size 10

    ROS_INFO_STREAM(ros::this_node::getName() << " initialized successfully.");
    ROS_INFO_STREAM("Subscribing to RGB topic: " << rgb_topic_);
    ROS_INFO_STREAM("Subscribing to Depth topic: " << depth_topic_);
    ROS_INFO_STREAM("Publishing Pose to topic: " << pose_topic_out_);
    return true;  // Indicate successful initialization
}

std::shared_ptr<open3d::geometry::Image> RealSenseO3DOdometry::convertRosImageToOpen3D(
    const sensor_msgs::ImageConstPtr& ros_img_msg, const std::string& expected_ros_encoding,
    int o3d_channels, int o3d_bytes_per_channel) {
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
            "Channel mismatch during conversion. CV_Mat channels: %d, Expected Open3D channels: %d",
            cv_ptr->image.channels(), o3d_channels);
        return nullptr;
    }

    // Create an Open3D image and prepare its dimensions and data type
    auto o3d_image = std::make_shared<open3d::geometry::Image>();
    o3d_image->Prepare(cv_ptr->image.cols, cv_ptr->image.rows, o3d_channels, o3d_bytes_per_channel);

    // Calculate the expected data size for verification
    size_t cv_data_size = cv_ptr->image.total() * cv_ptr->image.elemSize();
    size_t o3d_buffer_size = static_cast<size_t>(
        o3d_image->width_ * o3d_image->height_ * o3d_channels * o3d_bytes_per_channel);

    if (cv_data_size != o3d_buffer_size) {
        ROS_ERROR(
            "Data size mismatch during Open3D image preparation. CV Mat data size: %zu, Expected "
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
                static_cast<size_t>(
                    cv_ptr->image.cols * o3d_channels * o3d_bytes_per_channel));  // Size of one row
        }
    }
    return o3d_image;
}

void RealSenseO3DOdometry::imageCallback(
    const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg) {
    // Intrinsics are guaranteed to be available here due to the init() logic.
    ROS_DEBUG_ONCE(
        "Received first synchronized RGB and Depth pair for odometry processing.");  // Log only
                                                                                     // once

    // Convert ROS RGB image to Open3D format.
    // RealSense typically publishes BGR8. Open3D's CreateFromColorAndDepth can handle BGR.
    auto o3d_color_img = convertRosImageToOpen3D(
        rgb_msg, sensor_msgs::image_encodings::BGR8, 3, 1);  // 3 channels, 1 byte/channel

    // Convert ROS Depth image to Open3D format.
    // RealSense depth is typically 16UC1 (16-bit unsigned, single channel).
    auto o3d_depth_img = convertRosImageToOpen3D(
        depth_msg, sensor_msgs::image_encodings::TYPE_16UC1, 1, 2);  // 1 channel, 2 bytes/channel

    if (!o3d_color_img || !o3d_depth_img) {
        ROS_ERROR_THROTTLE(
            1.0, "Failed to convert one or both ROS images to Open3D format. Skipping frame.");
        return;
    }

    // Create an Open3D RGBDImage object from the color and depth images.
    // depth_scale_factor: Converts depth units (e.g., mm from RealSense) to meters for Open3D.
    // depth_trunc_value: Max depth in meters. Pixels beyond this are ignored.
    // convert_rgb_to_intensity: False, as we are providing an RGB image.
    auto current_rgbd_image_ptr = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
        *o3d_color_img, *o3d_depth_img, depth_scale_factor_, depth_trunc_value_, false);

    if (!current_rgbd_image_ptr || current_rgbd_image_ptr->IsEmpty()) {
        ROS_ERROR_THROTTLE(1.0,
            "Failed to create Open3D RGBDImage or the resulting image is empty. Skipping frame.");
        return;
    }

    // If this is the first valid frame, store it as the previous frame and return.
    if (!prev_rgbd_image_ptr_) {
        ROS_INFO("Setting the first valid frame as the previous frame for odometry computation.");
        prev_rgbd_image_ptr_ = current_rgbd_image_ptr;
        return;
    }

    // Initialize the transformation guess for odometry (identity matrix).
    Eigen::Matrix4d odo_init = Eigen::Matrix4d::Identity();

    // Perform RGBD odometry using Open3D.
    // This estimates the transformation from the previous (source) RGBD image to the current
    // (target) RGBD image.
    const auto& [success, transform, _] = open3d::pipelines::odometry::ComputeRGBDOdometry(
        *prev_rgbd_image_ptr_, *current_rgbd_image_ptr,  // Source and Target RGBD images
        o3d_intrinsics_,                                 // Camera intrinsic parameters
        odo_init,                                        // Initial transformation guess
        open3d::pipelines::odometry::RGBDOdometryJacobianFromHybridTerm(),  // Jacobian computation
                                                                            // method
        odom_option_                                                        // Odometry options
    );

    if (success) {
        ROS_INFO_STREAM(global_pose_);

        // Odometry was successful. Update the global pose.
        // result.transformation_ is T_previous_current (transform from previous frame to current
        // frame) global_pose_ (world to previous) * T_previous_current = new global_pose_ (world to
        // current)
        global_pose_ = global_pose_ * Eigen::Isometry3d(transform).inverse().matrix();

        // Create and populate PoseStamped message
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header.stamp = rgb_msg->header.stamp;  // Use timestamp from the current RGB image
        pose_msg.header.frame_id = odom_frame_id_;      // Pose is in the odometry frame

        // Set position
        pose_msg.pose.position.x = global_pose_(0, 3);
        pose_msg.pose.position.y = global_pose_(1, 3);
        pose_msg.pose.position.z = global_pose_(2, 3);

        // Set orientation (convert rotation matrix to quaternion)
        Eigen::Matrix3d rotation_matrix = global_pose_.block<3, 3>(0, 0);
        Eigen::Quaterniond quaternion(rotation_matrix);
        quaternion.normalize();  // Ensure it's a unit quaternion

        pose_msg.pose.orientation.x = quaternion.x();
        pose_msg.pose.orientation.y = quaternion.y();
        pose_msg.pose.orientation.z = quaternion.z();
        pose_msg.pose.orientation.w = quaternion.w();

        pose_pub_.publish(pose_msg);
    } else {
        ROS_WARN_THROTTLE(1.0, "Odometry computation failed for this frame pair.");
    }

    // Update the previous RGBD image pointer for the next iteration.
    prev_rgbd_image_ptr_ = current_rgbd_image_ptr;
}

int main(int argc, char** argv) {
    // Initialize the ROS node
    ros::init(argc, argv, "realsense_o3d_odometry_node");

    // Create ROS NodeHandles
    ros::NodeHandle nh;        // Public NodeHandle
    ros::NodeHandle pnh("~");  // Private NodeHandle for parameters

    // Create an instance of the odometry class
    RealSenseO3DOdometry odometry_node(nh, pnh);

    // Initialize the odometry node (loads params, fetches intrinsics, sets up subs/pubs)
    if (odometry_node.init()) {
        ROS_INFO("RealSense Open3D Odometry node spinning.");
        ros::spin();  // Keep the node alive and processing callbacks
    } else {
        ROS_ERROR("Failed to initialize RealSense Open3D Odometry node. Shutting down.");
    }

    return 0;
}
