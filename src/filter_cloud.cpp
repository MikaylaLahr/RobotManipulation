#include <open3d/geometry/BoundingVolume.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/FileFormatIO.h>
#include <open3d/io/ModelIO.h>
#include <open3d/io/TriangleMeshIO.h>
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

#include <open3d/Open3D.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <optional>
#include <vector>
#include "ros/console.h"

struct Object {
    size_t id;
    bool active;
    Eigen::Isometry3d pose;
    ros::Time last_detected;
};

class ObjectRegistration {
public:
    ObjectRegistration(): nh_("~"), tf_listener_(tf_buffer_) {
        // Read parameters
        nh_.param<double>("crop_min_x", crop_min_x_, 0.0);
        nh_.param<double>("crop_min_y", crop_min_y_, -0.2);
        nh_.param<double>("crop_min_z", crop_min_z_, 0.0);
        nh_.param<double>("crop_max_x", crop_max_x_, 0.6);
        nh_.param<double>("crop_max_y", crop_max_y_, 0.2);
        nh_.param<double>("crop_max_z", crop_max_z_, 0.15);
        nh_.param<double>("voxel_size", voxel_size_, 0.005);
        nh_.param<std::string>("base_link_frame", base_link_frame_, "wx250s/base_link");

        sub_ = nh_.subscribe("/point_cloud", 1, &ObjectRegistration::point_cloud_callback, this);
        filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/filtered_point_cloud", 1);

        ROS_INFO("Cloud filter node initialized.");
        ROS_INFO("Using frame %s as base link.", base_link_frame_.c_str());
        ROS_INFO("Crop Box Min: [%.3f, %.3f, %.3f]", crop_min_x_, crop_min_y_, crop_min_z_);
        ROS_INFO("Crop Box Max: [%.3f, %.3f, %.3f]", crop_max_x_, crop_max_y_, crop_max_z_);
        ROS_INFO("Voxel Size: %.4f", voxel_size_);
    }

    // Callback function for processing incoming PointCloud2 messages
    void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        ROS_DEBUG("Received PointCloud2 message (%dx%d points) in frame '%s'.", cloud_msg->width,
            cloud_msg->height, cloud_msg->header.frame_id.c_str());

        auto transformed_opt = transform_point_cloud(*cloud_msg, base_link_frame_);
        if (!transformed_opt) {
            return;
        }

        sensor_msgs::PointCloud2 transformed = std::move(*transformed_opt);
        open3d::geometry::PointCloud o3d_cloud = convert_cloud_ros_to_open3d(transformed);
        open3d::geometry::PointCloud cropped = *o3d_cloud.Crop(
            open3d::geometry::AxisAlignedBoundingBox(
                Eigen::Vector3d(crop_min_x_, crop_min_y_, crop_min_z_),
                Eigen::Vector3d(crop_max_x_, crop_max_y_, crop_max_z_)));
        open3d::geometry::PointCloud downsampled = *cropped.VoxelDownSample(
            voxel_size_);  // Downsample the cropped cloud

        // Publish the downsampled (and cropped) cloud
        sensor_msgs::PointCloud2 filtered_msg = convert_cloud_open3d_to_ros(
            downsampled, transformed.header);
        filtered_pub_.publish(filtered_msg);
    }

private:
    open3d::geometry::PointCloud convert_cloud_ros_to_open3d(
        const sensor_msgs::PointCloud2& ros_cloud) {
        open3d::geometry::PointCloud o3d_cloud;

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(ros_cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(ros_cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(ros_cloud, "z");
        sensor_msgs::PointCloud2ConstIterator<float> iter_rgb(ros_cloud, "rgb");

        size_t num_points = ros_cloud.width * ros_cloud.height;
        size_t valid_points = 0;  // Counter for valid points added

        // Resize buffers initially, we might shrink later if points are invalid
        o3d_cloud.points_.resize(num_points);
        o3d_cloud.colors_.resize(num_points);

        for (size_t i = 0; i < num_points; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_rgb) {
            // Check for NaN/Inf coordinates
            if (std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z)
                || std::isinf(*iter_x) || std::isinf(*iter_y) || std::isinf(*iter_z)) {
                continue;  // Skip invalid points
            }
            o3d_cloud.points_[valid_points] = Eigen::Vector3d(*iter_x, *iter_y, *iter_z);

            // --- RGB Extraction ---
            uint8_t r = 0, g = 0, b = 0;

            float rgb_data = *iter_rgb;

            // Packed FLOAT32 - Need to reinterpret bits as uint32
            uint32_t packed_rgb;
            std::memcpy(&packed_rgb, &rgb_data, sizeof(uint32_t));
            r = (packed_rgb >> 16) & 0xFF;
            g = (packed_rgb >> 8) & 0xFF;
            b = packed_rgb & 0xFF;

            // Normalize color values to [0.0, 1.0] for Open3D
            double r_norm = static_cast<double>(r) / 255.0;
            double g_norm = static_cast<double>(g) / 255.0;
            double b_norm = static_cast<double>(b) / 255.0;

            o3d_cloud.colors_[valid_points] = Eigen::Vector3d(r_norm, g_norm, b_norm);
            valid_points++;
        }
        // Resize down to the actual number of valid points processed
        o3d_cloud.points_.resize(valid_points);
        o3d_cloud.colors_.resize(valid_points);

        return o3d_cloud;
    }

    sensor_msgs::PointCloud2 convert_cloud_open3d_to_ros(
        const open3d::geometry::PointCloud& o3d_cloud, const std_msgs::Header& header) {
        sensor_msgs::PointCloud2 ros_cloud;

        ros_cloud.header = header;

        ros_cloud.height = 1;  // Default to unorganized cloud
        ros_cloud.width = o3d_cloud.points_.size();

        bool has_colors = o3d_cloud.HasColors();

        // Define PointFields
        ros_cloud.fields.clear();  // Clear existing fields
        int offset = 0;

        // Add XYZ fields (FLOAT32)
        sensor_msgs::PointField field_x, field_y, field_z, field_rgb;
        field_x.name = "x";
        field_x.offset = offset;
        field_x.datatype = sensor_msgs::PointField::FLOAT32;
        field_x.count = 1;
        ros_cloud.fields.push_back(field_x);
        offset += sizeof(float);

        field_y.name = "y";
        field_y.offset = offset;
        field_y.datatype = sensor_msgs::PointField::FLOAT32;
        field_y.count = 1;
        ros_cloud.fields.push_back(field_y);
        offset += sizeof(float);

        field_z.name = "z";
        field_z.offset = offset;
        field_z.datatype = sensor_msgs::PointField::FLOAT32;
        field_z.count = 1;
        ros_cloud.fields.push_back(field_z);
        offset += sizeof(float);

        // Add RGB field (packed into a FLOAT32) if colors exist
        if (has_colors) {
            field_rgb.name = "rgb";
            field_rgb.offset = offset;
            field_rgb.datatype =
                sensor_msgs::PointField::FLOAT32;  // Standard practice in PCL/ROS for packed RGB
            field_rgb.count = 1;
            ros_cloud.fields.push_back(field_rgb);
            offset += sizeof(float);
        }

        // Set PointCloud2 metadata
        ros_cloud.point_step = offset;  // Size of one point in bytes
        ros_cloud.row_step = ros_cloud.point_step * ros_cloud.width;
        ros_cloud.is_dense = true;       // Open3D clouds are typically dense after processing NaNs
        ros_cloud.is_bigendian = false;  // Typically false on x86/amd64

        // Allocate data buffer
        ros_cloud.data.resize(
            ros_cloud.row_step * ros_cloud.height);  // Should be row_step since height=1

        // Fill data buffer
        for (size_t i = 0; i < ros_cloud.width; ++i) {
            // Calculate point's base offset in the data buffer
            size_t base_offset = i * ros_cloud.point_step;

            // Copy XYZ (casting Eigen::Vector3d (double) to float)
            float x = static_cast<float>(o3d_cloud.points_[i].x());
            float y = static_cast<float>(o3d_cloud.points_[i].y());
            float z = static_cast<float>(o3d_cloud.points_[i].z());
            std::memcpy(&ros_cloud.data[base_offset + field_x.offset], &x, sizeof(float));
            std::memcpy(&ros_cloud.data[base_offset + field_y.offset], &y, sizeof(float));
            std::memcpy(&ros_cloud.data[base_offset + field_z.offset], &z, sizeof(float));

            // Copy RGB if present
            if (has_colors) {
                // Clamp and convert normalized double [0,1] color to uint8 [0,255]
                uint8_t r = static_cast<uint8_t>(
                    std::max(0.0, std::min(1.0, o3d_cloud.colors_[i].x())) * 255.0);
                uint8_t g = static_cast<uint8_t>(
                    std::max(0.0, std::min(1.0, o3d_cloud.colors_[i].y())) * 255.0);
                uint8_t b = static_cast<uint8_t>(
                    std::max(0.0, std::min(1.0, o3d_cloud.colors_[i].z())) * 255.0);

                // Pack RGB into a single uint32
                uint32_t packed_rgb = (static_cast<uint32_t>(r) << 16)
                                      | (static_cast<uint32_t>(g) << 8)
                                      | (static_cast<uint32_t>(b));

                // Reinterpret the uint32 as a float and copy
                float rgb_float;
                std::memcpy(&rgb_float, &packed_rgb, sizeof(uint32_t));
                std::memcpy(
                    &ros_cloud.data[base_offset + field_rgb.offset], &rgb_float, sizeof(float));
            }
        }

        return ros_cloud;
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
    ros::Subscriber sub_;
    ros::Publisher filtered_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string base_link_frame_;

    // Parameters
    double crop_min_x_, crop_min_y_, crop_min_z_;
    double crop_max_x_, crop_max_y_, crop_max_z_;
    double voxel_size_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_registration_node");
    ObjectRegistration loader;
    ros::spin();
    return 0;
}
