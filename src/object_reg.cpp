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
#include <map>
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
        sub_ = nh_.subscribe("/point_cloud", 1, &ObjectRegistration::point_cloud_callback, this);
        filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/filtered_point_cloud", 1);
        detection_pub_ = nh_.advertise<vision_msgs::Detection3DArray>("/detections", 1);

        visual_tools_.reset(
            new rviz_visual_tools::RvizVisualTools(base_link_frame_, "/rviz_debug", nh_));

        nh_.getParam("cube_mesh_path", cube_mesh_path_);

        open3d::io::ReadTriangleMeshOptions options;
        options.enable_post_processing = false;
        options.print_progress = false;

        open3d::geometry::TriangleMesh cube_mesh;
        open3d::io::ReadTriangleMesh(cube_mesh_path_, cube_mesh, options);

        cube_mesh_cloud_ = *cube_mesh.SamplePointsUniformly(500, true);
        cube_features_ = *open3d::pipelines::registration::ComputeFPFHFeature(cube_mesh_cloud_);

        ROS_INFO("Box detector node initialized.");
        ROS_INFO("Using frame %s as base link.", base_link_frame_.c_str());
    }

    // Callback function for processing incoming PointCloud2 messages
    void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        ROS_DEBUG("Received PointCloud2 message (%dx%d points) in frame '%s'.", cloud_msg->width,
            cloud_msg->height, cloud_msg->header.frame_id.c_str());

        visual_tools_->deleteAllMarkers();

        auto transformed_opt = transform_point_cloud(*cloud_msg, base_link_frame_);
        if (!transformed_opt) {
            return;
        }

        sensor_msgs::PointCloud2 transformed = std::move(*transformed_opt);
        open3d::geometry::PointCloud o3d_cloud = convert_cloud_ros_to_open3d(transformed);
        open3d::geometry::PointCloud filtered_cloud = filter_blue_points_hsv(o3d_cloud);
        sensor_msgs::PointCloud2 filtered_ros_cloud = convert_cloud_open3d_to_ros(
            filtered_cloud, transformed.header);

        auto cluster_clouds = cluster_cloud(filtered_cloud, 0.02, 10);
        update_objects(cluster_clouds, cloud_msg->header.stamp);
        draw_objects();

        filtered_pub_.publish(filtered_ros_cloud);

        visual_tools_->trigger();
    }

private:
    void draw_objects() {
        for (const auto& track: objects_) {
            if (!track.active) {
                continue;
            }

            Eigen::Isometry3d pose = track.pose;
            visual_tools_->publishMesh(pose, std::string("file://") + cube_mesh_path_,
                rviz_visual_tools::colors::TRANSLUCENT_DARK, 1.0, "mesh", track.id);

            Eigen::Isometry3d text_pose = pose;
            text_pose.translation() += Eigen::Vector3d(0, 0, 0.1);
            visual_tools_->publishText(text_pose, std::to_string(track.id),
                rviz_visual_tools::WHITE, rviz_visual_tools::MEDIUM, false);
        }
    }

    vision_msgs::Detection3DArray convert_objects_to_ros(std_msgs::Header header) {
        vision_msgs::Detection3DArray detection_array;
        detection_array.header = header;

        for (const auto& track: objects_) {
            if (!track.active) {
                continue;
            }

            vision_msgs::Detection3D detection;
            detection.header = header;

            vision_msgs::BoundingBox3D bbox;
            bbox.center = tf2::toMsg(track.pose);
            detection.bbox = bbox;

            vision_msgs::ObjectHypothesisWithPose hypothesis;
            hypothesis.id = track.id;
            hypothesis.pose.pose = tf2::toMsg(track.pose);
            hypothesis.score = 1.0;
            detection.results.push_back(hypothesis);

            detection_array.detections.push_back(detection);
        }

        return detection_array;
    }

    void update_objects(
        std::vector<open3d::geometry::PointCloud>& clusters, ros::Time detection_time) {
        using namespace open3d::pipelines::registration;

        std::vector<Eigen::Isometry3d> object_poses;

        for (auto& cluster: clusters) {
            if (cluster.points_.size() < 100) continue;

            // find most likely object correspondence
            Eigen::Vector3d cluster_approx_position = cluster.GetCenter();
            Eigen::Isometry3d closest_pose = Eigen::Isometry3d::Identity();
            for (const auto& object: objects_) {
                if (!object.active) {
                    continue;
                }

                if ((object.pose.translation() - cluster_approx_position).norm()
                    < (closest_pose.translation() - cluster_approx_position).norm()) {
                    closest_pose = object.pose;
                }
            }

            auto icp_result = RegistrationICP(cluster, cube_mesh_cloud_, 0.025,
                closest_pose.inverse().matrix(), TransformationEstimationPointToPlane());

            double fitness_cutoff = 0.7;

            RegistrationResult result;
            if (icp_result.fitness_ > fitness_cutoff) {
                result = icp_result;
            } else {
                cluster.EstimateNormals();
                auto cluster_features = *open3d::pipelines::registration::ComputeFPFHFeature(
                    cluster);

                RANSACConvergenceCriteria criteria(10000, 1.0);
                auto ransac_result = RegistrationRANSACBasedOnFeatureMatching(cluster,
                    cube_mesh_cloud_, cluster_features, cube_features_, false, 0.025,
                    TransformationEstimationPointToPoint(), 4, {}, criteria);

                result = RegistrationICP(cluster, cube_mesh_cloud_, 0.025,
                    ransac_result.transformation_, TransformationEstimationPointToPlane());
            }

            if (result.fitness_ <= fitness_cutoff) {
                continue;
            }

            Eigen::Isometry3d transform(result.transformation_);
            Eigen::Isometry3d pose = transform.inverse();

            object_poses.push_back(pose);
        }

        size_t start_size = objects_.size();
        std::vector<bool> claimed(start_size, false);

        for (const auto& pose: object_poses) {
            bool correspondence_found = false;

            for (size_t i = 0; i < start_size; i++) {
                Eigen::Vector3d delta = pose.translation() - objects_[i].pose.translation();

                if (!claimed[i] && delta.norm() < 0.05) {
                    correspondence_found = true;
                    claimed[i] = true;
                    objects_[i].active = true;
                    objects_[i].last_detected = detection_time;
                    objects_[i].pose = pose;
                    break;
                }
            }

            if (!correspondence_found) {
                objects_.push_back({
                    next_id++,
                    true,
                    pose,
                    detection_time,
                });
            }
        }

        for (size_t i = 0; i < start_size; i++) {
            if (!claimed[i]) {
                objects_[i].active = false;
            }
        }

        std::vector<Object> recent_objects;
        recent_objects.reserve(objects_.size());
        for (const auto& track: objects_) {
            if (!track.active && track.last_detected - detection_time > ros::Duration(30)) {
                continue;
            }

            recent_objects.push_back(track);
        }

        objects_ = std::move(recent_objects);
    }

    std::vector<open3d::geometry::PointCloud> cluster_cloud(
        const open3d::geometry::PointCloud& cloud, double dbscan_eps, size_t dbscan_min_pts) {
        std::vector<int> cluster_labels = cloud.ClusterDBSCAN(dbscan_eps, dbscan_min_pts);

        std::map<int, std::vector<size_t>> cluster_indices;
        for (size_t i = 0; i < cloud.points_.size(); ++i) {
            int label = cluster_labels[i];
            if (label != -1) {  // Ignore noise points (-1)
                cluster_indices[label].push_back(i);
            }
        }

        // Create a PointCloud for each cluster
        std::vector<open3d::geometry::PointCloud> cluster_clouds;
        cluster_clouds.reserve(cluster_indices.size());  // Reserve space

        for (const auto& pair: cluster_indices) {
            int label = pair.first;
            const std::vector<size_t>& indices = pair.second;

            open3d::geometry::PointCloud cluster_cloud;
            cluster_cloud.points_.reserve(indices.size());
            if (cloud.HasColors()) {
                cluster_cloud.colors_.reserve(indices.size());
            }

            for (size_t index: indices) {
                cluster_cloud.points_.push_back(cloud.points_[index]);
                if (cloud.HasColors()) {
                    cluster_cloud.colors_.push_back(cloud.colors_[index]);
                }
            }
            cluster_clouds.push_back(cluster_cloud);
        }

        return cluster_clouds;
    }

    open3d::geometry::PointCloud filter_blue_points_hsv(
        const open3d::geometry::PointCloud& input_cloud) {
        open3d::geometry::PointCloud filtered_cloud;

        if (!input_cloud.HasColors() || input_cloud.points_.empty()) {
            ROS_WARN("Input cloud for filtering has no colors or points. Returning empty cloud.");
            return filtered_cloud;  // Return empty if no colors or points
        }

        filtered_cloud.points_.reserve(input_cloud.points_.size());  // Reserve capacity
        filtered_cloud.colors_.reserve(input_cloud.colors_.size());

        const double h_low = (100.0 / 179.0) * 360.0;
        const double h_high = (140.0 / 179.0) * 360.0;
        const double s_low = 150.0 / 255.0;
        const double s_high = 1.0;
        const double v_low = 20.0 / 255.0;
        const double v_high = 1.0;

        for (size_t i = 0; i < input_cloud.points_.size(); ++i) {
            const Eigen::Vector3d& rgb = input_cloud.colors_[i];

            // Convert RGB [0,1] to HSV [0-360, 0-1, 0-1]
            auto [h, s, v] = rgb_to_hsv(rgb.x(), rgb.y(), rgb.z());

            // Check if the point falls within either red range
            bool in_range = (h >= h_low && h <= h_high && s >= s_low && s <= s_high && v >= v_low
                             && v <= v_high);

            if (in_range) {
                // Keep the point if it's in either range
                filtered_cloud.points_.push_back(input_cloud.points_[i]);
                filtered_cloud.colors_.push_back(
                    input_cloud.colors_[i]);  // Keep original RGB color
            }
        }

        return filtered_cloud;
    }

    std::tuple<double, double, double> rgb_to_hsv(double r, double g, double b) {
        double h = 0.0, s = 0.0, v = 0.0;
        double max_val = std::max({r, g, b});
        double min_val = std::min({r, g, b});
        double delta = max_val - min_val;

        v = max_val;  // Value is the maximum of r, g, b

        if (max_val > 1e-6) {     // Avoid division by zero or very small numbers
            s = delta / max_val;  // Saturation
        } else {
            // r = g = b = 0 (or very close)
            s = 0.0;
            h = 0.0;  // Undefined hue, often set to 0
            return {h, s, v};
        }

        if (delta > 1e-6) {  // If color is not grayscale
            if (max_val == r) {
                h = 60.0 * std::fmod(((g - b) / delta), 6.0);
            } else if (max_val == g) {
                h = 60.0 * (((b - r) / delta) + 2.0);
            } else {  // max_val == b
                h = 60.0 * (((r - g) / delta) + 4.0);
            }
        } else {
            h = 0.0;  // Hue is undefined for grayscale, set to 0
        }

        if (h < 0.0) {
            h += 360.0;  // Ensure hue is in [0, 360)
        }

        return {h, s, v};
    }

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
    ros::Publisher detection_pub_;

    size_t next_id = 0;
    std::vector<Object> objects_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string base_link_frame_ = "wx250s/base_link";

    rviz_visual_tools::RvizVisualToolsPtr visual_tools_;

    std::string cube_mesh_path_;
    open3d::geometry::PointCloud cube_mesh_cloud_;
    open3d::pipelines::registration::Feature cube_features_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_registration_node");
    ObjectRegistration loader;
    ros::spin();
    return 0;
}
