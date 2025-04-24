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
#include <boost/filesystem/operations.hpp>
#include <limits>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>
#include <boost/filesystem.hpp>

#include "Eigen/src/Geometry/Transform.h"
#include "gs_matching.h"

enum ObjectType { Cube, Milk, Wine, Eggs, ToiletPaper };

struct ObjectTypeInfo {
    std::string mesh_path;
    open3d::geometry::PointCloud point_cloud;
    open3d::pipelines::registration::Feature features;
};

struct Object {
    size_t id;
    bool active;
    Eigen::Isometry3d pose;
    ros::Time last_detected;
    ObjectType type;
};

class ObjectRegistration {
public:
    ObjectRegistration(): nh_("~"), tf_listener_(tf_buffer_) {
        sub1_ = nh_.subscribe("/point_cloud", 1, &ObjectRegistration::point_cloud_callback, this);
        filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/filtered_point_cloud", 1);
        detection_pub_ = nh_.advertise<vision_msgs::Detection3DArray>("/detections", 1);

        visual_tools_.reset(
            new rviz_visual_tools::RvizVisualTools(base_link_frame_, "/rviz_debug", nh_));

        read_meshes();

        ROS_INFO("Box detector node initialized.");
        ROS_INFO("Using frame %s as base link.", base_link_frame_.c_str());
        ROS_INFO("Found %zu object meshes.", object_types_.size());
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

        auto cluster_clouds = cluster_cloud(o3d_cloud, 0.02, 10);
        update_objects(std::move(cluster_clouds), cloud_msg->header.stamp);

        draw_objects();

        // TODO: use clusters to publish filtered cloud
        visual_tools_->trigger();
    }

private:
    void read_meshes() {
        std::string mesh_folder;
        nh_.getParam("mesh_folder", mesh_folder);

        std::unordered_map<std::string, ObjectType> name_type_map{
            {"cube", ObjectType::Cube},
            {"milk", ObjectType::Milk},
            {"wine", ObjectType::Wine},
            {"eggs", ObjectType::Eggs},
            {"toilet_paper", ObjectType::ToiletPaper},
        };

        open3d::io::ReadTriangleMeshOptions options;
        options.enable_post_processing = false;
        options.print_progress = false;

        for (const auto& entry:
            boost::make_iterator_range(boost::filesystem::directory_iterator(mesh_folder), {})) {
            open3d::geometry::TriangleMesh mesh;
            open3d::io::ReadTriangleMesh(entry.path().string(), mesh, options);

            auto point_cloud = *mesh.SamplePointsUniformly(500, true);
            auto features = *open3d::pipelines::registration::ComputeFPFHFeature(point_cloud);

            ObjectTypeInfo data{entry.path().string(), point_cloud, features};

            std::string filename = entry.path().filename().replace_extension("").string();
            ObjectType type = name_type_map[filename];

            object_types_[type] = data;
        }
    }

    void draw_objects() {
        for (const auto& track: objects_) {
            if (!track.active) {
                continue;
            }

            ObjectTypeInfo info = object_types_[track.type];

            Eigen::Isometry3d pose = track.pose;
            visual_tools_->publishMesh(pose, std::string("file://") + info.mesh_path,
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
        std::vector<open3d::geometry::PointCloud> clusters, ros::Time detection_time) {
        using namespace open3d::pipelines::registration;

        std::vector<open3d::geometry::PointCloud> valid_clusters;
        valid_clusters.reserve(clusters.size());
        for (size_t i = 0; i < clusters.size(); i++) {
            if (clusters[i].points_.size() < 100) continue;
            valid_clusters.push_back(std::move(clusters[i]));
        }

        std::vector<int> cluster_matches = gale_shapley<open3d::geometry::PointCloud, Object>(
            valid_clusters, objects_, [](const auto& cluster, const auto& object) {
                return (cluster.GetCenter() - object.pose.translation()).squaredNorm();
            });

        double fitness_threshold = 0.7;

        std::vector<bool> detected_objects(objects_.size(), false);

        for (size_t i = 0; i < cluster_matches.size(); i++) {
            auto& cluster = valid_clusters[i];

            // TODO: check distance in case we get a really wrong match?
            // there is a nearby object
            if (cluster_matches[i] != -1) {
                auto& object = objects_[cluster_matches[i]];

                auto icp_result = RegistrationICP(cluster, object_types_[object.type].point_cloud,
                    0.025, object.pose.inverse().matrix(), TransformationEstimationPointToPlane());

                if (icp_result.fitness_ > fitness_threshold) {
                    Eigen::Isometry3d transform(icp_result.transformation_);
                    Eigen::Isometry3d pose(transform.inverse());

                    object.active = true;
                    object.last_detected = detection_time;
                    object.pose = pose;
                    detected_objects[cluster_matches[i]] = true;
                } else {  // treat as new object
                    auto [type, pose, fitness] = detect_new_object(cluster);

                    if (fitness > fitness_threshold) {
                        objects_.push_back({next_id++, true, pose, detection_time, type});
                    }
                }
            } else {  // there is no nearby object
                auto [type, pose, fitness] = detect_new_object(cluster);

                if (fitness > fitness_threshold) {
                    objects_.push_back({next_id++, true, pose, detection_time, type});
                }
            }
        }

        // deactivate objects that were not re-detected
        // important to only go through non-newly-added objects here
        for (size_t i = 0; i < detected_objects.size(); i++) {
            if (detected_objects[i] == false) {
                objects_[i].active = false;
            }
        }

        // delete objects that haven't been detected in a long time
        std::vector<Object> recent_objects;
        recent_objects.reserve(objects_.size());
        for (const auto& track: objects_) {
            if (!track.active && track.last_detected - detection_time > ros::Duration(1)) {
                continue;
            }

            recent_objects.push_back(track);
        }

        objects_ = std::move(recent_objects);
    }

    std::tuple<ObjectType, Eigen::Isometry3d, double> detect_new_object(
        open3d::geometry::PointCloud& cluster) {
        using namespace open3d::pipelines::registration;

        cluster.EstimateNormals();
        auto cluster_features = *ComputeFPFHFeature(cluster);

        RANSACConvergenceCriteria criteria(1000, 1.0);
        RegistrationResult best_result;
        best_result.fitness_ = -std::numeric_limits<double>::infinity();
        ObjectType best_type;

        assert(!object_types_.empty());

        for (const auto& [type, data]: object_types_) {
            auto ransac_result = RegistrationRANSACBasedOnFeatureMatching(cluster, data.point_cloud,
                cluster_features, data.features, false, 0.025,
                TransformationEstimationPointToPoint(), 4, {}, criteria);

            auto icp_result = RegistrationICP(cluster, data.point_cloud, 0.025,
                ransac_result.transformation_, TransformationEstimationPointToPlane());

            if (icp_result.fitness_ > best_result.fitness_) {
                best_result = icp_result;
                best_type = type;
            }
        }

        Eigen::Isometry3d transform(best_result.transformation_);
        Eigen::Isometry3d pose(transform.inverse());

        return {best_type, pose, best_result.fitness_};
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
    ros::Subscriber sub1_;
    ros::Publisher filtered_pub_;
    ros::Publisher detection_pub_;

    size_t next_id = 0;
    std::vector<Object> objects_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string base_link_frame_ = "wx250s/base_link";

    rviz_visual_tools::RvizVisualToolsPtr visual_tools_;

    std::unordered_map<ObjectType, ObjectTypeInfo> object_types_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_registration_node");
    ObjectRegistration loader;
    ros::spin();
    return 0;
}
