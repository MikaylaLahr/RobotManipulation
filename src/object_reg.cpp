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
#include <cassert>
#include <exception>
#include <limits>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>
#include <boost/filesystem.hpp>
#include <open3d_conversions/open3d_conversions.h>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include "Eigen/src/Geometry/Transform.h"
#include "geometry_msgs/Vector3.h"
#include "gs_matching.h"
#include "opencv2/core/mat.hpp"
#include "ros/duration.h"
#include "ros/time.h"
#include "std_msgs/ColorRGBA.h"

enum ObjectType { Cube, Milk, Wine, Eggs, ToiletPaper, Can, None };

struct ObjectTypeInfo {
    std::string mesh_path;
    open3d::geometry::PointCloud point_cloud;
    open3d::pipelines::registration::Feature features;
};

struct Object {
    ObjectType type;
    ros::Time redetection_time;
    // has a pose if type != None
    Eigen::Isometry3d pose;
};

struct ClusterFeature {
    Eigen::Vector3d center;
    cv::Vec3b color_hsv;  // 0-180, 0-255, 0-255
    // features: convex hull volume? other volume?

    double distance(const ClusterFeature &other) {
        double hue_dist = 0.0;
        if (std::abs(color_hsv[0] - other.color_hsv[0]) > 90) {
            hue_dist = 180 - std::abs(color_hsv[0] - other.color_hsv[0]);
        } else {
            hue_dist = std::abs(color_hsv[0] - other.color_hsv[0]);
        }

        return (center - other.center).norm() + 0.005 * hue_dist;
    }
};

struct ClusterTrack {
    int id;
    bool active;
    ros::Time last_detected;
    ClusterFeature feature;
    open3d::geometry::PointCloud point_cloud;
    Object object;
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

private:
    // Callback function for processing incoming PointCloud2 messages
    void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        ROS_DEBUG("Received PointCloud2 message (%dx%d points) in frame '%s'.", cloud_msg->width,
            cloud_msg->height, cloud_msg->header.frame_id.c_str());

        visual_tools_->deleteAllMarkers();

        auto transformed_opt = transform_point_cloud(*cloud_msg, base_link_frame_);
        if (!transformed_opt) {
            return;
        }

        sensor_msgs::PointCloud2::ConstPtr transformed = boost::make_shared<sensor_msgs::PointCloud2>(std::move(*transformed_opt));
        open3d::geometry::PointCloud o3d_cloud;
        open3d_conversions::rosToOpen3d(transformed, o3d_cloud);

        auto cluster_clouds = cluster_cloud(o3d_cloud, 0.02, 50);
        update_tracks(std::move(cluster_clouds), cloud_msg->header.stamp);
        update_objects();
        draw_tracks();

        // TODO: use clusters to publish filtered cloud
        visual_tools_->trigger();
    }

    void read_meshes() {
        std::string mesh_folder;
        nh_.getParam("mesh_folder", mesh_folder);

        std::unordered_map<std::string, ObjectType> name_type_map{
            {"cube", ObjectType::Cube},
            {"milk", ObjectType::Milk},
            {"wine", ObjectType::Wine},
            {"eggs", ObjectType::Eggs},
            {"toilet_paper", ObjectType::ToiletPaper},
            {"can", ObjectType::Can}
        };

        open3d::io::ReadTriangleMeshOptions options;
        options.enable_post_processing = false;
        options.print_progress = false;

        for (const auto& entry:
            boost::make_iterator_range(boost::filesystem::directory_iterator(mesh_folder), {})) {
            open3d::geometry::TriangleMesh mesh;
            open3d::io::ReadTriangleMesh(entry.path().string(), mesh, options);

            auto point_cloud = *mesh.SamplePointsUniformly(100, true);
            auto features = *open3d::pipelines::registration::ComputeFPFHFeature(point_cloud);

            ObjectTypeInfo data{entry.path().string(), point_cloud, features};

            std::string filename = entry.path().filename().replace_extension("").string();
            if (name_type_map.find(filename) != name_type_map.end()) {
                ObjectType type = name_type_map[filename];
                object_types_[type] = data;
            }
        }
    }

    void draw_tracks() {
        for (const auto& track: tracks_) {
            if (!track.active) {
                continue;
            }

            cv::Vec3b rgb = hsv_to_rgb(track.feature.color_hsv);
            std_msgs::ColorRGBA rgba;
            rgba.r = static_cast<float>(rgb[0]) / 255;
            rgba.g = static_cast<float>(rgb[1]) / 255;
            rgba.b = static_cast<float>(rgb[2]) / 255;
            rgba.a = 1.0;

            geometry_msgs::Vector3 scale;
            scale.x = 0.01;
            scale.y = 0.01;
            scale.z = 0.01;

            Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
            text_pose.translation() = track.feature.center + Eigen::Vector3d(0, 0, 0.1);
            visual_tools_->publishSphere(track.feature.center, rgba, scale, "sphere", track.id);
            visual_tools_->publishText(text_pose, std::to_string(track.id),
                rviz_visual_tools::WHITE, rviz_visual_tools::MEDIUM, false);
            
            if (track.object.type == ObjectType::None) {
                continue;
            }

            ObjectTypeInfo info = object_types_[track.object.type];

            Eigen::Isometry3d pose = track.object.pose;
            visual_tools_->publishMesh(pose, std::string("file://") + info.mesh_path,
                rviz_visual_tools::colors::TRANSLUCENT_DARK, 1.0, "mesh", track.id);
        }
    }

    // vision_msgs::Detection3DArray convert_objects_to_ros(std_msgs::Header header) {
    //     vision_msgs::Detection3DArray detection_array;
    //     detection_array.header = header;

    //     for (const auto& track: tracks_) {
    //         if (!track.active || !track.object) {
    //             continue;
    //         }

    //         vision_msgs::Detection3D detection;
    //         detection.header = header;

    //         vision_msgs::BoundingBox3D bbox;
    //         bbox.center = tf2::toMsg(track.object->pose);
    //         detection.bbox = bbox;

    //         vision_msgs::ObjectHypothesisWithPose hypothesis;
    //         hypothesis.id = track.id;
    //         hypothesis.pose.pose = tf2::toMsg(track.object->pose);
    //         hypothesis.score = 1.0;
    //         detection.results.push_back(hypothesis);

    //         detection_array.detections.push_back(detection);
    //     }

    //     return detection_array;
    // }

    template<typename Distribution>
    ros::Time generate_future_time(Distribution dist) {
        double future_seconds = dist(gen);
        if (future_seconds < 0) {
            future_seconds = 0;
        }
        return ros::Time::now() + ros::Duration(future_seconds);
    }

    void update_tracks(
        std::vector<open3d::geometry::PointCloud> clusters, ros::Time detection_time) {
        using namespace open3d::pipelines::registration;

        // TODO: this is slow because compute_cluster_feature is called over and over
        std::vector<int> cluster_matches = gale_shapley<open3d::geometry::PointCloud, ClusterTrack>(
            clusters, tracks_, [&](const auto &cluster, const auto &track) {
                return compute_cluster_feature(cluster).distance(track.feature);
            });

        std::vector<bool> active_tracks(tracks_.size(), false);
        
        for (size_t i = 0; i < cluster_matches.size(); i++) {
            if (cluster_matches[i] != -1) {  // there is a matching track
                auto& track = tracks_[cluster_matches[i]];
                track.active = true;
                track.last_detected = detection_time;
                track.feature = compute_cluster_feature(clusters[i]);
                track.point_cloud = std::move(clusters[i]);
                active_tracks[cluster_matches[i]] = true;
            } else {  // there is no matching track, treat as new cluster
                ClusterTrack new_track;
                new_track.id = next_id++;
                new_track.active = true;
                new_track.last_detected = detection_time;
                new_track.feature = compute_cluster_feature(clusters[i]);
                new_track.point_cloud = std::move(clusters[i]);
                new_track.object.type = ObjectType::None;
                new_track.object.redetection_time = generate_future_time(std::normal_distribution<double>(1.0, 0.3));
                // no need to set pose, not allowed to read if type is none

                tracks_.push_back(new_track);
            }
        }

        // deactivate tracks that weren't matched
        for (size_t i = 0; i < active_tracks.size(); i++) {
            if (!active_tracks[i]) {
                tracks_[i].active = false;
            }
        }

        // delete tracks that haven't been detected in a long time
        std::vector<ClusterTrack> recent_tracks;
        recent_tracks.reserve(tracks_.size());
        for (const auto& track: tracks_) {
            if (!track.active && detection_time - track.last_detected > ros::Duration(30)) {
                continue;
            }
            
            recent_tracks.push_back(track);
        }

        tracks_ = std::move(recent_tracks);
    }

    void update_objects() {
        using namespace open3d::pipelines::registration;

        double fitness_threshold = 0.0;

        for (auto &track: tracks_) {
            if (!track.active) {
                continue;
            }

            if (ros::Time::now() > track.object.redetection_time) {
                ROS_INFO("Detecting from scratch track %d", track.id);
                const auto &[type, pose, fitness] = detect_object_from_scratch(track.point_cloud);
                if (fitness > fitness_threshold) {
                    track.object.type = type;
                    track.object.pose = pose;
                    track.object.redetection_time = generate_future_time(std::normal_distribution<double>(5.0, 2.0));
                } else {
                    track.object.type = ObjectType::None;
                    track.object.redetection_time = generate_future_time(std::normal_distribution<double>(10.0, 5.0));
                }
            } else {
                if (track.object.type == ObjectType::None) {
                    continue;
                }

                auto icp_result = RegistrationICP(track.point_cloud, object_types_[track.object.type].point_cloud,
                    0.05, track.object.pose.inverse().matrix(), TransformationEstimationPointToPlane());

                if (icp_result.fitness_ > fitness_threshold) {
                    ROS_INFO("Track id %d passed fitness test", track.id);
                    Eigen::Isometry3d transform(icp_result.transformation_);
                    Eigen::Isometry3d pose(transform.inverse());
                    track.object.pose = pose;
                } else {
                    track.object.type = ObjectType::None;
                    track.object.redetection_time = generate_future_time(std::normal_distribution<double>(1.0, 0.3));
                }
            }
        }
    }

    ClusterFeature compute_cluster_feature(const open3d::geometry::PointCloud &pc) {
        assert(pc.HasColors());

        Eigen::Vector3d avg_color = Eigen::Vector3d::Zero();
        for (const auto& color : pc.colors_) {
            avg_color += color;
        }
        avg_color /= pc.colors_.size();
        
        cv::Vec3b rgb(avg_color[0] * 255, avg_color[1] * 255, avg_color[2] * 255);

        return {
            pc.GetCenter(),
            rgb_to_hsv(rgb),
        };
    }

    cv::Vec3b hsv_to_rgb(const cv::Vec3b &hsv) {
        assert(hsv[0] < 180);
        cv::Mat3b hsv_mat(1, 1);
        hsv_mat(0, 0) = hsv;
        cv::Mat3b rgb_mat;
        cv::cvtColor(hsv_mat, rgb_mat, cv::COLOR_HSV2RGB);
        return rgb_mat(0, 0);
    }

    cv::Vec3b rgb_to_hsv(const cv::Vec3b &rgb) {
        cv::Mat3b rgb_mat(1, 1);
        rgb_mat(0, 0) = rgb;
        cv::Mat3b hsv_mat;
        cv::cvtColor(rgb_mat, hsv_mat, cv::COLOR_RGB2HSV);
        return hsv_mat(0, 0);
    }

    std::tuple<ObjectType, Eigen::Isometry3d, double> detect_object_from_scratch(
        open3d::geometry::PointCloud& cluster) {
        using namespace open3d::pipelines::registration;

        cluster.EstimateNormals();
        auto cluster_features = *ComputeFPFHFeature(cluster);

        RANSACConvergenceCriteria criteria(10000, 0.9999);
        RegistrationResult best_result;
        best_result.fitness_ = -std::numeric_limits<double>::infinity();
        ObjectType best_type;

        assert(!object_types_.empty());

        for (const auto& [type, data]: object_types_) {
            try {
                // auto result = RegistrationRANSACBasedOnFeatureMatching(cluster, data.point_cloud, cluster_features, data.features,
                // false, 0.05, TransformationEstimationPointToPoint(), 4, {}, criteria
                // );
                auto result = FastGlobalRegistrationBasedOnFeatureMatching(cluster, data.point_cloud, cluster_features, data.features);

                auto icp_result = RegistrationICP(cluster, data.point_cloud, 0.01,
                result.transformation_, TransformationEstimationPointToPlane());

                open3d::geometry::PointCloud copy = cluster;
                copy.Transform(icp_result.transformation_);
                double min_z = copy.GetAxisAlignedBoundingBox().GetMinBound().z();
                if (min_z < -0.1) {
                    continue;
                }

                if (icp_result.IsBetterRANSACThan(best_result)) {
                    best_result = icp_result;
                    best_type = type;
                }
            } catch (...) {
                ROS_WARN("Failed to run object registration.");
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
            
            if (cluster_cloud.points_.size() < 50)
                continue;

            auto bbox = cluster_cloud.GetMinimalOrientedBoundingBox();
            if (bbox.extent_.x() <= 0.02 || bbox.extent_.y() <= 0.02 || bbox.extent_.z() <= 0.02)
                continue;

            cluster_clouds.push_back(cluster_cloud);
        }

        return cluster_clouds;
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

    int next_id = 0;
    std::vector<ClusterTrack> tracks_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string base_link_frame_ = "wx250s/base_link";

    rviz_visual_tools::RvizVisualToolsPtr visual_tools_;

    std::unordered_map<ObjectType, ObjectTypeInfo> object_types_;

    std::random_device rd{};
    std::mt19937 gen{rd()};
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_registration_node");
    ObjectRegistration loader;
    ros::spin();
    return 0;
}
