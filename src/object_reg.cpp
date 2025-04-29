#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/filter_indices.h>
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
#include <boost/filesystem/operations.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>
#include <boost/filesystem.hpp>
#include "geometry_msgs/Vector3.h"
#include "gs_matching.h"
#include "ros/duration.h"
#include "ros/time.h"
#include "std_msgs/ColorRGBA.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types_conversion.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

enum ObjectType { Cube, Milk, Wine, Eggs, ToiletPaper, Can, None };

struct ObjectTypeInfo {
    std::string mesh_path;
    pcl::PointCloud<pcl::PointXYZ> point_cloud;
};

struct Object {
    ObjectType type;
    ros::Time redetection_time;
    // has a pose if type != None
    Eigen::Isometry3d pose;
};

struct ClusterTrack {
    int id;
    bool active;
    ros::Time last_detected;
    pcl::PointCloud<pcl::PointXYZHSV> point_cloud;
    pcl::PointXYZHSV mean;
    Object object;

    Eigen::Vector3d cluster_center() const {
        return Eigen::Vector3d(mean.x, mean.y, mean.z);
    }
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

        pcl::PCLPointCloud2 pc2;
        pcl_conversions::toPCL(*transformed_opt, pc2);
        pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
        pcl::fromPCLPointCloud2(pc2, pcl_cloud);

        pcl::PointCloud<pcl::PointXYZHSV> hsv;
        pcl::PointCloudXYZRGBtoXYZHSV(pcl_cloud, hsv);

        auto cloud_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZHSV>>(hsv);
        auto clusters = cluster_cloud(cloud_ptr);
        update_tracks(std::move(clusters), cloud_msg->header.stamp);
        draw_tracks();

        visual_tools_->trigger();
    }

    void read_meshes() {
        std::string mesh_folder;
        nh_.getParam("mesh_folder", mesh_folder);

        std::unordered_map<std::string, ObjectType> name_type_map{
            // {"cube", ObjectType::Cube},
            // {"milk", ObjectType::Milk},
            {"wine", ObjectType::Wine},
            // {"eggs", ObjectType::Eggs},
            // {"toilet_paper", ObjectType::ToiletPaper},
            // {"can", ObjectType::Can}
        };

        for (const auto& entry:
            boost::make_iterator_range(boost::filesystem::directory_iterator(mesh_folder), {})) {
            
            pcl::PolygonMesh mesh;
            pcl::io::loadPolygonFileSTL(entry.path().string(), mesh);
            pcl::PointCloud<pcl::PointXYZ> cloud;
            pcl::fromPCLPointCloud2(mesh.cloud, cloud);

            ObjectTypeInfo data{entry.path().string(), cloud};

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

            pcl::PointXYZRGB rgb;
            pcl::PointXYZHSVtoXYZRGB(track.mean, rgb);

            std_msgs::ColorRGBA rgba;
            rgba.r = static_cast<float>(rgb.r) / 255;
            rgba.g = static_cast<float>(rgb.g) / 255;
            rgba.b = static_cast<float>(rgb.b) / 255;
            rgba.a = 1.0;

            geometry_msgs::Vector3 scale;
            scale.x = 0.01;
            scale.y = 0.01;
            scale.z = 0.01;

            Eigen::Vector3d center(track.mean.x, track.mean.y, track.mean.z);

            Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
            text_pose.translation() = center + Eigen::Vector3d(0, 0, 0.1);
            // avoid id = 0
            visual_tools_->publishSphere(center, rgba, scale, "sphere", track.id + 1);
            visual_tools_->publishText(text_pose, std::to_string(track.id),
                rviz_visual_tools::WHITE, rviz_visual_tools::MEDIUM, false);
            
            if (track.object.type == ObjectType::None) {
                continue;
            }

            ObjectTypeInfo info = object_types_[track.object.type];

            Eigen::Isometry3d pose = track.object.pose;
            visual_tools_->publishMesh(pose, std::string("file://") + info.mesh_path,
                rviz_visual_tools::colors::TRANSLUCENT_DARK, 1.0, "mesh", track.id + 1);
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
        std::vector<pcl::PointCloud<pcl::PointXYZHSV>> clusters, ros::Time detection_time) {

        std::vector<pcl::PointXYZHSV> hsv_means(clusters.size());
        for (size_t i = 0; i < clusters.size(); i++) {
            hsv_means[i] = compute_cluster_mean(clusters[i]);
        }

        std::vector<int> track_matches(tracks_.size(), -1);
        std::vector<bool> clusters_taken(clusters.size(), false);
        
        for (size_t j = 0; j < tracks_.size(); j++) {
            const auto &track = tracks_[j];
            Eigen::Vector3d track_center(track.mean.x, track.mean.y, track.mean.z);

            int best_cluster = -1;

            for (size_t i = 0; i < clusters.size(); i++) {
                if (clusters_taken[i]) {
                    continue;
                }

                Eigen::Vector3d cluster_center(hsv_means[i].x, hsv_means[i].y, hsv_means[i].z);

                if ((track_center - cluster_center).norm() > 0.05) {
                    continue;
                }

                double hue_dist = 0;
                if (std::abs(track.mean.h - hsv_means[i].h) > 180) {
                    hue_dist = 360 - std::abs(track.mean.h - hsv_means[i].h);
                } else {
                    hue_dist = std::abs(track.mean.h - hsv_means[i].h);
                }

                if (hue_dist > 30) {
                    continue;
                }

                if (best_cluster == -1) {
                    best_cluster = i;
                    clusters_taken[i] = true;
                    continue;
                }

                Eigen::Vector3d other_cluster_center(hsv_means[best_cluster].x, hsv_means[best_cluster].y, hsv_means[best_cluster].z);
                if ((track_center - cluster_center).norm() < (track_center - other_cluster_center).norm()) {
                    best_cluster = i;
                    clusters_taken[i] = true;
                }   
            }

            track_matches[j] = best_cluster;
        }

        for (size_t i = 0; i < track_matches.size(); i++) {
            auto &track = tracks_[i];

            if (track_matches[i] != -1) {
                const auto &cluster = clusters[track_matches[i]];
                track.active = true;
                track.last_detected = detection_time;
                track.mean = compute_cluster_mean(cluster);
                track.point_cloud = std::move(cluster);
            } else {
                track.active = false;
            }
        }

        for (size_t i = 0; i < clusters.size(); i++) {
            if (!clusters_taken[i]) {
                ClusterTrack track;
                track.id = next_id++;
                track.active = true;
                track.last_detected = detection_time;
                track.mean = compute_cluster_mean(clusters[i]);
                track.point_cloud = std::move(clusters[i]);
                track.object.type = ObjectType::None;
                track.object.redetection_time = ros::Time::ZERO;
                tracks_.push_back(track);
            }
        }

        // delete tracks that haven't been detected
        std::vector<ClusterTrack> recent_tracks;
        recent_tracks.reserve(tracks_.size());
        for (const auto& track: tracks_) {
            if (!track.active && detection_time - track.last_detected > ros::Duration(1)) {
                continue;
            }
            
            recent_tracks.push_back(track);
        }

        tracks_ = std::move(recent_tracks);
    }

    pcl::PointXYZHSV compute_cluster_mean(const pcl::PointCloud<pcl::PointXYZHSV> &cluster) {
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        Eigen::Vector2f hue_avg = Eigen::Vector2f::Zero();
        float sat_avg = 0;
        float val_avg = 0;
        size_t num_hue_avg = 0;

        for (const auto &point: cluster) {
            centroid += point.getVector3fMap();

            if (point.s > 0.2) {
                float hue_rad = point.h / 180 * M_PI;
                Eigen::Vector2f hue_vec(std::cos(hue_rad), std::sin(hue_rad));
                hue_avg += hue_vec;
                num_hue_avg++;
            }

            sat_avg += point.s;
            val_avg += point.v;
        }
        centroid /= cluster.size();
        hue_avg /= num_hue_avg;
        sat_avg /= cluster.size();
        val_avg /= cluster.size();

        pcl::PointXYZHSV mean;
        mean.x = centroid.x();
        mean.y = centroid.y();
        mean.z = centroid.z();

        if (num_hue_avg > 0.1 * cluster.size()) {
            mean.h = std::fmod(std::atan2(hue_avg.y(), hue_avg.x()) / M_PI * 180, 180);
        } else {
            mean.h = 0;
        }
        mean.s = sat_avg;
        mean.v = val_avg;

        return mean;
    }

    // void update_objects() {
    //     for (auto &track: tracks_) {
    //         if (!track.active) {
    //             continue;
    //         }

    //         if (ros::Time::now() > track.object.redetection_time) {
    //             // detect
    //         } else {
    //             if (track.object.type == ObjectType::None) {
    //                 continue;
    //             }

    //             // track
    //         }
    //     }
    // }

    std::tuple<ObjectType, Eigen::Isometry3d, double> detect_object_from_scratch(
        pcl::PointCloud<pcl::PointXYZRGB>& cluster) {
        
    }

    std::vector<pcl::PointCloud<pcl::PointXYZHSV>> cluster_cloud(pcl::PointCloud<pcl::PointXYZHSV>::ConstPtr cloud) {
        pcl::ConditionalEuclideanClustering<pcl::PointXYZHSV> reg;
        pcl::search::Search<pcl::PointXYZHSV>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZHSV>);
        pcl::IndicesPtr indices(new std::vector<int>);

        pcl::removeNaNFromPointCloud(*cloud, *indices);

        auto condition_func = [](const pcl::PointXYZHSV &first, const pcl::PointXYZHSV &second, float squared_dist) {
            double hue_dist = 0.0;
            if (std::abs(second.h - first.h) > 180) {
                hue_dist = 360 - std::abs(second.h - first.h);
            } else {
                hue_dist = std::abs(second.h - first.h);
            }
            assert(hue_dist <= 180.0);

            double sat_dist = std::abs(second.s - first.s);
            double val_dist = std::abs(second.v - first.v);

            if (first.s > 0.1 && second.s > 0.1 && first.v > 0.1 && second.v > 0.1) {
                return hue_dist < 5 && sat_dist < 0.05 && val_dist < 0.05;
            } else {
                return sat_dist < 0.1 && val_dist < 0.1;
            }
        };

        reg.setInputCloud(cloud);
        reg.setIndices(indices);
        reg.setSearchMethod(tree);
        reg.setConditionFunction(std::function(condition_func));
        reg.setMinClusterSize(200);
        reg.setClusterTolerance(0.02);
      
        std::vector<pcl::PointIndices> point_indices;
        reg.segment(point_indices);

        std::vector<pcl::PointCloud<pcl::PointXYZHSV>> clusters(point_indices.size());
        for (size_t i = 0; i < point_indices.size(); i++) {
            clusters[i].reserve(point_indices[i].indices.size());
            for (auto point_idx : point_indices[i].indices) {
                clusters[i].push_back(cloud->points[point_idx]);
            }
        }

        return clusters;
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
