#include <geometry_msgs/Pose.h>
#include <geometry_msgs/TransformStamped.h>
#include <open3d/Open3D.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/centroid.h>
#include <pcl/conversions.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/correspondence.h>
#include <pcl/features/board.h>
#include <pcl/features/shot_omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <ros/ros.h>
#include <rviz_visual_tools/rviz_visual_tools.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <vision_msgs/BoundingBox3D.h>
#include <vision_msgs/Detection3D.h>
#include <vision_msgs/Detection3DArray.h>

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "gs_matching.h"
#include "open3d_conversions/open3d_conversions.h"
#include "ros/duration.h"
#include <opencv2/opencv.hpp>

enum ObjectType { Cube, Milk, Wine, Eggs, ToiletPaper, Can, None };

struct ObjectTypeInfo {
    std::string mesh_path;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr point_cloud;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr keypoints;
    pcl::PointCloud<pcl::Normal>::ConstPtr normals;
    pcl::PointCloud<pcl::SHOT352>::ConstPtr descriptors;
};

struct Object {
    size_t id;
    ObjectType type;
    // has a pose if type != None
    Eigen::Isometry3d pose;
    pcl::PointCloud<pcl::PointXYZHSV>::ConstPtr cloud;
    double fitness;
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

        ROS_INFO("Object registration node initialized.");
        ROS_INFO("Using frame %s as base link.", base_link_frame_.c_str());
        ROS_INFO("Found %zu object meshes.", object_types_.size());
    }

private:
    // Callback function for processing incoming PointCloud2 messages
    void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        ROS_INFO("Received PointCloud2 message (%dx%d points) in frame '%s'.", cloud_msg->width,
            cloud_msg->height, cloud_msg->header.frame_id.c_str());

        visual_tools_->deleteAllMarkers();

        auto transformed_opt = transform_point_cloud(*cloud_msg, base_link_frame_);
        if (!transformed_opt) {
            return;
        }

        pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
        pcl::fromROSMsg(*cloud_msg, pcl_cloud);

        pcl::PointCloud<pcl::PointXYZHSV> hsv;
        pcl::PointCloudXYZRGBtoXYZHSV(pcl_cloud, hsv);

        auto cloud_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZHSV>>(hsv);
        auto clusters = cluster_cloud(cloud_ptr);

        size_t non_none = 0;
        std::vector<Object> objects;
        for (size_t i = 0; i < clusters.size(); i++) {
            const auto& [type, pose, fitness] = detect_object_from_scratch(clusters[i]);
            ROS_INFO("Fitness: %f", fitness);

            // i don't care if ids are persistent
            objects.push_back({i, type, pose, clusters[i], fitness});
            if (type != ObjectType::None) {
                non_none++;
            }
        }
        ROS_INFO("%zu objects detected (%zu not ObjectType::None)", objects.size(), non_none);

        draw_objects(objects);

        vision_msgs::Detection3DArray detections = convert_objects_to_ros(
            objects, transformed_opt->header);

        detection_pub_.publish(detections);

        visual_tools_->trigger();
    }

    void read_meshes() {
        std::string mesh_folder;
        nh_.getParam("mesh_folder", mesh_folder);

        std::unordered_map<std::string, ObjectType> name_type_map{{"cube", ObjectType::Cube},
            {"milk", ObjectType::Milk}, {"wine", ObjectType::Wine}, {"eggs", ObjectType::Eggs},
            {"toilet_paper", ObjectType::ToiletPaper}, {"can", ObjectType::Can}};

        open3d::io::ReadTriangleMeshOptions options;
        options.enable_post_processing = false;
        options.print_progress = false;

        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
        norm_est.setKSearch(10);

        pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> descr_est;
        descr_est.setRadiusSearch(0.02);

        pcl::UniformSampling<pcl::PointXYZ> uniform_sampling;

        for (const auto& entry:
            boost::make_iterator_range(boost::filesystem::directory_iterator(mesh_folder), {})) {
            open3d::geometry::TriangleMesh mesh;
            open3d::io::ReadTriangleMesh(entry.path().string(), mesh, options);

            double surface_area = mesh.GetSurfaceArea();  // m^2
            double points_per_m_2 = 100000;               // based on voxel size?
            auto point_cloud = *mesh.SamplePointsUniformly(surface_area * points_per_m_2, false);

            pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& o3d_point: point_cloud.points_) {
                pcl::PointXYZ pcl_point;
                pcl_point.getVector3fMap() = o3d_point.cast<float>();
                pcl_cloud->push_back(pcl_point);
            }

            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
            norm_est.setInputCloud(pcl_cloud);
            norm_est.compute(*normals);

            pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>());
            uniform_sampling.setInputCloud(pcl_cloud);
            uniform_sampling.setRadiusSearch(0.03);
            uniform_sampling.filter(*keypoints);

            pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>());
            descr_est.setInputCloud(keypoints);
            descr_est.setInputNormals(normals);
            descr_est.setSearchSurface(pcl_cloud);
            descr_est.compute(*descriptors);

            ObjectTypeInfo data{entry.path().string(), pcl_cloud, keypoints, normals, descriptors};

            std::string filename = entry.path().filename().replace_extension("").string();
            if (name_type_map.find(filename) != name_type_map.end()) {
                ObjectType type = name_type_map[filename];
                object_types_[type] = data;
            }
        }
    }

    void draw_objects(const std::vector<Object>& objects) {
        pcl::MomentOfInertiaEstimation<pcl::PointXYZHSV> feature_extractor;

        for (size_t i = 0; i < objects.size(); i++) {
            const auto& object = objects[i];

            feature_extractor.setInputCloud(object.cloud);
            feature_extractor.compute();

            pcl::PointXYZHSV min_point_OBB;
            pcl::PointXYZHSV max_point_OBB;
            pcl::PointXYZHSV position_OBB;
            Eigen::Matrix3f rotational_matrix_OBB;
            feature_extractor.getOBB(
                min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

            Eigen::Isometry3d bb_pose;
            bb_pose.linear() = rotational_matrix_OBB.cast<double>();
            bb_pose.translation() = position_OBB.getVector3fMap().cast<double>();

            visual_tools_->publishWireframeCuboid(bb_pose,
                min_point_OBB.getVector3fMap().cast<double>(),
                max_point_OBB.getVector3fMap().cast<double>());
            visual_tools_->publishAxis(bb_pose, rviz_visual_tools::SMALL);

            if (object.type == ObjectType::None) {
                continue;
            }

            ObjectTypeInfo info = object_types_[object.type];

            Eigen::Isometry3d pose = object.pose;
            visual_tools_->publishMesh(pose, std::string("file://") + info.mesh_path,
                rviz_visual_tools::colors::TRANSLUCENT_DARK, 1.0, "mesh", i + 1);
        }
    }

    vision_msgs::Detection3DArray convert_objects_to_ros(
        const std::vector<Object>& objects, std_msgs::Header header) {
        vision_msgs::Detection3DArray detection_array;
        detection_array.header = header;

        for (const auto& object: objects) {
            if (object.type == ObjectType::None) {
                continue;
            }

            vision_msgs::Detection3D detection;
            detection.header = header;

            vision_msgs::BoundingBox3D bbox;
            bbox.center = tf2::toMsg(object.pose);
            detection.bbox = bbox;

            vision_msgs::ObjectHypothesisWithPose hypothesis;
            hypothesis.id = 10000 * object.type + object.id;
            hypothesis.pose.pose = tf2::toMsg(object.pose);
            hypothesis.score = object.fitness;
            detection.results.push_back(hypothesis);

            detection_array.detections.push_back(detection);
        }

        return detection_array;
    }

    std::tuple<ObjectType, Eigen::Isometry3d, double> detect_object_from_scratch(
        const pcl::PointCloud<pcl::PointXYZHSV>::ConstPtr cluster) {
        ROS_INFO("Detecting new objects");

        pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        xyz_cluster->reserve(cluster->size());
        for (const auto& point: cluster->points) {
            xyz_cluster->emplace_back(point.x, point.y, point.z);
        }

        sensor_msgs::PointCloud2::Ptr cluster_pc_ros(new sensor_msgs::PointCloud2);
        pcl::toROSMsg(*xyz_cluster, *cluster_pc_ros);
        open3d::geometry::PointCloud cluster_pc_o3d;
        open3d_conversions::rosToOpen3d(cluster_pc_ros, cluster_pc_o3d);

        using namespace open3d::pipelines::registration;

        cluster_pc_o3d.EstimateNormals();
        auto cluster_features = *ComputeFPFHFeature(cluster_pc_o3d);
        open3d::geometry::OrientedBoundingBox cluster_bb = z_axis_bounding_box(cluster_pc_o3d);

        RegistrationResult best_result;
        best_result.transformation_ = Eigen::Matrix4d::Identity();
        best_result.fitness_ = -std::numeric_limits<double>::infinity();
        ObjectType best_type = ObjectType::None;

        for (const auto& [type, data]: object_types_) {
            try {
                sensor_msgs::PointCloud2::Ptr object_pc_ros(new sensor_msgs::PointCloud2);
                pcl::toROSMsg(*data.point_cloud, *object_pc_ros);
                open3d::geometry::PointCloud object_pc_o3d;
                open3d_conversions::rosToOpen3d(object_pc_ros, object_pc_o3d);

                open3d::geometry::OrientedBoundingBox object_bb =
                    object_pc_o3d.GetMinimalOrientedBoundingBox();

                std::vector<double> cluster_extent{
                    cluster_bb.extent_.x(), cluster_bb.extent_.y(), cluster_bb.extent_.z()};
                std::sort(cluster_extent.begin(), cluster_extent.end());
                std::vector<double> object_extent{
                    object_bb.extent_.x(), object_bb.extent_.y(), object_bb.extent_.z()};
                std::sort(object_extent.begin(), object_extent.end());

                double ratio = cluster_extent[2] / object_extent[2];
                // if (ratio > 1.5 || ratio < 0.5) {
                //     ROS_INFO("Extent ratio test failed");
                //     continue;
                // }

                object_pc_o3d.EstimateNormals();
                auto object_features = *ComputeFPFHFeature(object_pc_o3d);

                auto result = FastGlobalRegistrationBasedOnFeatureMatching(cluster_pc_o3d,
                    object_pc_o3d, cluster_features, object_features,
                    FastGlobalRegistrationOption());

                auto icp_result = RegistrationICP(cluster_pc_o3d, object_pc_o3d, 0.01,
                    result.transformation_, TransformationEstimationPointToPlane());

                if (icp_result.IsBetterRANSACThan(best_result)) {
                    best_result = icp_result;
                    best_type = type;
                }
            } catch (...) {
                ROS_WARN("failed ransac");
            }
        }

        double fitness_threshold = 0.9;
        if (best_result.fitness_ < fitness_threshold) {
            best_type = ObjectType::None;
        }

        Eigen::Isometry3d transform(best_result.transformation_);
        Eigen::Isometry3d pose(transform.inverse());

        return {best_type, pose, best_result.fitness_};
    }

    open3d::geometry::OrientedBoundingBox z_axis_bounding_box(
        const open3d::geometry::PointCloud& pc) {
        std::vector<cv::Point2f> projected(pc.points_.size());

        for (size_t i = 0; i < pc.points_.size(); ++i) {
            Eigen::Vector3d point = pc.points_[i];
            projected[i] = cv::Point2f(point.x(), point.y());
        }

        cv::RotatedRect rect = cv::minAreaRect(projected);

        double z_extent = pc.GetMaxBound().z() - pc.GetMinBound().z();

        Eigen::Vector3d center(rect.center.x, rect.center.y, z_extent / 2 + pc.GetMinBound().z());
        Eigen::Matrix3d rotation(
            Eigen::AngleAxisd(rect.angle * M_PI / 180.0, Eigen::Vector3d::UnitZ()));
        Eigen::Vector3d extent(rect.size.width, rect.size.height, z_extent);

        return open3d::geometry::OrientedBoundingBox(center, rotation, extent);
    }

    std::vector<pcl::PointCloud<pcl::PointXYZHSV>::Ptr> cluster_cloud(
        pcl::PointCloud<pcl::PointXYZHSV>::ConstPtr cloud) {
        pcl::ConditionalEuclideanClustering<pcl::PointXYZHSV> clustering;
        pcl::search::Search<pcl::PointXYZHSV>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZHSV>);
        pcl::IndicesPtr indices(new std::vector<int>);

        pcl::removeNaNFromPointCloud(*cloud, *indices);

        auto condition_func = [](const pcl::PointXYZHSV& first, const pcl::PointXYZHSV& second,
                                  float squared_dist) {
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
                return sat_dist < 0.03 && val_dist < 0.03;
            }
        };

        clustering.setInputCloud(cloud);
        clustering.setIndices(indices);
        clustering.setSearchMethod(tree);
        clustering.setConditionFunction(std::function(condition_func));
        clustering.setMinClusterSize(200);
        clustering.setClusterTolerance(0.01);

        std::vector<pcl::PointIndices> point_indices;
        clustering.segment(point_indices);

        std::vector<pcl::PointCloud<pcl::PointXYZHSV>::Ptr> clusters(point_indices.size());
        for (size_t i = 0; i < point_indices.size(); i++) {
            clusters[i] = boost::make_shared<pcl::PointCloud<pcl::PointXYZHSV>>();
            clusters[i]->reserve(point_indices[i].indices.size());
            for (auto point_idx: point_indices[i].indices) {
                clusters[i]->push_back(cloud->points[point_idx]);
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
