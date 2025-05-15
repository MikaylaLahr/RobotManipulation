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

class ObjectRegistration {
public:
    ObjectRegistration(): nh_("~"), tf_listener_(tf_buffer_) {
        sub1_ = nh_.subscribe("/point_cloud", 1, &ObjectRegistration::point_cloud_callback, this);
        filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/filtered_point_cloud", 1);
        detection_pub_ = nh_.advertise<vision_msgs::Detection3DArray>("/detections", 1);

        visual_tools_.reset(
            new rviz_visual_tools::RvizVisualTools(base_link_frame_, "/rviz_debug", nh_));
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

        pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
        pcl::fromROSMsg(*cloud_msg, pcl_cloud);

        pcl::PointCloud<pcl::PointXYZHSV> hsv;
        pcl::PointCloudXYZRGBtoXYZHSV(pcl_cloud, hsv);

        auto cloud_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZHSV>>(hsv);
        auto clusters = cluster_cloud(cloud_ptr);
        auto detections = convert_objects_to_ros(clusters, transformed_opt->header);

        detection_pub_.publish(detections);

        visual_tools_->trigger();
    }

    vision_msgs::Detection3DArray convert_objects_to_ros(
        const std::vector<pcl::PointCloud<pcl::PointXYZHSV>::Ptr>& objects,
        std_msgs::Header header) {
        vision_msgs::Detection3DArray detection_array;
        detection_array.header = header;

        size_t i = 0;
        for (const auto& cluster: objects) {
            vision_msgs::Detection3D detection;
            detection.header = header;

            pcl::PointCloud<pcl::PointXYZRGB> rgb;

            for (const auto& point: cluster->points) {
                pcl::PointXYZRGB xyzrgb;
                pcl::PointXYZHSVtoXYZRGB(point, xyzrgb);
                rgb.push_back(xyzrgb);
            }

            pcl::CentroidPoint<pcl::PointXYZRGB> rgb_centroid;
            for (const auto& point: rgb.points) {
                rgb_centroid.add(point);
            }
            pcl::PointXYZRGB p;
            rgb_centroid.get(p);

            vision_msgs::BoundingBox3D bbox;
            bbox.center.position.x = p.x;
            bbox.center.position.y = p.y;
            bbox.center.position.z = p.z;
            detection.bbox = bbox;

            std_msgs::ColorRGBA color;
            color.r = p.r / 255.0;
            color.g = p.g / 255.0;
            color.b = p.b / 255.0;
            color.a = 1.0;

            vision_msgs::ObjectHypothesisWithPose hyp;

            // Cube colors - Blue
            hyp.id = 10000 * ObjectType::None + i;
            if (color.r < 0.15 && color.g < 0.3 && color.b > 0.4) {
                hyp.id = 10000 * ObjectType::Cube + i;
            }

            // Milk colors - Bright green
            if (color.r < 0.35 && color.g > 0.39 && color.b < 0.3) {
                hyp.id = 10000 * ObjectType::Milk + i;
            }

            // Toilet Paper colors - Yellow
            if (color.r > 0.6 && color.g > 0.55 && color.b < 0.45) {
                hyp.id = 10000 * ObjectType::ToiletPaper + i;
            }

            // Egg colors - Grey
            if (color.r < 0.45 && color.r > 0.3 && color.g > 0.35 && color.b > 0.38) {
                hyp.id = 10000 * ObjectType::Eggs + i;
            }

            // Can colors - whiteish
            if (color.r > 0.55 && color.g > 0.55 && color.b > 0.55) {
                hyp.id = 10000 * ObjectType::Can + i;
            }

            // Wine colors - blackish
            if (color.r < 0.25 && color.g < 0.25 && color.b < 0.25) {
                hyp.id = 10000 * ObjectType::Wine + i;
            }

            detection.results.push_back(hyp);

            detection_array.detections.push_back(detection);

            geometry_msgs::Vector3 scale;
            scale.x = 0.01;
            scale.y = 0.01;
            scale.z = 0.01;

            Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
            text_pose.translation() = p.getVector3fMap().cast<double>()
                                      + Eigen::Vector3d(0.0, 0.0, 0.1);

            std::stringstream ss;
            ss << "rgb: " << color.r << ", " << color.g << ", " << color.b;
            ss << " " << hyp.id;

            visual_tools_->publishSphere(p.getVector3fMap().cast<double>(), color, scale);
            visual_tools_->publishText(text_pose, ss.str(), rviz_visual_tools::colors::WHITE,
                rviz_visual_tools::scales::MEDIUM, false);

            i++;
        }

        return detection_array;
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
                return hue_dist < 5 && sat_dist < 0.07 && val_dist < 0.07;
            } else {
                return sat_dist < 0.07 && val_dist < 0.07;
            }
        };

        clustering.setInputCloud(cloud);
        clustering.setIndices(indices);
        clustering.setSearchMethod(tree);
        clustering.setConditionFunction(std::function(condition_func));
        clustering.setMinClusterSize(150);
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
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_registration_node");
    ObjectRegistration loader;
    ros::spin();
    return 0;
}
