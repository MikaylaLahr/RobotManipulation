#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "robot_manipulation/GenerateGraspCandidates.h"
#include "robot_manipulation/GraspCandidates.h"
#include "robot_manipulation/GraspCandidate.h"
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include "tf2_eigen/tf2_eigen.h"
#include "gpd/grasp_detector.h"

// https://github.com/atenpas/gpd_ros/blob/master/src/gpd_ros/grasp_messages.cpp
robot_manipulation::GraspCandidate create_grasp_message(const gpd::candidate::Hand& hand) {
    robot_manipulation::GraspCandidate msg;
    msg.position.x = hand.getPosition().x();
    msg.position.y = hand.getPosition().y();
    msg.position.z = hand.getPosition().z();
    tf2::toMsg(hand.getApproach(), msg.approach);
    tf2::toMsg(hand.getBinormal(), msg.binormal);
    tf2::toMsg(hand.getAxis(), msg.axis);
    msg.width = hand.getGraspWidth();
    msg.score = hand.getScore();
    msg.sample.x = hand.getSample().x();
    msg.sample.y = hand.getSample().y();
    msg.sample.z = hand.getSample().z();

    return msg;
}

robot_manipulation::GraspCandidates create_grasp_list_message(
    const std::vector<std::unique_ptr<gpd::candidate::Hand>>& hands,
    const std_msgs::Header& header) {
    robot_manipulation::GraspCandidates msg;

    for (int i = 0; i < hands.size(); i++) {
        msg.candidates.push_back(create_grasp_message(*hands[i]));
    }

    msg.header = header;

    return msg;
}

class GraspGenerator {
public:
    GraspGenerator(): nh("~"), detector(nh.param<std::string>("config_file", "")) {
        service = nh.advertiseService(
            "generate_grasp_candidates", &GraspGenerator::generate_candidates_callback, this);
    }

    bool generate_candidates_callback(robot_manipulation::GenerateGraspCandidates::Request& req,
        robot_manipulation::GenerateGraspCandidates::Response& res) {
        const sensor_msgs::PointCloud2& cloud = req.input_cloud;

        if (cloud.width * cloud.height == 0) {
            return false;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(cloud, *pcl_cloud);

        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(
            new pcl::PointCloud<pcl::PointNormal>);
        pcl::copyPointCloud(*pcl_cloud, *cloud_with_normals);  // normals will be calculated for us?

        Eigen::Matrix3Xd view_points(3, 1);
        view_points << 0, 0, 0;

        Eigen::MatrixXi camera_source = Eigen::MatrixXi::Ones(1, cloud.width * cloud.height);

        gpd::util::Cloud gpd_cloud(cloud_with_normals, camera_source, view_points);

        detector.preprocessPointCloud(gpd_cloud);

        std::vector<std::unique_ptr<gpd::candidate::Hand>> grasps = detector.detectGrasps(
            gpd_cloud);

        if (grasps.size() > 0) {
            // Publish the detected grasps.
            robot_manipulation::GraspCandidates selected_grasps_msg = create_grasp_list_message(
                grasps, cloud.header);

            res.candidates = selected_grasps_msg;
            ROS_INFO_STREAM(
                "Detected " << selected_grasps_msg.candidates.size() << " highest-scoring grasps.");
            return true;
        }

        return false;
    }

private:
    ros::NodeHandle nh;
    gpd::GraspDetector detector;
    ros::ServiceServer service;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "grasp_generator");

    GraspGenerator node;
    ROS_INFO("Point cloud processing service ready.");

    ros::spin();

    return 0;
}
