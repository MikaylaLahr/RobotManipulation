import rospy
import sys
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from dataclasses import dataclass
from robot_manipulation.srv import ActivePerception, ActivePerceptionRequest, ActivePerceptionResponse

@dataclass
class MotionGoal:
    pose: Pose
    take_snapshot: bool


GOALS = [
    MotionGoal(pose=Pose(position=Point(x=0.15, y=0.0, z=0.35), orientation=Quaternion(x=0, y=0.6427876, z=0, w=0.7660444)), take_snapshot=True),
    MotionGoal(pose=Pose(position=Point(x=0.0, y=0.0, z=0.35), orientation=Quaternion(x=0, y=0.428, z=0, w=0.904)), take_snapshot=True),
    MotionGoal(pose=Pose(position=Point(x=0.047, y=0.227, z=0.144), orientation=Quaternion(x=0.07129, y=0.1788946, z=-0.3632623, w=0.9115673)), take_snapshot=True),
    MotionGoal(pose=Pose(position=Point(x=0.047, y=-0.227, z=0.144), orientation=Quaternion(x=-0.07129, y=0.1788946, z=0.3632623, w=0.9115673)), take_snapshot=True),
]

SERVICE_NAME = "/active_perception/take_snapshot"


class MoveItInterface:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander(robot_description="wx250s/robot_description", ns="wx250s")
        self.scene = moveit_commander.PlanningSceneInterface(ns="wx250s")
        self.move_group = moveit_commander.MoveGroupCommander("interbotix_arm", robot_description="wx250s/robot_description", ns="wx250s", wait_for_servers=10.0)
        self.display_trajectory_publisher = rospy.Publisher("move_group/display_planned_path",
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)
    
    def display_trajectory(self, plan):
        trajectory = moveit_msgs.msg.DisplayTrajectory()
        trajectory.trajectory_start = self.robot.get_current_state()
        trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(trajectory)


def main():
    rospy.init_node("active_perception_runner")
    moveit = MoveItInterface()

    moveit.move_group.allow_replanning(True)
    moveit.move_group.set_goal_tolerance(0.01)
    moveit.move_group.set_num_planning_attempts(10)
    moveit.move_group.set_max_velocity_scaling_factor(0.3)

    box_name = "objects_area"
    moveit.scene.remove_world_object(box_name)

    box_pose = PoseStamped()
    box_pose.header.frame_id = "wx250s/base_link"
    
    # Position the center of the box
    box_pose.pose.position.x = 0.3
    box_pose.pose.position.y = 0.0
    box_pose.pose.position.z = 0.1
    box_pose.pose.orientation.w = 1.0 # No rotation

    box_dimensions = (0.3, 0.3, 0.2)

    moveit.scene.add_box(box_name, box_pose, size=box_dimensions)

    try:
        rospy.wait_for_service(SERVICE_NAME, timeout=5.0)
    except rospy.ROSException as e:
        rospy.logerr(f"Service '{SERVICE_NAME}' not available: {e}")
        return
    
    take_snapshot = rospy.ServiceProxy(SERVICE_NAME, ActivePerception, persistent=True)

    for i, goal in enumerate(GOALS):
        rospy.loginfo(f"Planning goal {i}.")
        moveit.move_group.set_pose_target(goal.pose)
        success, plan, plan_time, code = moveit.move_group.plan()
        if not success:
            rospy.logerr(f"Failed to plan to pose {i}.")
            break

        moveit.display_trajectory(plan)
        answer = input("Trajectory okay [y/n]? ")
        if answer.lower() == "y":
            moveit.move_group.execute(plan)
        else:
            break

        if goal.take_snapshot:
            try:
                take_snapshot(ActivePerceptionRequest())
                rospy.loginfo("Snapshot taken.")
            except rospy.ServiceException as e:
                rospy.logwarn(f"Failed to take snapshot! {e}")


if __name__ == '__main__':
    main()
