import math

from interbotix_xs_modules.arm import InterbotixManipulatorXS


def main():
    #   Initialize the arm module along with the pointcloud and armtag modules
    bot = InterbotixManipulatorXS("wx250s", moving_time=1.5, accel_time=0.75)
    bot.arm.go_to_sleep_pose()
    # bot.arm.set_ee_cartesian_trajectory(x=0.1, roll=-math.pi/6, pitch=-0.1)
    # bot.arm.set_ee_cartesian_trajectory(x=0.1, pitch=0.0)

if __name__ == '__main__':
    main()