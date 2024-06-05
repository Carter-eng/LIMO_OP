import numpy as np
import matplotlib.pyplot as plt
import pygame
import socket
import time
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDrive
import rospy
from LIMO_ROS_SETUP import CAV
import LIMO_LQR

def main():
    #initialize CAV, PID values, and another parameters
    isMain1 = False
    CAV1 = CAV("limo770")
    CAV1.generate_map(isMain1)
    transmissionRate = 30
    dt = 1/transmissionRate # or 0.1
    rate = rospy.Rate(transmissionRate) # 1Hz
    v_ref_CAV1 = 0.5 # set between 0.5 and 0.6
    lqr = LIMO_LQR.LQR()
    speed = 0
    count = 120
    x1 = np.array([
        [.8],
        [.8],
        [1.57],
        [v_ref_CAV1]
    ])
    x2 = np.array([
        [1.6],
        [1.6],
        [0],
        [v_ref_CAV1]
    ])
    xs = [x1,x2]

    #depending on if the car starts at main path or merging path, initialize different starting paths, points, and PID value
    # access the array that stores the distance of each line, then change velocity if the length is quite large
    # v_ref_CAV1 = 0.5 # set between 0.5 and 0.6, or higher if line is longer

    for xd in xs:
        for count in range(120,1,-1):
        #if the cav is near a critical point (which are turning corners), set current line, starting point, and PID values to be that of the next line
        
            x=np.array([
                [CAV1.position_z],
                [CAV1.position_x],
                [CAV1.position_yaw],
                [speed]
            ])
            u = lqr.lqr(x,xd,count)

            drive_msg_CAV1 = CAV1.control(u[0,0],u[1,0])
            CAV1.pub.publish(drive_msg_CAV1)
            speed = u[1,0]
            time.sleep(dt)

            rospy.spin()

if __name__ == '__main__':
    main()
