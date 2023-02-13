# Cube-Pose-Estimation-with-ArUco
Cube pose estimation using ArUco

This is a ROS package for detecting 3d pose and orientation of the center of a cube using Zed camera (every other camera can be easily used changing some lines). It is written in OpenCV using python.

We set firstly the rotation matrix for every face to rotate that specific frame into the frame with id=0.
Then we compute the orientation and traslation of a single face in an hierarchical way (aruco with lower ID first) and we apply an homogeneous transformation to obtain the rototranslated frame of the cube's center expressed in the camera frame.
Then we converted rotation in quaternions and publish the pose in a ROS messagge: geometry_msgs:PoseStamped.
We set the rotation matrix for every face to rotate that specific frame into the frame with id=0.

Put this folder inside your ROS1 workspace and then build with:
```
catkin_make
```
Run with:
```
rosrun cube-pose-estimation-aruco-ros arucoPoseEstimation.py
```


This is the output:
(In magenta we have the centroid and segment represent frame of the cube)

https://user-images.githubusercontent.com/93198865/218557430-dea50a92-8e3b-4b19-bab9-e27cc1c84c7a.mp4
