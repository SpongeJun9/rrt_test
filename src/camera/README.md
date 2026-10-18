博客：
在ros2 -humble下大华工业相机和海康工业相机的标定
https://blog.csdn.net/qq_64079631/article/details/136423941


编译
colcon build



运行海康相机节点
. install/setup.bash
ros2 launch hik_camera hik_camera.launch.py



运行大华相机的节点
. install/setup.bash
ros2 run ros2_dahua ros2_dahua 




相机的曝光和增益等参数，根据需求自行更改















