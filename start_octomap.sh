#!/bin/bash
# 启动OctoMap服务器

source /opt/ros/jazzy/setup.bash

# 启动octomap_server
ros2 run octomap_server octomap_server_node \
    --ros-args \
    --params-file $(dirname "$0")/config/octomap_config.yaml \
    -r cloud_in:=/camera/depth/color/points
