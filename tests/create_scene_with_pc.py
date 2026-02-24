#!/usr/bin/env python3
"""
Create MoveIt planning scene from point cloud using PointCloud2 and octomap_server.
This script:
1. Publishes PointCloud2 to /scene_cloud
2. Subscribes to /octomap_binary from octomap_server
3. Republishes as PlanningScene message for MoveIt
"""

from pathlib import Path
import json
import cv2
import sys
import numpy as np
import os
import time
import struct
sys.path.append('data/3DGrasp-BMv1/')
sys.path.append('../data/3DGrasp-BMv1/')

from frame_cvt import *

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from moveit_msgs.msg import PlanningScene, PlanningSceneWorld
from octomap_msgs.msg import Octomap


def load_res(dataset_dir):
    config_path = os.path.join(dataset_dir, 'config.json')
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    init = cfg.get("robot", {}).get("init_pos_mm_deg", {}).get("topdown", None)
    end_pose_mm_deg = np.array(
        [init["x"], init["y"], init["z"], init["roll"], init["pitch"], init["yaw"]],
        dtype=np.float64,
    )

    cameras = cfg.get("cameras", {})

    filename = 'grasp-wrist'
    rgb_fp = os.path.join(dataset_dir, f'{filename}-rgb.jpg')
    rgb = cv2.imread(rgb_fp)
    dpt_fp = os.path.join(dataset_dir, f'{filename}-dpt_opt.png')
    depth_raw = cv2.imread(dpt_fp, cv2.IMREAD_UNCHANGED) / 1000.0

    wrist_cfg = cfg["cameras"]["wrist"]
    pc_cam = dpt_to_pc(dpt=depth_raw, cam=wrist_cfg, max_depth=2.0)

    flange_to_tool = np.asarray(cfg["robot"]["flange_to_tool_translation_m"], dtype=np.float64)

    end_pose_mm_deg[2] += 200
    wrist_base = wrist_cam_to_base(
        pc_cam,
        end_pose_mm_deg,
        cam_to_flan_translation_m=wrist_cfg["cam_to_flan_translation_m"],
        cam_to_flan_quat_xyzw=wrist_cfg["cam_to_flan_quat_xyzw"],
        flange_to_tool_translation_m=flange_to_tool,
    )


    return wrist_base, cfg


def create_pointcloud2(points, frame_id='world'):
    """
    Create a PointCloud2 message from numpy array.

    Args:
        points: Nx3 numpy array of XYZ points
        frame_id: Frame ID for the point cloud

    Returns:
        PointCloud2 message
    """
    # Flatten if needed
    if points.ndim == 3:
        points = points.reshape(-1, 3)

    # Filter out invalid points
    valid_mask = (points[:, 2] > 0.05) & (points[:, 2] < 1.5)  # Z between 5cm and 1.5m
    valid_mask &= np.all(np.isfinite(points), axis=1)
    points = points[valid_mask]

    print(f"Valid points: {len(points)}")

    if len(points) < 10:
        raise ValueError("Not enough valid points")

    # Create PointCloud2 message
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()

    # Define fields (x, y, z)
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    msg.is_bigendian = False
    msg.point_step = 12  # 3 floats * 4 bytes
    msg.row_step = msg.point_step * len(points)
    msg.is_dense = True
    msg.width = len(points)
    msg.height = 1

    # Pack point data
    buffer = []
    for point in points:
        buffer.append(struct.pack('fff', point[0], point[1], point[2]))

    msg.data = b''.join(buffer)

    return msg


class OctomapBridge(Node):
    """Bridge between octomap_server and MoveIt planning scene."""

    def __init__(self):
        super().__init__('octomap_bridge')

        # Publisher for planning scene
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        # Subscriber for octomap
        self.octomap_sub = self.create_subscription(
            Octomap,
            '/octomap_binary',
            self.octomap_callback,
            10
        )

        self.get_logger().info('Octomap bridge started')
        self.get_logger().info('Subscribing to /octomap_binary')
        self.get_logger().info('Publishing to /planning_scene')

    def octomap_callback(self, octomap_msg):
        """Receive octomap and publish as planning scene."""
        self.get_logger().info(f'Received octomap: {len(octomap_msg.data)} bytes')

        # Create planning scene message
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.octomap.header = octomap_msg.header
        scene.world.octomap.octomap = octomap_msg

        # Publish to planning scene
        self.scene_pub.publish(scene)
        self.get_logger().info('Published octomap to planning scene')


def main():
    # Load point cloud
    dataset_dir = 'data/3DGrasp-BMv1/'
    print(f"Loading point cloud from {dataset_dir}...")
    pc, cfg = load_res(dataset_dir)

    # Initialize ROS2
    rclpy.init()

    # Create nodes
    pc_node = Node('scene_pointcloud_publisher')
    bridge_node = OctomapBridge()

    # Create publisher for PointCloud2
    pc_pub = pc_node.create_publisher(PointCloud2, '/scene_cloud', 10)

    # Wait for connections
    print("Waiting for connections...")
    time.sleep(1.0)

    # Create and publish PointCloud2
    pc_msg = create_pointcloud2(pc, frame_id='world')

    print(f"Publishing PointCloud2 to /scene_cloud...")
    print(f"  Points: {pc_msg.width}")
    print(f"  Frame: {pc_msg.header.frame_id}")

    # Publish point cloud
    for i in range(5):
        pc_pub.publish(pc_msg)
        time.sleep(0.2)

    print("✓ Published PointCloud2")
    print("\nWaiting for octomap from octomap_server...")
    print("The bridge will automatically forward it to MoveIt's planning scene")

    # Spin the bridge node to receive and forward octomaps
    try:
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # Cleanup
    pc_node.destroy_node()
    bridge_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
