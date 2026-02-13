#!/usr/bin/env python3
"""
Create MoveIt planning scene from point cloud using mesh collision objects.
"""

from pathlib import Path
import json
import cv2
import sys
import numpy as np
import os
import time
sys.path.append('data/3DGrasp-BMv1/')
sys.path.append('../data/3DGrasp-BMv1/')

from frame_cvt import *

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from geometry_msgs.msg import Pose, Point
from scipy.spatial import ConvexHull


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

    end_pose_mm_deg[2] += 100
    wrist_base = wrist_cam_to_base(
        pc_cam,
        end_pose_mm_deg,
        cam_to_flan_translation_m=wrist_cfg["cam_to_flan_translation_m"],
        cam_to_flan_quat_xyzw=wrist_cfg["cam_to_flan_quat_xyzw"],
        flange_to_tool_translation_m=flange_to_tool,
    )

    return wrist_base, cfg


def pc2boxes(pcd, voxel_size=0.05):
    """
    Convert point cloud to box collision objects.
    This is more stable for MoveIt than complex meshes.
    """
    # Flatten point cloud if needed
    if pcd.ndim == 3:
        pcd = pcd.reshape(-1, 3)

    # Filter out invalid points
    valid_mask = (pcd[:, 2] > 0.05) & (pcd[:, 2] < 1.5)  # Z between 5cm and 1.5m
    valid_mask &= np.all(np.isfinite(pcd), axis=1)
    pcd_valid = pcd[valid_mask]

    print(f"Valid points: {len(pcd_valid)} / {len(pcd)}")

    if len(pcd_valid) < 10:
        raise ValueError("Not enough valid points")

    # Voxelize to create boxes
    print(f"Voxelizing with size {voxel_size}m...")
    voxel_coords = np.floor(pcd_valid / voxel_size).astype(np.int32)
    unique_voxels = np.unique(voxel_coords, axis=0)

    # Convert voxel coordinates back to world coordinates (voxel centers)
    box_centers = (unique_voxels + 0.5) * voxel_size

    print(f"Created {len(box_centers)} boxes")

    return box_centers, voxel_size


def create_box_collision_object(box_centers, box_size, frame_id='world', object_id='scene_boxes'):
    """
    Create a CollisionObject message with box primitives.
    """
    collision_object = CollisionObject()
    collision_object.header.frame_id = frame_id
    collision_object.id = object_id
    collision_object.operation = CollisionObject.ADD

    # Add each box as a primitive
    for center in box_centers:
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [box_size, box_size, box_size]

        pose = Pose()
        pose.position.x = float(center[0])
        pose.position.y = float(center[1])
        pose.position.z = float(center[2])
        pose.orientation.w = 1.0

        collision_object.primitives.append(box)
        collision_object.primitive_poses.append(pose)

    return collision_object


def main():
    # Load point cloud
    dataset_dir = 'data/3DGrasp-BMv1/'
    print(f"Loading point cloud from {dataset_dir}...")
    pc, cfg = load_res(dataset_dir)

    # Convert to boxes (more stable than mesh for MoveIt)
    box_centers, box_size = pc2boxes(pc, voxel_size=0.05)

    # Initialize ROS2
    rclpy.init()
    node = Node('scene_box_publisher')

    # Create publisher
    pub = node.create_publisher(PlanningScene, '/planning_scene', 10)

    # Wait for subscribers
    print("Waiting for /planning_scene subscribers...")
    while pub.get_subscription_count() == 0:
        time.sleep(0.1)
    print(f"Found {pub.get_subscription_count()} subscriber(s)")

    # Create planning scene message
    planning_scene = PlanningScene()
    planning_scene.is_diff = True
    planning_scene.world.collision_objects.append(
        create_box_collision_object(box_centers, box_size, frame_id='world', object_id='scene_boxes')
    )

    # Publish
    print(f"Publishing box collision objects to /planning_scene...")
    pub.publish(planning_scene)
    time.sleep(0.5)

    print(f"✓ Published {len(box_centers)} boxes (voxel size: {box_size}m)")
    print("✓ Check RViz PlanningScene display to see the boxes")

    # Cleanup - only destroy node, don't shutdown rclpy context
    # This prevents interfering with other ROS nodes
    node.destroy_node()
    # Note: Not calling rclpy.shutdown() to avoid affecting other nodes


if __name__ == '__main__':
    main()
