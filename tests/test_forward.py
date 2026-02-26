#!/usr/bin/env python3
"""
测试 GraspClient.forward() 远程调用示例

用法:
    # 1. 传入文件路径
    python test_forward.py --host 192.168.1.100 --port 14086 \
        --depth test_data/grasp-wrist-dpt_opt.png \
        --seg test_data/rgb_detection_wrist.json \
        --affordance test_data/affordance.json

    # 2. 指定目标物体
    python test_forward.py --host 192.168.1.100 --port 14086 \
        --depth test_data/grasp-wrist-dpt_opt.png \
        --seg test_data/rgb_detection_wrist.json \
        --affordance test_data/affordance.json \
        --target 0
"""

import argparse
import json
import sys

from grasp_client import GraspClient


def test_with_file_paths(client, depth_path, seg_path, affordance_path, target):
    """方式1: 传入本地文件路径（forward 内部读取文件）"""
    print("=" * 60)
    print("测试1: 传入文件路径")
    print("=" * 60)
    result = client.forward(
        depth=depth_path,
        seg_json=seg_path,
        affordance_json=affordance_path,
        target_object_index=target,
    )
    print_result(result)
    return result


def test_with_bytes(client, depth_path, seg_path, affordance_path, target):
    """方式2: 传入 bytes / dict（模拟真正的远程调用场景）"""
    print("=" * 60)
    print("测试2: 传入 bytes + dict")
    print("=" * 60)

    with open(depth_path, "rb") as f:
        depth_bytes = f.read()
    with open(seg_path, "r") as f:
        seg_dict = json.load(f)
    with open(affordance_path, "r") as f:
        affordance_dict = json.load(f)

    result = client.forward(
        depth=depth_bytes,
        seg_json=seg_dict,
        affordance_json=affordance_dict,
        target_object_index=target,
    )
    print_result(result)
    return result


def print_result(result):
    success = result.get("success", False)
    print(f"  success: {success}")
    print(f"  message: {result.get('message', '')}")

    if not success:
        print("  规划失败!")
        return

    for i, r in enumerate(result.get("results", [])):
        print(f"\n  --- result[{i}] ---")
        print(f"  instance_id:    {r.get('instance_id')}")
        print(f"  success:        {r.get('success')}")
        print(f"  planning_time:  {r.get('planning_time', 0):.2f}s")
        print(f"  rank:           {r.get('rank')}")

        for key in ("approaching_trajectory", "carrying_trajectory", "returning_trajectory"):
            traj = r.get(key)
            if traj:
                print(f"  {key}: {traj.get('num_points')} points, {traj.get('duration', 0):.2f}s")

        steps = r.get("execution_steps", [])
        print(f"  execution_steps: {len(steps)} steps")
        for j, step in enumerate(steps):
            action = step["action"]
            label = step.get("label", "")
            if action == "execute_trajectory":
                traj = step.get("trajectory")
                n_pts = len(traj["points"]) if traj else 0
                print(f"    [{j}] {action} ({label}) — {n_pts} points")
            elif action == "set_gripper":
                print(f"    [{j}] {action} ({label}) — pos={step.get('position')}")
            else:
                print(f"    [{j}] {action} ({label})")


def main():
    parser = argparse.ArgumentParser(description="测试 GraspClient.forward() 远程调用")
    parser.add_argument("--host", default="localhost", help="服务端地址 (default: localhost)")
    parser.add_argument("--port", type=int, default=14086, help="服务端口 (default: 14086)")
    parser.add_argument("--robot", default="xarm7", help="机械臂名称 (default: xarm7)")
    parser.add_argument("--target", type=int, default=None, help="目标物体索引 (default: None)")
    parser.add_argument("--depth", default="test_data/grasp-wrist-dpt_opt.png", help="深度图路径")
    parser.add_argument("--seg", default="test_data/rgb_detection_wrist.json", help="分割结果路径")
    parser.add_argument("--affordance", default="test_data/affordance.json", help="affordance路径")
    parser.add_argument("--mode", choices=["file", "bytes", "both"], default="both",
                        help="测试模式: file=文件路径, bytes=bytes+dict, both=两种都测")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"服务地址: {base_url}")
    print(f"机械臂:   {args.robot}")
    print(f"目标物体: {args.target}")
    print(f"深度图:   {args.depth}")
    print(f"分割JSON: {args.seg}")
    print(f"Afford:   {args.affordance}")
    print()

    client = GraspClient(base_url=base_url, robot_name=args.robot)

    if args.mode in ("file", "both"):
        result = test_with_file_paths(client, args.depth, args.seg, args.affordance, args.target)
        if not result.get("success"):
            sys.exit(1)

    if args.mode in ("bytes", "both"):
        result = test_with_bytes(client, args.depth, args.seg, args.affordance, args.target)
        if not result.get("success"):
            sys.exit(1)

    print("\n所有测试完成!")


if __name__ == "__main__":
    main()
