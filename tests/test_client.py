#!/usr/bin/env python3
"""
Grasp Client — 适配 moveit_server.py (GET / + POST /predict)

Usage:
    python test_client.py                                          # 默认参数
    python test_client.py --host 192.168.1.100 --port 14086        # 远程主机
    python test_client.py --target 3 --servo-dt 0.02               # 指定目标+插值
"""

import os
import sys
import json
import argparse
import requests

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''


def main():
    pa = argparse.ArgumentParser(description="Grasp Client — 适配 moveit_server.py")
    pa.add_argument("--host", default="localhost", help="服务主机 (default: localhost)")
    pa.add_argument("--port", type=int, default=14086, help="服务端口 (default: 14086)")
    pa.add_argument("--robot", default="xarm7", help="机械臂名称 (default: xarm7)")
    pa.add_argument("--target", type=int, default=None, help="目标物体索引, 不指定则自动选最优")
    pa.add_argument("--depth", default="test_data/grasp-wrist-dpt_opt.png", help="深度图路径")
    pa.add_argument("--seg", default="test_data/rgb_detection_wrist.json", help="分割结果路径")
    pa.add_argument("--affordance", default="test_data/affordance.json", help="Affordance数据路径")
    pa.add_argument("--start-pos", type=json.loads,
                    default=[0.27, 0, 0.307, -3.14159, 0, 0],
                    help="起始位姿 [x,y,z,r,p,y]")
    pa.add_argument("--target-pos", type=json.loads,
                    default=[0.4, -0.55, 0.1, -3.14159, 0, 0],
                    help="放置框中心 [x,y,z,r,p,y]")
    pa.add_argument("--end-pos", type=json.loads,
                    default=[0.27, 0, 0.307, -3.14159, 0, 0],
                    help="结束位姿 [x,y,z,r,p,y]")
    pa.add_argument("--camera", type=json.loads,
                    default={"intrinsics": {"fx": 909.665, "fy": 909.533, "cx": 636.739, "cy": 376.35},
                             "extrinsics": {"translation": [0.325, 0.028, 0.658],
                                            "quaternion": [-0.703, 0.71, -0.026, 0.021]}},
                    help="相机内外参 JSON")
    pa.add_argument("--servo-dt", type=float, default=None,
                    help="伺服插值步长(秒), 如 0.02=50Hz, 不指定则返回原始点")
    pa.add_argument("--output-dir", default="data/out", help="输出目录 (default: data/out)")
    a = pa.parse_args()

    base_url = f"http://{a.host}:{a.port}"

    print("=" * 50)
    print("Grasp Client (moveit_server.py)")
    print("=" * 50)
    print(f"  Server:    {base_url}")
    print(f"  Robot:     {a.robot}")
    print(f"  Target:    {a.target}")
    print(f"  Servo dt:  {a.servo_dt}")
    print(f"  Depth:     {a.depth}")
    print(f"  Seg:       {a.seg}")
    print(f"  Affordance:{a.affordance}")
    print()

    # 检查文件
    for path, label in [(a.depth, "depth"), (a.seg, "seg"), (a.affordance, "affordance")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    # 健康检查 GET /
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        health = resp.json()
        print(f"Health: {health}")
        srv_robot = health.get("robot", "")
        if srv_robot != "unknown" and srv_robot != a.robot:
            print(f"WARNING: Server robot='{srv_robot}', client robot='{a.robot}'")
        if health.get("status") != "healthy":
            print(f"WARNING: Server status={health.get('status')}")
    except Exception as e:
        print(f"ERROR: Cannot reach server — {e}")
        sys.exit(1)

    # 构建请求
    payload = {
        "robot_name": a.robot,
        "dpt": os.path.abspath(a.depth),
        "objs": os.path.abspath(a.affordance),
        "seg_json": os.path.abspath(a.seg),
        "camera": a.camera,
        "start_pos": a.start_pos,
        "target_pos": a.target_pos,
        "end_pos": a.end_pos,
    }
    if a.target is not None and a.target >= 0:
        payload["target_object_index"] = a.target
    if a.servo_dt:
        payload["servo_dt"] = a.servo_dt

    print(f"\nPOST {base_url}/predict ...")
    try:
        resp = requests.post(f"{base_url}/predict", json=payload, timeout=1800)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to {base_url}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out (30min)")
        sys.exit(1)

    result = resp.json()
    print(f"\nStatus: {resp.status_code}  success={result.get('success')}")

    if not result.get("success"):
        print(f"ERROR: {result.get('message', 'unknown')}")
        sys.exit(1)

    # 打印结果摘要
    print(f"Object:        index={result.get('obj_index')}  id={result.get('instance_id')}")
    print(f"Planning time: {result.get('planning_time', 0):.2f}s")
    print(f"Trajectory phases:")
    traj = result.get("trajectory", {})
    for phase, data in traj.items():
        n = len(data.get("positions", []))
        dt_info = f"  dt={data['dt']}s  duration={data['duration']:.2f}s" if "dt" in data else ""
        print(f"  {phase}: {n} waypoints{dt_info}")

    # 保存结果
    os.makedirs(a.output_dir, exist_ok=True)
    out_path = os.path.join(a.output_dir, "grasp_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
