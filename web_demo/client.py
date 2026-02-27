#!/usr/bin/env python3
"""
MoveIt Grasp Server client — call POST /predict and save trajectory.

This service performs robotic grasp planning:
  1. Takes a depth image, object affordances, and segmentation data as input
  2. Uses MoveIt to plan a full pick-and-place trajectory for a 7-DOF robot arm
  3. Returns 5-phase trajectory (approaching → grasp_approach → retreat → carrying → returning)
     with joint positions, velocities, and accelerations for each phase

Usage:
  python client.py --dpt depth.png --objs affordance.json --seg detection.json
  python client.py --server http://remote:14086 --robot xarm7 --dpt depth.png --objs aff.json --seg seg.json
"""

import argparse
import json
import os
import sys
import requests


def main():
    pa = argparse.ArgumentParser(description="MoveIt Grasp Server client")
    pa.add_argument("--server", default="http://localhost:14086",
                    help="Server URL (default: http://localhost:14086)")
    pa.add_argument("--robot", default="xarm7", help="Robot name (default: xarm7)")
    pa.add_argument("--dpt", required=True, help="Path to depth image (uint16 PNG)")
    pa.add_argument("--objs", required=True, help="Path to affordance JSON")
    pa.add_argument("--seg", required=True, help="Path to segmentation JSON")
    pa.add_argument("--start-pos", type=json.loads,
                    default=[0.27, 0, 0.307, -3.14159, 0, 0],
                    help='Start pose [x,y,z,r,p,y] (default: [0.27,0,0.307,-3.14159,0,0])')
    pa.add_argument("--target-pos", type=json.loads,
                    default=[0.4, -0.55, 0.1, -3.14159, 0, 0],
                    help='Basket center [x,y,z,r,p,y] (default: [0.4,-0.55,0.1,-3.14159,0,0])')
    pa.add_argument("--end-pos", type=json.loads,
                    default=[0.27, 0, 0.307, -3.14159, 0, 0],
                    help='End pose [x,y,z,r,p,y] (default: [0.27,0,0.307,-3.14159,0,0])')
    pa.add_argument("--camera", type=json.loads,
                    default={"intrinsics": {"fx": 909.665, "fy": 909.533, "cx": 636.739, "cy": 376.35},
                             "extrinsics": {"translation": [0.325, 0.028, 0.658],
                                            "quaternion": [-0.703, 0.71, -0.026, 0.021]}},
                    help="Camera intrinsics/extrinsics JSON")
    pa.add_argument("--target-object-index", type=int, default=-1,
                    help="Target object index, -1 for auto (default: -1)")
    pa.add_argument("--output-dir", default="./data/out",
                    help="Directory to save result (default: ./data/out)")
    a = pa.parse_args()

    for path, label in [(a.dpt, "depth"), (a.objs, "affordance"), (a.seg, "segmentation")]:
        if not os.path.exists(path):
            print(f"Error: {label} file not found: {path}")
            sys.exit(1)

    url = a.server.rstrip("/")

    # Health check
    try:
        r = requests.get(url + "/", timeout=5)
        info = r.json()
        print(f"Server: {info.get('service')}  status={info.get('status')}  robot={info.get('robot')}")
    except Exception as e:
        print(f"Error: cannot reach server at {url} — {e}")
        sys.exit(1)

    # Build request
    payload = {
        "robot_name": a.robot,
        "dpt_path": os.path.abspath(a.dpt),
        "objs_path": os.path.abspath(a.objs),
        "seg_json_path": os.path.abspath(a.seg),
        "camera": a.camera,
        "start_pos": a.start_pos,
        "target_pos": a.target_pos,
        "end_pos": a.end_pos,
    }
    if a.target_object_index >= 0:
        payload["target_object_index"] = a.target_object_index

    print(f"\nCalling POST {url}/predict ...")
    print(f"  robot={a.robot}  dpt={a.dpt}  objs={a.objs}  seg={a.seg}")

    try:
        r = requests.post(url + "/predict", json=payload, timeout=1800)
    except Exception as e:
        print(f"Error: request failed — {e}")
        sys.exit(1)

    result = r.json()
    print(f"\nStatus: {r.status_code}  success={result.get('success')}")

    if not result.get("success"):
        print(f"Error: {result.get('message', 'unknown')}")
        sys.exit(1)

    print(f"Object: index={result.get('obj_index')}  id={result.get('instance_id')}")
    print(f"Planning time: {result.get('planning_time', 0):.2f}s")

    traj = result.get("trajectory", {})
    for phase, data in traj.items():
        n = len(data.get("positions", []))
        print(f"  {phase}: {n} waypoints")

    # Save result
    os.makedirs(a.output_dir, exist_ok=True)
    out_path = os.path.join(a.output_dir, "grasp_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved result to {out_path}")


if __name__ == "__main__":
    main()
