#!/usr/bin/env python3
"""
Grasp Client — 纯 HTTP 客户端，接收规划结果 + 远程执行

通过 HTTP 调用 moveit_service.py 进行路径规划和轨迹执行，
无需 ROS2 环境，可在 Docker 外部运行。

Usage:
    python grasp_client.py                                    # 默认参数
    python grasp_client.py --port 8000 --target 3             # 指定端口和目标物体
    python grasp_client.py --host 192.168.1.100 --port 8000   # 指定远程主机
"""

import os
import sys
import json
import time
import argparse
import requests

# 禁用代理，避免干扰本地连接
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''


class GraspClient:
    """纯 HTTP 客户端：通过 HTTP 调用服务端进行规划和执行"""

    def __init__(self, base_url: str, robot_name: str = "xarm7"):
        self.base_url = base_url
        self.robot_name = robot_name

    def _log(self, msg: str):
        print(f"[GraspClient] {msg}")

    def _warn(self, msg: str):
        print(f"[GraspClient] WARN: {msg}")

    def _error(self, msg: str):
        print(f"[GraspClient] ERROR: {msg}")

    # ── 基础动作（通过 HTTP 调用服务端 ROS2）────────────────────

    def move_to_home(self) -> bool:
        """通过 HTTP 调用服务端移动到 HOME 位置"""
        self._log("Moving to HOME position...")
        try:
            resp = requests.post(f"{self.base_url}/move_home", json={"robot": self.robot_name}, timeout=120)
            data = resp.json()
            if data.get("success"):
                self._log("HOME position reached")
                return True
            else:
                self._error(f"HOME failed: {data.get('message')}")
                return False
        except requests.exceptions.Timeout:
            self._error("move_home request timed out")
            return False
        except Exception as e:
            self._error(f"move_home request failed: {e}")
            return False

    def execute_trajectory(self, traj_json: dict) -> bool:
        """通过 HTTP 调用服务端执行轨迹"""
        n_pts = len(traj_json.get("points", []))
        self._log(f"  Executing trajectory ({n_pts} points)...")
        try:
            resp = requests.post(
                f"{self.base_url}/execute_trajectory",
                json={"trajectory": traj_json},
                timeout=180
            )
            data = resp.json()
            if data.get("success"):
                self._log("  Trajectory executed OK")
                return True
            else:
                self._error(f"  Trajectory failed: {data.get('message')}")
                return False
        except requests.exceptions.Timeout:
            self._error("  execute_trajectory timed out")
            return False
        except Exception as e:
            self._error(f"  execute_trajectory request failed: {e}")
            return False

    def set_gripper(self, position: float) -> bool:
        """通过 HTTP 调用服务端控制夹爪"""
        self._log(f"  Setting gripper to {position:.4f} rad...")
        try:
            resp = requests.post(
                f"{self.base_url}/set_gripper",
                json={"position": position},
                timeout=30
            )
            data = resp.json()
            if data.get("success"):
                self._log(f"  Gripper set to {position:.4f} rad OK")
                return True
            else:
                self._error(f"  Gripper failed: {data.get('message')}")
                return False
        except Exception as e:
            self._error(f"  set_gripper request failed: {e}")
            return False

    # ── 场景操作（HTTP）──────────────────────────────────────────

    def request_grasp_plan(
        self,
        depth_path: str,
        seg_json_path: str,
        affordance_path: str,
        target_object_index: int = None
    ) -> dict:
        """向 moveit_service 发送 HTTP 抓取规划请求（plan_only + return_trajectories）"""
        data = {
            "robot": self.robot_name,
            "depth_path": depth_path,
            "seg_json_path": seg_json_path,
            "affordance_path": affordance_path,
            "target_object_index": target_object_index,
            "plan_only": True,
            "return_trajectories": True
        }

        self._log(f"Sending grasp plan request to {self.base_url}/grasp ...")
        self._log(f"  target_object_index={target_object_index}")

        response = requests.post(
            f"{self.base_url}/grasp",
            json=data,
            timeout=1800  # 30分钟超时
        )

        return response.json()

    def forward(self, depth, seg_json, affordance_json,
                robot_name=None, target_object_index=None):
        """远程调用 /forward 端点，直接传入数据（无需共享文件系统）

        Args:
            depth: 深度图 — str(文件路径) / np.ndarray / bytes
            seg_json: 分割结果 — str(文件路径) / dict
            affordance_json: affordance — str(文件路径) / dict
            robot_name: 机械臂名称，默认 self.robot_name
            target_object_index: 目标物体索引，None 表示无目标物
        """
        import numpy as np

        # --- depth -> bytes ---
        if isinstance(depth, str):
            with open(depth, "rb") as f:
                depth_bytes = f.read()
        elif isinstance(depth, np.ndarray):
            import cv2
            ok, buf = cv2.imencode(".png", depth)
            if not ok:
                raise ValueError("cv2.imencode failed for depth array")
            depth_bytes = buf.tobytes()
        elif isinstance(depth, (bytes, bytearray)):
            depth_bytes = bytes(depth)
        else:
            raise TypeError(f"Unsupported depth type: {type(depth)}")

        # --- seg_json -> str ---
        if isinstance(seg_json, (dict, list)):
            seg_json_str = json.dumps(seg_json)
        elif isinstance(seg_json, str) and os.path.isfile(seg_json):
            with open(seg_json, "r") as f:
                seg_json_str = f.read()
        elif isinstance(seg_json, str):
            seg_json_str = seg_json
        else:
            raise TypeError(f"Unsupported seg_json type: {type(seg_json)}")

        # --- affordance_json -> str ---
        if isinstance(affordance_json, (dict, list)):
            affordance_json_str = json.dumps(affordance_json)
        elif isinstance(affordance_json, str) and os.path.isfile(affordance_json):
            with open(affordance_json, "r") as f:
                affordance_json_str = f.read()
        elif isinstance(affordance_json, str):
            affordance_json_str = affordance_json
        else:
            raise TypeError(f"Unsupported affordance_json type: {type(affordance_json)}")

        robot_name = robot_name or self.robot_name
        toi = target_object_index if target_object_index is not None else -1

        files = {"depth": ("depth.png", depth_bytes, "image/png")}
        data = {
            "seg_json": seg_json_str,
            "affordance_json": affordance_json_str,
            "robot_name": robot_name,
            "target_object_index": toi,
        }

        self._log(f"Sending forward request to {self.base_url}/forward ...")
        response = requests.post(
            f"{self.base_url}/forward",
            files=files, data=data,
            timeout=1800,
        )
        return response.json()

    def request_remove_object(self, instance_id: str) -> bool:
        """通知服务端从场景移除物体"""
        try:
            resp = requests.post(
                f"{self.base_url}/remove_object",
                json={"instance_id": instance_id}, timeout=10)
            return resp.json().get("success", False)
        except Exception as e:
            self._warn(f"remove_object request failed: {e}")
            return False

    def request_attach_object(self, instance_id: str) -> bool:
        """通知服务端将物体附着到夹爪"""
        try:
            resp = requests.post(
                f"{self.base_url}/attach_object",
                json={"instance_id": instance_id}, timeout=10)
            return resp.json().get("success", False)
        except Exception as e:
            self._warn(f"attach_object request failed: {e}")
            return False

    def request_detach_object(self) -> bool:
        """通知服务端从夹爪分离物体"""
        try:
            resp = requests.post(
                f"{self.base_url}/detach_object", json={}, timeout=10)
            return resp.json().get("success", False)
        except Exception as e:
            self._warn(f"detach_object request failed: {e}")
            return False

    def request_cleanup(self):
        """通知服务端清理场景"""
        try:
            resp = requests.post(f"{self.base_url}/cleanup", json={}, timeout=30)
            data = resp.json()
            if data.get("success"):
                self._log("Scene cleanup OK")
            else:
                self._warn(f"Scene cleanup failed: {data.get('message')}")
        except Exception as e:
            self._warn(f"Cleanup request failed: {e}")

    # ── 执行规划步骤 ─────────────────────────────────────────────

    def execute_plan(self, steps: list) -> bool:
        """按顺序执行所有 execution_steps"""
        total = len(steps)
        self._log(f"Executing {total} steps...")

        for i, step in enumerate(steps, 1):
            action = step["action"]
            label = step.get("label", "unknown")
            self._log(f"[{i}/{total}] {action}: {label}")

            if action == "set_gripper":
                ok = self.set_gripper(step["position"])
                if not ok:
                    self._error(f"Step {i} ({label}) failed")
                    return False
                time.sleep(0.3)

            elif action == "execute_trajectory":
                traj_data = step["trajectory"]
                if traj_data is None:
                    self._warn(f"Step {i} ({label}) has no trajectory, skipping")
                    continue
                ok = self.execute_trajectory(traj_data)
                if not ok:
                    self._error(f"Step {i} ({label}) failed")
                    return False
                time.sleep(0.3)

            elif action == "remove_object":
                ok = self.request_remove_object(step["instance_id"])
                if not ok:
                    self._warn(f"Step {i} ({label}) failed, continuing")
                time.sleep(0.5)

            elif action == "attach_object":
                ok = self.request_attach_object(step["instance_id"])
                if not ok:
                    self._warn(f"Step {i} ({label}) failed, continuing")
                time.sleep(0.5)

            elif action == "detach_object":
                ok = self.request_detach_object()
                if not ok:
                    self._warn(f"Step {i} ({label}) failed, continuing")
                time.sleep(0.5)

            else:
                self._warn(f"Unknown action: {action}, skipping")

        self._log("All steps executed successfully")
        return True

    # ── 主流程 ───────────────────────────────────────────────────

    def run(
        self,
        depth_path: str,
        seg_json_path: str,
        affordance_path: str,
        target_object_index: int = None
    ) -> bool:
        """主流程：HOME → HTTP规划 → 执行"""

        # 1. 移动到 HOME
        self._log("=" * 50)
        self._log("Phase 1: Moving to HOME")
        self._log("=" * 50)
        if not self.move_to_home():
            self._error("Failed to move to HOME, aborting")
            return False

        # 2. HTTP 请求规划
        self._log("=" * 50)
        self._log("Phase 2: Requesting grasp plan via HTTP")
        self._log("=" * 50)

        try:
            result = self.request_grasp_plan(
                depth_path=depth_path,
                seg_json_path=seg_json_path,
                affordance_path=affordance_path,
                target_object_index=target_object_index
            )
        except requests.exceptions.ConnectionError:
            self._error(f"Cannot connect to {self.base_url}")
            return False
        except requests.exceptions.Timeout:
            self._error("HTTP request timed out (30min)")
            return False

        if not result.get("success"):
            self._error(f"Grasp planning failed: {result.get('message')}")
            return False

        # 3. 解析 execution_steps
        results_list = result.get("results", [])
        if not results_list:
            self._error("No results returned")
            return False

        # 诊断日志
        self._log(f"Received {len(results_list)} result(s):")
        for i, r in enumerate(results_list):
            has_steps = "execution_steps" in r and r["execution_steps"]
            n_steps = len(r["execution_steps"]) if has_steps else 0
            self._log(
                f"  [{i}] instance={r.get('instance_id')}, success={r.get('success')}, "
                f"planning_time={r.get('planning_time', 0):.2f}s, "
                f"execution_steps={n_steps}, keys={list(r.keys())}"
            )

        # 优先选成功且有 execution_steps 的结果
        target_result = None
        for r in results_list:
            if r.get("success") and r.get("execution_steps"):
                target_result = r
                break
        if target_result is None:
            for r in results_list:
                if r.get("execution_steps"):
                    self._warn(
                        f"Using partial result (success=False) for instance '{r.get('instance_id')}'"
                    )
                    target_result = r
                    break

        if target_result is None:
            self._error(
                "No result with execution_steps found. "
                "Server may not have return_trajectories support or all planning failed."
            )
            return False

        steps = target_result["execution_steps"]
        self._log(
            f"Received plan for instance '{target_result.get('instance_id')}': "
            f"{len(steps)} steps, planning_time={target_result.get('planning_time', 0):.2f}s, "
            f"full_success={target_result.get('success')}"
        )
        for i, step in enumerate(steps):
            label = step.get("label", "?")
            action = step["action"]
            if action == "set_gripper":
                self._log(f"  step {i}: {action} pos={step.get('position'):.4f} ({label})")
            elif action in ("remove_object", "attach_object", "detach_object"):
                self._log(f"  step {i}: {action} ({label})")
            else:
                traj = step.get("trajectory")
                n_pts = len(traj["points"]) if traj else 0
                self._log(f"  step {i}: {action} points={n_pts} ({label})")

        # 4. 执行
        self._log("=" * 50)
        self._log("Phase 3: Executing grasp sequence")
        self._log("=" * 50)

        success = self.execute_plan(steps)

        # 5. 清理场景
        self._log("=" * 50)
        self._log("Phase 4: Cleaning up scene")
        self._log("=" * 50)
        self.request_cleanup()

        if success:
            self._log("=" * 50)
            self._log("Grasp sequence completed successfully!")
            self._log("=" * 50)
        else:
            self._error("Grasp sequence failed during execution")

        return success


def main():
    parser = argparse.ArgumentParser(description="Grasp Client - 纯HTTP客户端，可在Docker外运行")
    parser.add_argument("--host", default="localhost", help="服务主机 (default: localhost)")
    parser.add_argument("--port", type=int, default=14086, help="服务端口 (default: 8000)")
    parser.add_argument("--robot", default="xarm7", help="机械臂名称，需与服务端一致 (default: xarm7)")
    parser.add_argument("--target", type=int, default=None, help="目标物体索引")
    parser.add_argument("--depth", default="test_data/grasp-wrist-dpt_opt.png", help="深度图路径")
    parser.add_argument("--seg", default="test_data/rgb_detection_wrist.json", help="分割结果路径")
    parser.add_argument("--affordance", default="test_data/affordance.json", help="Affordance数据路径")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print("=" * 50)
    print("Grasp Client (Pure HTTP)")
    print("=" * 50)
    print(f"  Service URL: {base_url}")
    print(f"  Robot: {args.robot}")
    print(f"  Target object: {args.target}")
    print(f"  Depth: {args.depth}")
    print(f"  Seg: {args.seg}")
    print(f"  Affordance: {args.affordance}")
    print()

    # 健康检查 + 机械臂匹配校验
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        health = resp.json()
        print(f"Service health: {health}")
        service_robot = health.get("robot", "")
        if service_robot != args.robot:
            print(f"ERROR: Robot mismatch! Client expects '{args.robot}' but service is running '{service_robot}'")
            print(f"  Hint: restart service with correct robot, or use --robot {service_robot}")
            sys.exit(1)
    except Exception as e:
        print(f"Service not reachable: {e}")
        print("Make sure moveit_service.py is running")
        sys.exit(1)

    client = GraspClient(base_url=base_url, robot_name=args.robot)
    success = client.run(
        depth_path=args.depth,
        seg_json_path=args.seg,
        affordance_path=args.affordance,
        target_object_index=args.target
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
