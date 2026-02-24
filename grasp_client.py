#!/usr/bin/env python3
"""
Grasp Client — 接收规划结果 + 仿真执行

通过 HTTP 调用 moveit_service.py 获取完整轨迹规划结果，
然后通过 ROS2 Action Client 逐步执行，驱动仿真中的机械臂。

Usage:
    python grasp_client.py                                    # 默认参数
    python grasp_client.py --port 8000 --target 3             # 指定端口和目标物体
    python grasp_client.py --port 8000 --robot xarm7          # 指定机器人
"""

import os
import sys
import json
import time
import argparse
import requests

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    RobotTrajectory,
    RobotState,
)
from sensor_msgs.msg import JointState

# 禁用代理，避免干扰本地连接
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''


class GraspClient(Node):
    """ROS2 节点：通过 HTTP 获取规划结果，通过 Action Client 执行轨迹"""

    def __init__(self, base_url: str, robot_name: str = "xarm7"):
        super().__init__('grasp_client')
        self.base_url = base_url
        self.robot_name = robot_name
        self.timeout_sec = 10.0

        # 加载配置
        config_path = f"config/{robot_name}_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.home_joints = self.config.get("home", {}).get("joints", [0.0] * 7)
        gripper_cfg = self.config.get("gripper", {})
        self.gripper_joint_max = gripper_cfg.get("joint_max", 0.085)

        # Action Clients
        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/xarm_gripper_traj_controller/follow_joint_trajectory'
        )

        self.get_logger().info(f"Waiting for action servers...")

        if not self.move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup action server not available")
        else:
            self.get_logger().info("  MoveGroup OK")

        if not self.execute_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("ExecuteTrajectory action server not available")
        else:
            self.get_logger().info("  ExecuteTrajectory OK")

        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Gripper FollowJointTrajectory action server not available")
        else:
            self.get_logger().info("  Gripper OK")

        self.get_logger().info(f"GraspClient initialized (url={base_url}, robot={robot_name})")

    # ── JSON → ROS Message 转换 ──────────────────────────────

    def json_to_trajectory(self, data: dict) -> RobotTrajectory:
        """将 JSON 轨迹数据反序列化为 RobotTrajectory 消息"""
        traj = RobotTrajectory()
        jt = traj.joint_trajectory
        jt.joint_names = data["joint_names"]

        for pt_data in data["points"]:
            pt = JointTrajectoryPoint()
            pt.positions = list(pt_data["positions"])
            if pt_data.get("velocities"):
                pt.velocities = list(pt_data["velocities"])
            if pt_data.get("accelerations"):
                pt.accelerations = list(pt_data["accelerations"])
            tfs = pt_data["time_from_start"]
            pt.time_from_start = Duration(sec=tfs["sec"], nanosec=tfs["nanosec"])
            jt.points.append(pt)

        return traj

    # ── 执行动作 ─────────────────────────────────────────────

    def execute_trajectory(self, traj: RobotTrajectory, timeout_sec: float = 120.0) -> bool:
        """通过 ExecuteTrajectory action 执行轨迹"""
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f"  Sending trajectory ({len(traj.joint_trajectory.points)} points)...")

        future = self.execute_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout_sec)

        if not future.done() or future.result() is None:
            self.get_logger().error("  Goal send failed/timeout")
            return False

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("  Goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)

        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("  Execution timeout")
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
            return False

        error_code = result_future.result().result.error_code.val
        if error_code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("  Trajectory executed OK")
            return True
        else:
            self.get_logger().error(f"  Execution failed, error_code={error_code}")
            return False

    def set_gripper(self, rad: float) -> bool:
        """通过 FollowJointTrajectory action 控制夹爪"""
        rad = float(max(0.0, min(self.gripper_joint_max, rad)))

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["drive_joint"]

        pt = JointTrajectoryPoint()
        pt.positions = [rad]
        pt.time_from_start = Duration(sec=1, nanosec=0)
        goal.trajectory.points = [pt]

        self.get_logger().info(f"  Setting gripper to {rad:.4f} rad...")

        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done() or future.result() is None:
            self.get_logger().error("  Gripper goal send failed")
            return False

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("  Gripper goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)

        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("  Gripper execution timeout")
            return False

        error_code = result_future.result().result.error_code
        if error_code == 0:
            self.get_logger().info(f"  Gripper set to {rad:.4f} rad OK")
            return True
        else:
            self.get_logger().error(f"  Gripper failed, error_code={error_code}")
            return False

    def move_to_home(self) -> bool:
        """通过 MoveGroup action 移动到 HOME 位置（plan + execute）"""
        self.get_logger().info("Moving to HOME position...")

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.robot_name
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 10.0
        goal_msg.request.max_velocity_scaling_factor = 1.0
        goal_msg.request.max_acceleration_scaling_factor = 1.0
        goal_msg.planning_options.plan_only = False  # plan + execute

        # 设置关节目标约束
        constraints = Constraints()
        for name, value in zip(self.joint_names, self.home_joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal_msg.request.goal_constraints.append(constraints)

        future = self.move_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout_sec)

        if not future.done() or future.result() is None:
            self.get_logger().error("HOME goal send failed")
            return False

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("HOME goal rejected")
            return False

        self.get_logger().info("HOME goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)

        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("HOME execution timeout")
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
            return False

        error_code = result_future.result().result.error_code.val
        if error_code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("HOME position reached")
            return True
        else:
            self.get_logger().error(f"HOME failed, error_code={error_code}")
            return False

    # ── HTTP 请求 ────────────────────────────────────────────

    def request_grasp_plan(
        self,
        depth_path: str,
        seg_json_path: str,
        affordance_path: str,
        target_object_index: int = None
    ) -> dict:
        """向 moveit_service 发送 HTTP 抓取规划请求（plan_only + return_trajectories）"""
        data = {
            "depth_path": depth_path,
            "seg_json_path": seg_json_path,
            "affordance_path": affordance_path,
            "target_object_index": target_object_index,
            "plan_only": True,
            "return_trajectories": True
        }

        self.get_logger().info(f"Sending grasp plan request to {self.base_url}/grasp ...")
        self.get_logger().info(f"  target_object_index={target_object_index}")

        response = requests.post(
            f"{self.base_url}/grasp",
            json=data,
            timeout=1800  # 30分钟超时
        )

        result = response.json()
        return result

    def request_remove_object(self, instance_id: str) -> bool:
        """通知服务端从场景移除物体"""
        try:
            resp = requests.post(
                f"{self.base_url}/remove_object",
                json={"instance_id": instance_id}, timeout=10)
            return resp.json().get("success", False)
        except Exception as e:
            self.get_logger().warn(f"remove_object request failed: {e}")
            return False

    def request_attach_object(self, instance_id: str) -> bool:
        """通知服务端将物体附着到夹爪"""
        try:
            resp = requests.post(
                f"{self.base_url}/attach_object",
                json={"instance_id": instance_id}, timeout=10)
            return resp.json().get("success", False)
        except Exception as e:
            self.get_logger().warn(f"attach_object request failed: {e}")
            return False

    def request_detach_object(self) -> bool:
        """通知服务端从夹爪分离物体"""
        try:
            resp = requests.post(
                f"{self.base_url}/detach_object", json={}, timeout=10)
            return resp.json().get("success", False)
        except Exception as e:
            self.get_logger().warn(f"detach_object request failed: {e}")
            return False

    def request_cleanup(self):
        """通知服务端清理场景"""
        try:
            resp = requests.post(f"{self.base_url}/cleanup", json={}, timeout=30)
            data = resp.json()
            if data.get("success"):
                self.get_logger().info("Scene cleanup OK")
            else:
                self.get_logger().warn(f"Scene cleanup failed: {data.get('message')}")
        except Exception as e:
            self.get_logger().warn(f"Cleanup request failed: {e}")

    # ── 执行规划步骤 ─────────────────────────────────────────

    def execute_plan(self, steps: list) -> bool:
        """按顺序执行所有 execution_steps"""
        total = len(steps)
        self.get_logger().info(f"Executing {total} steps...")

        for i, step in enumerate(steps, 1):
            action = step["action"]
            label = step.get("label", "unknown")
            self.get_logger().info(f"[{i}/{total}] {action}: {label}")

            if action == "set_gripper":
                ok = self.set_gripper(step["position"])
                if not ok:
                    self.get_logger().error(f"Step {i} ({label}) failed")
                    return False
                time.sleep(0.3)

            elif action == "execute_trajectory":
                traj_data = step["trajectory"]
                if traj_data is None:
                    self.get_logger().warn(f"Step {i} ({label}) has no trajectory, skipping")
                    continue
                traj = self.json_to_trajectory(traj_data)
                ok = self.execute_trajectory(traj)
                if not ok:
                    self.get_logger().error(f"Step {i} ({label}) failed")
                    return False
                time.sleep(0.3)

            elif action == "remove_object":
                ok = self.request_remove_object(step["instance_id"])
                if not ok:
                    self.get_logger().warn(f"Step {i} ({label}) failed, continuing")
                time.sleep(0.5)

            elif action == "attach_object":
                ok = self.request_attach_object(step["instance_id"])
                if not ok:
                    self.get_logger().warn(f"Step {i} ({label}) failed, continuing")
                time.sleep(0.5)

            elif action == "detach_object":
                ok = self.request_detach_object()
                if not ok:
                    self.get_logger().warn(f"Step {i} ({label}) failed, continuing")
                time.sleep(0.5)

            else:
                self.get_logger().warn(f"Unknown action: {action}, skipping")

        self.get_logger().info("All steps executed successfully")
        return True

    # ── 主流程 ───────────────────────────────────────────────

    def run(
        self,
        depth_path: str,
        seg_json_path: str,
        affordance_path: str,
        target_object_index: int = None
    ) -> bool:
        """主流程：HOME → HTTP规划 → 执行"""

        # 1. 移动到 HOME
        self.get_logger().info("=" * 50)
        self.get_logger().info("Phase 1: Moving to HOME")
        self.get_logger().info("=" * 50)
        if not self.move_to_home():
            self.get_logger().error("Failed to move to HOME, aborting")
            return False

        # 2. HTTP 请求规划
        self.get_logger().info("=" * 50)
        self.get_logger().info("Phase 2: Requesting grasp plan via HTTP")
        self.get_logger().info("=" * 50)

        try:
            result = self.request_grasp_plan(
                depth_path=depth_path,
                seg_json_path=seg_json_path,
                affordance_path=affordance_path,
                target_object_index=target_object_index
            )
        except requests.exceptions.ConnectionError:
            self.get_logger().error(f"Cannot connect to {self.base_url}")
            return False
        except requests.exceptions.Timeout:
            self.get_logger().error("HTTP request timed out (30min)")
            return False

        if not result.get("success"):
            self.get_logger().error(f"Grasp planning failed: {result.get('message')}")
            return False

        # 3. 解析 execution_steps
        results_list = result.get("results", [])
        if not results_list:
            self.get_logger().error("No results returned")
            return False

        # 诊断日志：显示每个结果的概要
        self.get_logger().info(f"Received {len(results_list)} result(s):")
        for i, r in enumerate(results_list):
            has_steps = "execution_steps" in r and r["execution_steps"]
            n_steps = len(r["execution_steps"]) if has_steps else 0
            self.get_logger().info(
                f"  [{i}] instance={r.get('instance_id')}, success={r.get('success')}, "
                f"planning_time={r.get('planning_time', 0):.2f}s, "
                f"execution_steps={n_steps}, keys={list(r.keys())}"
            )

        # 优先选成功且有 execution_steps 的结果，其次选有 execution_steps 的
        target_result = None
        for r in results_list:
            if r.get("success") and r.get("execution_steps"):
                target_result = r
                break
        if target_result is None:
            for r in results_list:
                if r.get("execution_steps"):
                    self.get_logger().warn(
                        f"Using partial result (success=False) for instance '{r.get('instance_id')}'"
                    )
                    target_result = r
                    break

        if target_result is None:
            self.get_logger().error(
                "No result with execution_steps found. "
                "Server may not have return_trajectories support or all planning failed."
            )
            return False

        steps = target_result["execution_steps"]
        self.get_logger().info(
            f"Received plan for instance '{target_result.get('instance_id')}': "
            f"{len(steps)} steps, planning_time={target_result.get('planning_time', 0):.2f}s, "
            f"full_success={target_result.get('success')}"
        )
        for i, step in enumerate(steps):
            label = step.get("label", "?")
            action = step["action"]
            if action == "set_gripper":
                self.get_logger().info(f"  step {i}: {action} pos={step.get('position'):.4f} ({label})")
            elif action in ("remove_object", "attach_object", "detach_object"):
                self.get_logger().info(f"  step {i}: {action} ({label})")
            else:
                traj = step.get("trajectory")
                n_pts = len(traj["points"]) if traj else 0
                self.get_logger().info(f"  step {i}: {action} points={n_pts} ({label})")

        # 4. 执行
        self.get_logger().info("=" * 50)
        self.get_logger().info("Phase 3: Executing grasp sequence")
        self.get_logger().info("=" * 50)

        success = self.execute_plan(steps)

        # 5. 清理场景（无论成功与否都要清理）
        self.get_logger().info("=" * 50)
        self.get_logger().info("Phase 4: Cleaning up scene")
        self.get_logger().info("=" * 50)
        self.request_cleanup()

        if success:
            self.get_logger().info("=" * 50)
            self.get_logger().info("Grasp sequence completed successfully!")
            self.get_logger().info("=" * 50)
        else:
            self.get_logger().error("Grasp sequence failed during execution")

        return success


def main():
    parser = argparse.ArgumentParser(description="Grasp Client - 接收规划结果并仿真执行")
    parser.add_argument("--host", default="localhost", help="服务主机 (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="服务端口 (default: 8000)")
    parser.add_argument("--robot", default="xarm7", help="机器人名称 (default: xarm7)")
    parser.add_argument("--target", type=int, default=None, help="目标物体索引")
    parser.add_argument("--depth", default="test_data/grasp-wrist-dpt_opt.png", help="深度图路径")
    parser.add_argument("--seg", default="test_data/rgb_detection_wrist.json", help="分割结果路径")
    parser.add_argument("--affordance", default="test_data/affordance.json", help="Affordance数据路径")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print("=" * 50)
    print("Grasp Client")
    print("=" * 50)
    print(f"  Service URL: {base_url}")
    print(f"  Robot: {args.robot}")
    print(f"  Target object: {args.target}")
    print(f"  Depth: {args.depth}")
    print(f"  Seg: {args.seg}")
    print(f"  Affordance: {args.affordance}")
    print()

    # 健康检查
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        print(f"Service health: {resp.json()}")
    except Exception as e:
        print(f"Service not reachable: {e}")
        print("Make sure moveit_service.py is running")
        sys.exit(1)

    rclpy.init()

    try:
        client = GraspClient(base_url=base_url, robot_name=args.robot)
        success = client.run(
            depth_path=args.depth,
            seg_json_path=args.seg,
            affordance_path=args.affordance,
            target_object_index=args.target
        )
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            if 'client' in dir():
                client.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()
