#!/usr/bin/env python3
"""
从末端执行器工作空间推导关节空间限制

使用方法:
    python3 workspace_to_joint_limits.py
    python3 workspace_to_joint_limits.py --config config/xarm7_config.json
    python3 workspace_to_joint_limits.py --cone-angle 45 --num-samples 3000

功能:
1. 从 config JSON 读取 workspace_limits 和 home 姿态
2. 在 workspace box 内采样位置，在 home 姿态的圆锥内采样朝向
3. 对每个采样点求解 IK，收集所有可行的关节配置
4. 统计分析关节角度分布，生成推荐的关节限制
"""

import argparse
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ROS2 imports
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose
from scipy.spatial.transform import Rotation as R


class WorkspaceAnalyzer(Node):
    """分析末端工作空间并推导关节限制"""

    def __init__(self):
        super().__init__('workspace_analyzer')

        # IK 服务客户端
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

        self.get_logger().info("Waiting for IK service...")
        self.ik_client.wait_for_service(timeout_sec=10.0)
        self.get_logger().info("IK service ready!")

        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7'
        ]

        self.sampled_poses = []
        self.ik_solutions = []
        self.failed_poses = []

    # ── 采样 ─────────────────────────────────────────────────────

    def sample_workspace_cone(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        home_rpy: Tuple[float, float, float],
        cone_half_angle_deg: float = 30.0,
        num_samples: int = 2000,
    ) -> List[Pose]:
        """
        在 workspace box 内采样位置，在 home 姿态的圆锥内采样朝向。

        圆锥定义: 末端 z 轴与 home z 轴的夹角 ≤ cone_half_angle_deg。
        圆锥内采用均匀立体角采样，同时对 roll 角均匀采样。

        Args:
            x/y/z_range: 工作空间 box
            home_rpy: home 姿态的 (roll, pitch, yaw)，单位弧度
            cone_half_angle_deg: 圆锥半角（度）
            num_samples: 采样数量
        """
        self.get_logger().info(
            f"Sampling: box x={x_range} y={y_range} z={z_range}, "
            f"cone={cone_half_angle_deg*2:.0f}° around home_rpy={[f'{v:.3f}' for v in home_rpy]}"
        )
        self.get_logger().info(f"Number of samples: {num_samples}")

        R_home = R.from_euler('xyz', home_rpy)
        half_angle = np.radians(cone_half_angle_deg)
        cos_half = np.cos(half_angle)

        poses = []
        for _ in range(num_samples):
            # 位置：box 内均匀采样
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            z = np.random.uniform(z_range[0], z_range[1])

            # 姿态：圆锥内均匀立体角采样
            # cos(theta) 均匀分布在 [cos(half_angle), 1]  =>  theta ∈ [0, half_angle]
            cos_theta = np.random.uniform(cos_half, 1.0)
            theta = np.arccos(cos_theta)       # 倾斜角
            phi = np.random.uniform(0, 2 * np.pi)  # 倾斜方向
            psi = np.random.uniform(-np.pi, np.pi)  # roll

            # 构建局部扰动旋转
            # 1. tilt: 绕垂直于 z 轴的方向旋转 theta
            tilt_axis = np.array([-np.sin(phi), np.cos(phi), 0.0])
            R_tilt = R.from_rotvec(theta * tilt_axis)
            # 2. roll: 绕 z 轴旋转 psi
            R_roll = R.from_rotvec(psi * np.array([0.0, 0.0, 1.0]))

            R_final = R_home * R_tilt * R_roll
            q = R_final.as_quat()  # [x, y, z, w]

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            poses.append(pose)

        self.sampled_poses = poses
        return poses

    # ── IK ────────────────────────────────────────────────────────

    def solve_ik(self, pose: Pose, timeout: float = 0.1) -> Optional[List[float]]:
        """求解单个位姿的 IK"""
        request = GetPositionIK.Request()
        request.ik_request.group_name = "xarm7"
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = 0
        request.ik_request.timeout.nanosec = int(timeout * 1e9)

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "link_base"
        pose_stamped.pose = pose
        request.ik_request.pose_stamped = pose_stamped

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout + 0.5)

        if future.result() is not None:
            response = future.result()
            if response.error_code.val == 1:  # SUCCESS
                joint_state = response.solution.joint_state
                joint_positions = []
                for joint_name in self.joint_names:
                    if joint_name in joint_state.name:
                        idx = joint_state.name.index(joint_name)
                        joint_positions.append(joint_state.position[idx])
                if len(joint_positions) == 7:
                    return joint_positions
        return None

    # ── 分析 ──────────────────────────────────────────────────────

    def analyze(self, poses: List[Pose]) -> Optional[Dict]:
        """对已采样的 poses 批量求解 IK 并统计"""
        self.get_logger().info(f"Solving IK for {len(poses)} poses...")
        self.ik_solutions = []
        self.failed_poses = []

        for i, pose in enumerate(poses):
            if i % 100 == 0:
                self.get_logger().info(f"Progress: {i}/{len(poses)}")

            solution = self.solve_ik(pose)
            if solution is not None:
                self.ik_solutions.append({'pose': pose, 'joints': solution})
            else:
                self.failed_poses.append(pose)

        success_rate = len(self.ik_solutions) / len(poses) * 100
        self.get_logger().info(f"\nIK Success Rate: {success_rate:.1f}%")
        self.get_logger().info(f"Successful: {len(self.ik_solutions)}")
        self.get_logger().info(f"Failed: {len(self.failed_poses)}")

        if len(self.ik_solutions) == 0:
            self.get_logger().error("No IK solutions found!")
            return None

        all_joints = np.array([sol['joints'] for sol in self.ik_solutions])

        joint_stats = {}
        for i, jn in enumerate(self.joint_names):
            a = all_joints[:, i]
            joint_stats[jn] = {
                'min': float(np.min(a)),
                'max': float(np.max(a)),
                'mean': float(np.mean(a)),
                'std': float(np.std(a)),
                'percentile_1': float(np.percentile(a, 1)),
                'percentile_99': float(np.percentile(a, 99)),
                'percentile_5': float(np.percentile(a, 5)),
                'percentile_95': float(np.percentile(a, 95)),
            }

        return {
            'ik_success_rate': success_rate,
            'num_solutions': len(self.ik_solutions),
            'num_failed': len(self.failed_poses),
            'joint_statistics': joint_stats,
        }

    # ── 关节限位生成 ──────────────────────────────────────────────

    def generate_joint_limits(
        self, result: Dict,
        safety_margin: float = 0.1,
        use_percentile: bool = True,
        percentile: float = 99.0
    ) -> Dict[str, Tuple[float, float]]:
        """生成推荐的关节限制"""
        joint_limits = {}
        for jn, stats in result['joint_statistics'].items():
            if use_percentile:
                lo = stats[f'percentile_{int(100 - percentile)}'] - safety_margin
                hi = stats[f'percentile_{int(percentile)}'] + safety_margin
            else:
                lo = stats['min'] - safety_margin
                hi = stats['max'] + safety_margin
            joint_limits[jn] = (lo, hi)
        return joint_limits

    # ── 验证（使用已收集的 IK 解，不重新求解）────────────────────

    def verify_limits_offline(
        self, joint_limits: Dict[str, Tuple[float, float]]
    ) -> Dict:
        """用已有的 ik_solutions 验证限位覆盖率（无需重新调 IK）"""
        total = len(self.ik_solutions)
        if total == 0:
            return {'total_tested': 0, 'lost_solutions': 0, 'loss_rate': 0, 'retained_rate': 100}

        lost = 0
        for sol in self.ik_solutions:
            for j, jn in enumerate(self.joint_names):
                lo, hi = joint_limits[jn]
                if sol['joints'][j] < lo or sol['joints'][j] > hi:
                    lost += 1
                    break

        loss_rate = lost / total * 100
        result = {
            'total_tested': total,
            'lost_solutions': lost,
            'loss_rate': loss_rate,
            'retained_rate': 100 - loss_rate,
        }

        self.get_logger().info(f"Verification (offline): {total} tested, "
                               f"{lost} lost ({loss_rate:.2f}%), "
                               f"{result['retained_rate']:.2f}% retained")
        return result

    # ── 可视化 ────────────────────────────────────────────────────

    def plot_joint_distributions(self, result: Dict, joint_limits: Dict = None):
        if len(self.ik_solutions) == 0:
            return
        all_joints = np.array([sol['joints'] for sol in self.ik_solutions])

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        for i, jn in enumerate(self.joint_names):
            ax = axes[i]
            a = all_joints[:, i]
            stats = result['joint_statistics'][jn]
            ax.hist(a, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(stats['min'], color='r', ls='--', label=f"Min {stats['min']:.2f}")
            ax.axvline(stats['max'], color='r', ls='--', label=f"Max {stats['max']:.2f}")
            ax.axvline(stats['mean'], color='g', lw=2, label=f"Mean {stats['mean']:.2f}")
            if joint_limits and jn in joint_limits:
                lo, hi = joint_limits[jn]
                ax.axvline(lo, color='b', ls=':', lw=2, label=f"Limit {lo:.2f}")
                ax.axvline(hi, color='b', ls=':', lw=2, label=f"Limit {hi:.2f}")
            ax.set_xlabel('Angle (rad)')
            ax.set_ylabel('Frequency')
            ax.set_title(jn)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        for i in range(len(self.joint_names), len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plt.savefig('joint_distributions.png', dpi=150)
        self.get_logger().info("Saved: joint_distributions.png")
        plt.close()

    def plot_workspace_coverage(self):
        if not self.ik_solutions and not self.failed_poses:
            return
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        if self.ik_solutions:
            xs = [s['pose'].position.x for s in self.ik_solutions]
            ys = [s['pose'].position.y for s in self.ik_solutions]
            zs = [s['pose'].position.z for s in self.ik_solutions]
            ax.scatter(xs, ys, zs, c='g', marker='o', s=10, alpha=0.5, label='IK OK')
        if self.failed_poses:
            xs = [p.position.x for p in self.failed_poses]
            ys = [p.position.y for p in self.failed_poses]
            zs = [p.position.z for p in self.failed_poses]
            ax.scatter(xs, ys, zs, c='r', marker='x', s=20, alpha=0.8, label='IK Fail')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Workspace Coverage')
        ax.legend()
        plt.savefig('workspace_coverage.png', dpi=150)
        self.get_logger().info("Saved: workspace_coverage.png")
        plt.close()

    # ── 输出 ──────────────────────────────────────────────────────

    def save_results(self, result: Dict, joint_limits: Dict, verification: Dict,
                     filename: str = 'workspace_analysis.json'):
        output = {
            'analysis': result,
            'recommended_joint_limits': {k: list(v) for k, v in joint_limits.items()},
            'verification': verification,
        }
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        self.get_logger().info(f"Saved: {filename}")

    def print_launch_file_config(self, joint_limits: Dict):
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("LAUNCH FILE CONFIGURATION")
        self.get_logger().info("=" * 60)
        print("joint_limits_yaml = {")
        print("    'robot_description_planning': {")
        print("        'joint_limits': {")
        for jn, (lo, hi) in joint_limits.items():
            print(f"            '{jn}': {{")
            print(f"                'has_velocity_limits': True,")
            print(f"                'max_velocity': 1.0,")
            print(f"                'has_acceleration_limits': True,")
            print(f"                'max_acceleration': 2.0,")
            print(f"                'has_position_limits': True,")
            print(f"                'min_position': {lo:.4f},")
            print(f"                'max_position': {hi:.4f}")
            print(f"            }},")
        print("        }")
        print("    }")
        print("}")


# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="从工作空间推导关节限位")
    parser.add_argument("--config", default="config/xarm7_config.json", help="机器人配置文件路径")
    parser.add_argument("--cone-angle", type=float, default=60.0,
                        help="圆锥全角（度），默认 60 即 home 姿态 ±30°")
    parser.add_argument("--num-samples", type=int, default=2000, help="采样数量")
    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    ws = config["workspace_limits"]
    x_range = tuple(ws["x"])
    y_range = tuple(ws["y"])
    z_range = tuple(ws["z"])

    home_rpy = tuple(config["home"]["orientation"])  # (roll, pitch, yaw)
    cone_half = args.cone_angle / 2.0

    print("=" * 60)
    print("Workspace → Joint Limits Analyzer")
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Workspace:    x={list(x_range)} y={list(y_range)} z={list(z_range)}")
    print(f"  Home RPY:     {list(home_rpy)}")
    print(f"  Cone:         {args.cone_angle:.0f}° (half={cone_half:.0f}°)")
    print(f"  Samples:      {args.num_samples}")
    print()

    rclpy.init()
    analyzer = WorkspaceAnalyzer()

    try:
        # Step 1: 采样
        analyzer.get_logger().info("\n" + "=" * 60)
        analyzer.get_logger().info("STEP 1: Sampling workspace (box + cone)")
        analyzer.get_logger().info("=" * 60)

        poses = analyzer.sample_workspace_cone(
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            home_rpy=home_rpy,
            cone_half_angle_deg=cone_half,
            num_samples=args.num_samples,
        )

        # Step 2: IK 求解 + 统计
        analyzer.get_logger().info("\n" + "=" * 60)
        analyzer.get_logger().info("STEP 2: Solving IK & statistics")
        analyzer.get_logger().info("=" * 60)

        result = analyzer.analyze(poses)
        if result is None:
            analyzer.get_logger().error("Analysis failed!")
            return

        # Step 3: 生成关节限位
        analyzer.get_logger().info("\n" + "=" * 60)
        analyzer.get_logger().info("STEP 3: Generating Joint Limits")
        analyzer.get_logger().info("=" * 60)

        limits_conservative = analyzer.generate_joint_limits(
            result, safety_margin=0.1, use_percentile=True, percentile=99.0)
        limits_balanced = analyzer.generate_joint_limits(
            result, safety_margin=0.15, use_percentile=True, percentile=95.0)
        limits_aggressive = analyzer.generate_joint_limits(
            result, safety_margin=0.05, use_percentile=False)

        for label, lim in [("A: Conservative (P99+0.1)", limits_conservative),
                           ("B: Balanced (P95+0.15)", limits_balanced),
                           ("C: Aggressive (min/max+0.05)", limits_aggressive)]:
            analyzer.get_logger().info(f"\n=== {label} ===")
            for jn, (lo, hi) in lim.items():
                analyzer.get_logger().info(f"  {jn}: [{lo:.3f}, {hi:.3f}]")

        # Step 4: 离线验证（使用已有 IK 解，不重新调服务）
        analyzer.get_logger().info("\n" + "=" * 60)
        analyzer.get_logger().info("STEP 4: Offline Verification")
        analyzer.get_logger().info("=" * 60)

        v_con = analyzer.verify_limits_offline(limits_conservative)
        v_bal = analyzer.verify_limits_offline(limits_balanced)
        v_agg = analyzer.verify_limits_offline(limits_aggressive)

        analyzer.get_logger().info(f"\n=== Summary ===")
        analyzer.get_logger().info(f"  Conservative: {v_con['retained_rate']:.2f}% retained")
        analyzer.get_logger().info(f"  Balanced:     {v_bal['retained_rate']:.2f}% retained")
        analyzer.get_logger().info(f"  Aggressive:   {v_agg['retained_rate']:.2f}% retained")

        # Step 5: 可视化
        analyzer.get_logger().info("\n" + "=" * 60)
        analyzer.get_logger().info("STEP 5: Visualizations")
        analyzer.get_logger().info("=" * 60)

        analyzer.plot_joint_distributions(result, limits_balanced)
        analyzer.plot_workspace_coverage()

        # Step 6: 保存 & 输出
        analyzer.save_results(result, limits_balanced, v_bal)
        analyzer.print_launch_file_config(limits_balanced)

        # 推荐
        analyzer.get_logger().info("\n" + "=" * 60)
        analyzer.get_logger().info("RECOMMENDATION")
        analyzer.get_logger().info("=" * 60)
        if v_bal['retained_rate'] >= 95:
            analyzer.get_logger().info("✓ Balanced strategy recommended (≥95% retained)")
        elif v_con['retained_rate'] >= 95:
            analyzer.get_logger().info("✓ Conservative strategy recommended (≥95% retained)")
        else:
            analyzer.get_logger().info("⚠ All strategies lose >5%. Consider expanding limits or reducing workspace.")

    except Exception as e:
        analyzer.get_logger().error(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
