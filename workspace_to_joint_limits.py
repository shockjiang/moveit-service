#!/usr/bin/env python3
"""
从末端执行器工作空间推导关节空间限制

使用方法:
    python3 workspace_to_joint_limits.py

功能:
1. 在指定的末端工作空间中采样
2. 对每个采样点求解 IK，收集所有可行的关节配置
3. 统计分析关节角度分布
4. 生成推荐的关节限制
5. 验证限制后是否会丢解
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ROS2 imports
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose
import tf_transformations


class WorkspaceAnalyzer(Node):
    """分析末端工作空间并推导关节限制"""

    def __init__(self):
        super().__init__('workspace_analyzer')

        # IK 服务客户端
        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik'
        )

        # 等待服务
        self.get_logger().info("Waiting for IK service...")
        self.ik_client.wait_for_service(timeout_sec=10.0)
        self.get_logger().info("IK service ready!")

        # 关节名称
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7'
        ]

        # 存储采样结果
        self.sampled_poses = []  # 采样的末端位姿
        self.ik_solutions = []   # 所有 IK 解
        self.failed_poses = []   # 无解的位姿

    def sample_workspace(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        num_samples: int = 1000,
        orientations: Optional[List[Tuple[float, float, float]]] = None
    ) -> List[Pose]:
        """
        在工作空间中采样

        Args:
            x_range: (x_min, x_max) 单位: 米
            y_range: (y_min, y_max) 单位: 米
            z_range: (z_min, z_max) 单位: 米
            num_samples: 采样点数量
            orientations: 姿态列表 [(roll, pitch, yaw), ...], 如果为 None 则使用默认姿态

        Returns:
            采样的位姿列表
        """
        self.get_logger().info(f"Sampling workspace: x={x_range}, y={y_range}, z={z_range}")
        self.get_logger().info(f"Number of samples: {num_samples}")

        poses = []

        # 默认姿态: 向下
        if orientations is None:
            orientations = [
                (-np.pi, 0.0, 0.0),  # 向下
                (-np.pi, 0.0, np.pi/4),  # 向下 + 旋转 45°
                (-np.pi, 0.0, -np.pi/4),  # 向下 + 旋转 -45°
                (-np.pi, 0.0, np.pi/2),  # 向下 + 旋转 90°
            ]

        # 在空间中均匀采样
        samples_per_orientation = num_samples // len(orientations)

        for rpy in orientations:
            for _ in range(samples_per_orientation):
                # 随机采样位置
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(y_range[0], y_range[1])
                z = np.random.uniform(z_range[0], z_range[1])

                # 创建位姿
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z

                # 设置姿态
                q = tf_transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                pose.orientation.x = q[0]
                pose.orientation.y = q[1]
                pose.orientation.z = q[2]
                pose.orientation.w = q[3]

                poses.append(pose)

        self.sampled_poses = poses
        return poses

    def solve_ik(self, pose: Pose, timeout: float = 0.1) -> Optional[List[float]]:
        """
        求解单个位姿的 IK

        Args:
            pose: 目标位姿
            timeout: 超时时间（秒）

        Returns:
            关节角度列表，如果无解返回 None
        """
        # 创建 IK 请求
        request = GetPositionIK.Request()
        request.ik_request.group_name = "xarm7"
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = 0
        request.ik_request.timeout.nanosec = int(timeout * 1e9)

        # 设置目标位姿
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "link_base"
        pose_stamped.pose = pose
        request.ik_request.pose_stamped = pose_stamped

        # 调用服务
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout + 0.5)

        if future.result() is not None:
            response = future.result()
            if response.error_code.val == 1:  # SUCCESS
                # 提取关节角度
                joint_state = response.solution.joint_state
                joint_positions = []
                for joint_name in self.joint_names:
                    if joint_name in joint_state.name:
                        idx = joint_state.name.index(joint_name)
                        joint_positions.append(joint_state.position[idx])

                if len(joint_positions) == 7:
                    return joint_positions

        return None

    def analyze_workspace(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        num_samples: int = 1000,
        orientations: Optional[List[Tuple[float, float, float]]] = None
    ) -> Dict:
        """
        分析工作空间并收集 IK 解

        Returns:
            分析结果字典
        """
        # 采样工作空间
        poses = self.sample_workspace(x_range, y_range, z_range, num_samples, orientations)

        # 求解 IK
        self.get_logger().info(f"Solving IK for {len(poses)} poses...")
        self.ik_solutions = []
        self.failed_poses = []

        for i, pose in enumerate(poses):
            if i % 100 == 0:
                self.get_logger().info(f"Progress: {i}/{len(poses)}")

            solution = self.solve_ik(pose)
            if solution is not None:
                self.ik_solutions.append({
                    'pose': pose,
                    'joints': solution
                })
            else:
                self.failed_poses.append(pose)

        success_rate = len(self.ik_solutions) / len(poses) * 100
        self.get_logger().info(f"\nIK Success Rate: {success_rate:.1f}%")
        self.get_logger().info(f"Successful: {len(self.ik_solutions)}")
        self.get_logger().info(f"Failed: {len(self.failed_poses)}")

        # 统计分析
        if len(self.ik_solutions) == 0:
            self.get_logger().error("No IK solutions found!")
            return None

        # 提取所有关节角度
        all_joints = np.array([sol['joints'] for sol in self.ik_solutions])

        # 计算统计信息
        joint_stats = {}
        for i, joint_name in enumerate(self.joint_names):
            joint_angles = all_joints[:, i]
            joint_stats[joint_name] = {
                'min': float(np.min(joint_angles)),
                'max': float(np.max(joint_angles)),
                'mean': float(np.mean(joint_angles)),
                'std': float(np.std(joint_angles)),
                'percentile_1': float(np.percentile(joint_angles, 1)),
                'percentile_99': float(np.percentile(joint_angles, 99)),
                'percentile_5': float(np.percentile(joint_angles, 5)),
                'percentile_95': float(np.percentile(joint_angles, 95)),
            }

        result = {
            'workspace': {
                'x_range': x_range,
                'y_range': y_range,
                'z_range': z_range,
                'num_samples': num_samples,
            },
            'ik_success_rate': success_rate,
            'num_solutions': len(self.ik_solutions),
            'num_failed': len(self.failed_poses),
            'joint_statistics': joint_stats,
        }

        return result

    def generate_joint_limits(
        self,
        result: Dict,
        safety_margin: float = 0.1,
        use_percentile: bool = True,
        percentile: float = 99.0
    ) -> Dict[str, Tuple[float, float]]:
        """
        生成推荐的关节限制

        Args:
            result: analyze_workspace 的结果
            safety_margin: 安全裕度（弧度）
            use_percentile: 是否使用百分位数而不是绝对最小/最大值
            percentile: 使用的百分位数（如 99 表示 1%-99%）

        Returns:
            {joint_name: (min_limit, max_limit)}
        """
        joint_limits = {}

        for joint_name, stats in result['joint_statistics'].items():
            if use_percentile:
                # 使用百分位数（更保守）
                lower_key = f'percentile_{int(100 - percentile)}'
                upper_key = f'percentile_{int(percentile)}'
                min_val = stats[lower_key] - safety_margin
                max_val = stats[upper_key] + safety_margin
            else:
                # 使用绝对最小/最大值
                min_val = stats['min'] - safety_margin
                max_val = stats['max'] + safety_margin

            joint_limits[joint_name] = (min_val, max_val)

        return joint_limits

    def verify_limits(
        self,
        joint_limits: Dict[str, Tuple[float, float]],
        test_poses: Optional[List[Pose]] = None
    ) -> Dict:
        """
        验证关节限制是否会导致丢解

        Args:
            joint_limits: 要验证的关节限制
            test_poses: 测试位姿列表，如果为 None 则使用之前采样的位姿

        Returns:
            验证结果
        """
        if test_poses is None:
            test_poses = self.sampled_poses

        self.get_logger().info(f"\nVerifying joint limits on {len(test_poses)} poses...")

        lost_solutions = 0
        total_tested = 0

        for i, pose in enumerate(test_poses):
            if i % 100 == 0:
                self.get_logger().info(f"Verification progress: {i}/{len(test_poses)}")

            solution = self.solve_ik(pose)
            if solution is not None:
                total_tested += 1
                # 检查解是否在限制范围内
                within_limits = True
                for j, joint_name in enumerate(self.joint_names):
                    min_limit, max_limit = joint_limits[joint_name]
                    if solution[j] < min_limit or solution[j] > max_limit:
                        within_limits = False
                        break

                if not within_limits:
                    lost_solutions += 1

        loss_rate = (lost_solutions / total_tested * 100) if total_tested > 0 else 0

        result = {
            'total_tested': total_tested,
            'lost_solutions': lost_solutions,
            'loss_rate': loss_rate,
            'retained_rate': 100 - loss_rate
        }

        self.get_logger().info(f"\n=== Verification Results ===")
        self.get_logger().info(f"Total tested: {total_tested}")
        self.get_logger().info(f"Lost solutions: {lost_solutions}")
        self.get_logger().info(f"Loss rate: {loss_rate:.2f}%")
        self.get_logger().info(f"Retained rate: {result['retained_rate']:.2f}%")

        return result

    def plot_joint_distributions(self, result: Dict, joint_limits: Dict = None):
        """绘制关节角度分布图"""
        if len(self.ik_solutions) == 0:
            self.get_logger().warn("No solutions to plot")
            return

        all_joints = np.array([sol['joints'] for sol in self.ik_solutions])

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for i, joint_name in enumerate(self.joint_names):
            ax = axes[i]
            joint_angles = all_joints[:, i]
            stats = result['joint_statistics'][joint_name]

            # 绘制直方图
            ax.hist(joint_angles, bins=50, alpha=0.7, edgecolor='black')

            # 绘制统计线
            ax.axvline(stats['min'], color='r', linestyle='--', label=f"Min: {stats['min']:.2f}")
            ax.axvline(stats['max'], color='r', linestyle='--', label=f"Max: {stats['max']:.2f}")
            ax.axvline(stats['mean'], color='g', linestyle='-', linewidth=2, label=f"Mean: {stats['mean']:.2f}")

            # 如果提供了限制，绘制限制线
            if joint_limits and joint_name in joint_limits:
                min_limit, max_limit = joint_limits[joint_name]
                ax.axvline(min_limit, color='b', linestyle=':', linewidth=2, label=f"Limit Min: {min_limit:.2f}")
                ax.axvline(max_limit, color='b', linestyle=':', linewidth=2, label=f"Limit Max: {max_limit:.2f}")

            ax.set_xlabel('Angle (rad)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{joint_name}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 删除多余的子图
        for i in range(len(self.joint_names), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig('joint_distributions.png', dpi=150)
        self.get_logger().info("Saved joint distribution plot to: joint_distributions.png")
        plt.close()

    def plot_workspace_coverage(self):
        """绘制工作空间覆盖图"""
        if len(self.ik_solutions) == 0 and len(self.failed_poses) == 0:
            self.get_logger().warn("No poses to plot")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制成功的位姿
        if len(self.ik_solutions) > 0:
            success_x = [sol['pose'].position.x for sol in self.ik_solutions]
            success_y = [sol['pose'].position.y for sol in self.ik_solutions]
            success_z = [sol['pose'].position.z for sol in self.ik_solutions]
            ax.scatter(success_x, success_y, success_z, c='g', marker='o', s=10, alpha=0.5, label='IK Success')

        # 绘制失败的位姿
        if len(self.failed_poses) > 0:
            failed_x = [pose.position.x for pose in self.failed_poses]
            failed_y = [pose.position.y for pose in self.failed_poses]
            failed_z = [pose.position.z for pose in self.failed_poses]
            ax.scatter(failed_x, failed_y, failed_z, c='r', marker='x', s=20, alpha=0.8, label='IK Failed')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Workspace Coverage')
        ax.legend()

        plt.savefig('workspace_coverage.png', dpi=150)
        self.get_logger().info("Saved workspace coverage plot to: workspace_coverage.png")
        plt.close()

    def save_results(self, result: Dict, joint_limits: Dict, verification: Dict, filename: str = 'workspace_analysis.json'):
        """保存分析结果到 JSON 文件"""
        output = {
            'analysis': result,
            'recommended_joint_limits': {k: list(v) for k, v in joint_limits.items()},
            'verification': verification
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        self.get_logger().info(f"Saved results to: {filename}")

    def print_launch_file_config(self, joint_limits: Dict):
        """打印可以直接用于 launch 文件的配置"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("LAUNCH FILE CONFIGURATION")
        self.get_logger().info("="*60)
        self.get_logger().info("\nCopy this into your launch file:\n")

        print("joint_limits_yaml = {")
        print("    'robot_description_planning': {")
        print("        'joint_limits': {")

        for joint_name, (min_limit, max_limit) in joint_limits.items():
            print(f"            '{joint_name}': {{")
            print(f"                'has_velocity_limits': True,")
            print(f"                'max_velocity': 1.0,")
            print(f"                'has_acceleration_limits': True,")
            print(f"                'max_acceleration': 2.0,")
            print(f"                'has_position_limits': True,")
            print(f"                'min_position': {min_limit:.4f},")
            print(f"                'max_position': {max_limit:.4f}")
            print(f"            }},")

        print("        }")
        print("    }")
        print("}")


def main():
    """主函数"""
    rclpy.init()

    analyzer = WorkspaceAnalyzer()

    # ========================================
    # 配置你的工作空间
    # ========================================

    # 示例 1: 桌面抓取场景
    workspace_config = {
        'x_range': (0.2, 0.6),    # 前方 20cm 到 60cm
        'y_range': (-0.3, 0.3),   # 左右 ±30cm
        'z_range': (0.1, 0.5),    # 高度 10cm 到 50cm
        'num_samples': 2000,      # 采样点数量
        'orientations': [
            (-np.pi, 0.0, 0.0),           # 向下
            (-np.pi, 0.0, np.pi/4),       # 向下 + 旋转 45°
            (-np.pi, 0.0, -np.pi/4),      # 向下 + 旋转 -45°
            (-np.pi, 0.0, np.pi/2),       # 向下 + 旋转 90°
            (-np.pi + 0.3, 0.0, 0.0),     # 稍微倾斜
        ]
    }

    # 示例 2: 更大的工作空间
    # workspace_config = {
    #     'x_range': (0.1, 0.7),
    #     'y_range': (-0.5, 0.5),
    #     'z_range': (0.0, 0.6),
    #     'num_samples': 3000,
    #     'orientations': None  # 使用默认姿态
    # }

    try:
        # 步骤 1: 分析工作空间
        analyzer.get_logger().info("\n" + "="*60)
        analyzer.get_logger().info("STEP 1: Analyzing Workspace")
        analyzer.get_logger().info("="*60)

        result = analyzer.analyze_workspace(**workspace_config)

        if result is None:
            analyzer.get_logger().error("Analysis failed!")
            return

        # 步骤 2: 生成关节限制（多种策略）
        analyzer.get_logger().info("\n" + "="*60)
        analyzer.get_logger().info("STEP 2: Generating Joint Limits")
        analyzer.get_logger().info("="*60)

        # 策略 A: 保守（99% 百分位 + 0.1 rad 裕度）
        limits_conservative = analyzer.generate_joint_limits(
            result,
            safety_margin=0.1,
            use_percentile=True,
            percentile=99.0
        )

        # 策略 B: 平衡（95% 百分位 + 0.15 rad 裕度）
        limits_balanced = analyzer.generate_joint_limits(
            result,
            safety_margin=0.15,
            use_percentile=True,
            percentile=95.0
        )

        # 策略 C: 激进（绝对最小/最大 + 0.05 rad 裕度）
        limits_aggressive = analyzer.generate_joint_limits(
            result,
            safety_margin=0.05,
            use_percentile=False
        )

        # 打印所有策略
        analyzer.get_logger().info("\n=== Strategy A: Conservative (99% percentile + 0.1 rad) ===")
        for joint_name, (min_limit, max_limit) in limits_conservative.items():
            analyzer.get_logger().info(f"{joint_name}: [{min_limit:.3f}, {max_limit:.3f}]")

        analyzer.get_logger().info("\n=== Strategy B: Balanced (95% percentile + 0.15 rad) ===")
        for joint_name, (min_limit, max_limit) in limits_balanced.items():
            analyzer.get_logger().info(f"{joint_name}: [{min_limit:.3f}, {max_limit:.3f}]")

        analyzer.get_logger().info("\n=== Strategy C: Aggressive (absolute min/max + 0.05 rad) ===")
        for joint_name, (min_limit, max_limit) in limits_aggressive.items():
            analyzer.get_logger().info(f"{joint_name}: [{min_limit:.3f}, {max_limit:.3f}]")

        # 步骤 3: 验证限制（使用平衡策略）
        analyzer.get_logger().info("\n" + "="*60)
        analyzer.get_logger().info("STEP 3: Verifying Joint Limits")
        analyzer.get_logger().info("="*60)

        verification_conservative = analyzer.verify_limits(limits_conservative)
        verification_balanced = analyzer.verify_limits(limits_balanced)
        verification_aggressive = analyzer.verify_limits(limits_aggressive)

        analyzer.get_logger().info("\n=== Verification Summary ===")
        analyzer.get_logger().info(f"Conservative: {verification_conservative['retained_rate']:.2f}% retained")
        analyzer.get_logger().info(f"Balanced: {verification_balanced['retained_rate']:.2f}% retained")
        analyzer.get_logger().info(f"Aggressive: {verification_aggressive['retained_rate']:.2f}% retained")

        # 步骤 4: 生成可视化
        analyzer.get_logger().info("\n" + "="*60)
        analyzer.get_logger().info("STEP 4: Generating Visualizations")
        analyzer.get_logger().info("="*60)

        analyzer.plot_joint_distributions(result, limits_balanced)
        analyzer.plot_workspace_coverage()

        # 步骤 5: 保存结果
        analyzer.get_logger().info("\n" + "="*60)
        analyzer.get_logger().info("STEP 5: Saving Results")
        analyzer.get_logger().info("="*60)

        analyzer.save_results(result, limits_balanced, verification_balanced)

        # 步骤 6: 打印 launch 文件配置
        analyzer.print_launch_file_config(limits_balanced)

        # 推荐
        analyzer.get_logger().info("\n" + "="*60)
        analyzer.get_logger().info("RECOMMENDATION")
        analyzer.get_logger().info("="*60)

        if verification_balanced['retained_rate'] >= 95:
            analyzer.get_logger().info("✓ Balanced strategy is recommended (>95% solutions retained)")
        elif verification_conservative['retained_rate'] >= 95:
            analyzer.get_logger().info("✓ Conservative strategy is recommended (>95% solutions retained)")
        else:
            analyzer.get_logger().warn("⚠ All strategies lose >5% of solutions. Consider:")
            analyzer.get_logger().warn("  1. Expanding the joint limits")
            analyzer.get_logger().warn("  2. Reducing the workspace size")
            analyzer.get_logger().warn("  3. Using workspace constraints instead of joint limits")

    except Exception as e:
        analyzer.get_logger().error(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
