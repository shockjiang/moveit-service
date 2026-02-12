#!/usr/bin/env python3
"""
Simple script to move the xArm using MoveIt action interface.
This is more reliable than MoveItPy for basic usage.

Usage:
1. First launch MoveIt in another terminal:
   ros2 launch xarm7_moveit_config demo.launch.py

2. Then run this script:
   python3 tests/move_arm_simple.py
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
import sys
import json
import math

class MoveArmClient(Node):
    def __init__(self, end_effector_link='link_eef'):
        super().__init__('move_arm_client')
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self._fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.timeout_sec = 5.0
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.end_effector_link = end_effector_link  # 'link_eef' or 'link_tcp'
        self.get_logger().info(f'Using end effector link: {self.end_effector_link}')

    def is_moveit_server_available(self):
        """Check if the MoveIt action server is available"""
        self.get_logger().info('Checking for MoveIt action server...')
        alive = self._action_client.wait_for_server(timeout_sec=5.0)
        #add checking for FK and IK services as well, any of them is not available, we consider MoveIt is not fully available
        fk_alive = self._fk_client.wait_for_service(timeout_sec=2.0)
        ik_alive = self._ik_client.wait_for_service(timeout_sec=2.0)
        alive = alive and fk_alive and ik_alive
        if alive:
            self.get_logger().info('moveit service is available!')
        else:
            self.get_logger().error('moveit service is not available, make sure to launch MoveIt first!') 
        return alive
    
    def send_joint_goal(self, joint_values, plan_only=False, start_pos=None):
        """Move to specific joint positions

        Args:
            joint_values: List of 7 joint angles in radians
            plan_only: If True, only plan without executing
            start_pos: Optional list of 7 joint angles in radians as start state
        """
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "xarm7"
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 1.0 #[0, 1]
        goal_msg.request.max_acceleration_scaling_factor = 1.0 #[0, 1]

        # Set whether to only plan or also execute
        goal_msg.planning_options.plan_only = plan_only

        # Set start state if provided
        if start_pos is not None:
            start_state = RobotState()
            start_state.joint_state.name = self.joint_names
            start_state.joint_state.position = start_pos
            goal_msg.request.start_state = start_state
            self.get_logger().info(f'Using custom start state: {[f"{math.degrees(j):.1f}°" for j in start_pos]}')

        # Set joint constraints (all joints must be satisfied together)
        constraints = Constraints()
        for name, value in zip(self.joint_names, joint_values):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = value
            constraint.tolerance_above = 0.01
            constraint.tolerance_below = 0.01
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        goal_msg.request.goal_constraints.append(constraints)

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info('Sending goal...')
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=self.timeout_sec)

        if not send_goal_future.done():
            self.get_logger().error(f'Goal submission timed out after {self.timeout_sec}s')
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.timeout_sec * 6)

        if not result_future.done():
            self.get_logger().error(f'Planning/execution timed out after {self.timeout_sec * 6}s')
            # Cancel the goal
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
            return False

        result = result_future.result().result
        self.get_logger().info(f'Result: {result.error_code.val}')
        # self.get_logger().info(f'Result: {result}')
        self.parse_result(result, 'moveit_result_joint.json')  # Save result for debugging
        # breakpoint()

        return result.error_code.val == 1  # SUCCESS

    def parse_result(self, result, out_file):
        """Parse the MoveIt result and save to JSON file for debugging

        Args:
            result: The result message from MoveIt
            out_file: Path to save the parsed result
        """
        import json

        # Convert result to dictionary
        result_dict = {
            'error_code': result.error_code.val,
            'planning_time': result.planning_time,
            'planned_trajectory': self._trajectory_to_dict(result.planned_trajectory),
            'executed_trajectory': self._trajectory_to_dict(result.executed_trajectory) if hasattr(result, 'executed_trajectory') else None
        }

        # Save to JSON file
        with open(out_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

        self.get_logger().info(f'Result saved to {out_file}')

    def _trajectory_to_dict(self, trajectory):
        """Convert RobotTrajectory message to dictionary with end-effector poses

        Args:
            trajectory: RobotTrajectory message

        Returns:
            Dictionary representation of the trajectory with end-effector poses
        """
        if not trajectory or not trajectory.joint_trajectory.points:
            return None

        # Check if FK service is available
        fk_available = self._fk_client.wait_for_service(timeout_sec=2.0)
        if not fk_available:
            self.get_logger().warn('FK service not available, end_effector poses will be None')

        traj_dict = {
            'joint_names': list(trajectory.joint_trajectory.joint_names),
            'points': []
        }

        for point in trajectory.joint_trajectory.points:
            point_dict = {
                'positions': list(point.positions),
                'velocities': list(point.velocities) if point.velocities else [],
                'accelerations': list(point.accelerations) if point.accelerations else [],
                'time_from_start': {
                    'sec': point.time_from_start.sec,
                    'nanosec': point.time_from_start.nanosec
                },
                'end_effector': None
            }

            # Calculate end-effector pose using FK
            if fk_available:
                request = GetPositionFK.Request()
                request.header.frame_id = 'link_base'
                request.fk_link_names = [self.end_effector_link]

                joint_state = JointState()
                joint_state.name = self.joint_names
                joint_state.position = list(point.positions)
                request.robot_state.joint_state = joint_state

                future = self._fk_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

                if future.done():
                    result = future.result()
                    if result.error_code.val == 1:  # SUCCESS
                        pose_stamped = result.pose_stamped[0]
                        point_dict['end_effector'] = {
                            'position': {
                                'x': pose_stamped.pose.position.x,
                                'y': pose_stamped.pose.position.y,
                                'z': pose_stamped.pose.position.z
                            },
                            'orientation': {
                                'x': pose_stamped.pose.orientation.x,
                                'y': pose_stamped.pose.orientation.y,
                                'z': pose_stamped.pose.orientation.z,
                                'w': pose_stamped.pose.orientation.w
                            }
                        }

            traj_dict['points'].append(point_dict)

        return traj_dict


    def send_pose_goal(self, end_pos, plan_only=False, start_pos=None):
        """Move to specific end-effector pose

        Args:
            x, y, z: Target position in meters (optional if end_pos is provided)
            qx, qy, qz, qw: Target orientation as quaternion (optional if end_pos is provided)
            plan_only: If True, only plan without executing
            start_pos: Optional list of 7 joint angles in radians as start state
            end_pos: Optional dict with 'position' [x,y,z] and 'orientation' [qx,qy,qz,qw]
        """
        x, y, z = end_pos['position']
        qx, qy, qz, qw = end_pos['orientation']
        # Validate that we have position
        if x is None or y is None or z is None:
            self.get_logger().error('Position (x, y, z) must be provided either directly or via end_pos')
            return False
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "xarm7"
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 1.0
        goal_msg.request.max_acceleration_scaling_factor = 1.0

        # Set whether to only plan or also execute
        goal_msg.planning_options.plan_only = plan_only

        # Set start state if provided
        if start_pos is not None:
            start_state = RobotState()
            start_state.joint_state.name = self.joint_names
            start_state.joint_state.position = start_pos
            goal_msg.request.start_state = start_state
            self.get_logger().info(f'Using custom start state: {[f"{math.degrees(j):.1f}°" for j in start_pos]}')

        # Set pose constraint
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "link_base"
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw

        constraints = Constraints()

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "link_base"
        pos_constraint.link_name = self.end_effector_link
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.001]  # Small tolerance
        pos_constraint.constraint_region.primitives.append(primitive)
        pos_constraint.constraint_region.primitive_poses.append(pose_stamped.pose)
        pos_constraint.weight = 1.0

        constraints.position_constraints.append(pos_constraint)

        # Orientation constraint
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = "link_base"
        orient_constraint.link_name = self.end_effector_link
        orient_constraint.orientation = pose_stamped.pose.orientation
        orient_constraint.absolute_x_axis_tolerance = 0.1
        orient_constraint.absolute_y_axis_tolerance = 0.1
        orient_constraint.absolute_z_axis_tolerance = 0.1
        orient_constraint.weight = 1.0

        constraints.orientation_constraints.append(orient_constraint)
        goal_msg.request.goal_constraints.append(constraints)

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info(f'Sending pose goal: {end_pos}')
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=self.timeout_sec)

        if not send_goal_future.done():
            self.get_logger().error(f'Goal submission timed out after {self.timeout_sec}s')
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.timeout_sec * 6)

        if not result_future.done():
            self.get_logger().error(f'Planning/execution timed out after {self.timeout_sec * 6}s')
            # Cancel the goal
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
            return False

        result = result_future.result().result
        self.get_logger().info(f'Result: {result.error_code.val}')
        self.parse_result(result, 'moveit_result_cart.json')  # Save result for debugging
        # self.get_logger().info(f'Result: {result}')
        return result.error_code.val == 1  # SUCCESS

def main():
    rclpy.init()
    end = 'link_tcp' if len(sys.argv) > 1 and sys.argv[1] == 'tcp' else 'link_eef'
    end = 'link_tcp'
    client = MoveArmClient(end_effector_link=end)

    print("\n=== xArm Movement Test ===")

    alive = client.is_moveit_server_available()
    if not alive:
        print("MoveIt action server is not available. Please launch MoveIt before running this script: make run")
        return

    # Load home position from config

    home = {
        "end_effector": {
            "position": [0.270, 0.0, 0.307],
            "orientation": [-3.14159, 0.0, 0.0],
        },
        # "joints": [0.00175, -0.71728, -0.01047, 0.78900, 0.01222, 1.50631, 0.01047]
        # "joints": [0.001745, -0.717278, -0.010472, 0.789003, 0.012217, 1.506311, 0.010472]
        "joints": [-0.000, -0.638, 0.000, 0.332, 0.000, 0.970, -0.000]
    }

    target = {
        "joints": [
          1.6622234460021021,
          1.2401318323092207,
          -1.7764093688457474,
          0.9979773983738144,
          1.2252734989481147,
          -1.485025259766139,
          -2.5484964788645064
        ],
        "end_effector": {
            "position": [0.3005336978498776, 0.20005642529744397, 0.5008136122083992],
            "orientation": [0.04380878657316783, 0.031363280885389304, 0.04066292615045238, 0.9977192296773548]
        }
    } 

    for point3d in [home, target]:
        # calculate home joint angles using IK if only position/orientation is provided
        print("Calculating home joint angles using IK...")

        # Check if IK service is available
        if not client._ik_client.wait_for_service(timeout_sec=2.0):
            print("✗ IK service not available")
            return

        # Convert orientation from euler angles (roll, pitch, yaw) to quaternion
        
        if len(point3d['end_effector']['orientation']) == 3:
            euler = home['end_effector']['orientation']  # [roll, pitch, yaw] in radians
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_euler('xyz', euler).as_quat()  # [x, y, z, w]
        elif len(point3d['end_effector']['orientation']) == 4:
            quat = point3d['end_effector']['orientation']  # Already in quaternion format
        else:
            raise NotImplementedError("Orientation must be either 3 (euler) or 4 (quaternion) values")
        
        # breakpoint()

        # Prepare IK request
        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = "xarm7"
        ik_request.ik_request.avoid_collisions = True
        ik_request.ik_request.timeout.sec = 5

        # Set the link name for which to compute IK
        ik_request.ik_request.ik_link_name = client.end_effector_link

        # Set target pose
        ik_request.ik_request.pose_stamped.header.frame_id = "link_base"
        ik_request.ik_request.pose_stamped.pose.position.x = point3d['end_effector']['position'][0]
        ik_request.ik_request.pose_stamped.pose.position.y =  point3d['end_effector']['position'][1]
        ik_request.ik_request.pose_stamped.pose.position.z =  point3d['end_effector']['position'][2]
        ik_request.ik_request.pose_stamped.pose.orientation.x = quat[0]
        ik_request.ik_request.pose_stamped.pose.orientation.y = quat[1]
        ik_request.ik_request.pose_stamped.pose.orientation.z = quat[2]
        ik_request.ik_request.pose_stamped.pose.orientation.w = quat[3]

        # Call IK service
        future = client._ik_client.call_async(ik_request)
        rclpy.spin_until_future_complete(client, future, timeout_sec=5.0)

        if future.done():
            result = future.result()
            if result.error_code.val == 1:  # SUCCESS
                joint_positions = result.solution.joint_state.position
                # point3d['joints'] = joint_positions
                joints_deg = [math.degrees(j) for j in joint_positions]
                print(f"✓ IK solved deg: {[f'{j:.1f}' for j in joints_deg]}")
                print(f"✓ IK solved rad: {[f'{j:.3f}' for j in joint_positions]}")
                print(f'original  joint: {point3d["joints"]}')
            else:
                print(f"✗ IK failed with error code: {result.error_code.val}")
                return
        else:
            print("✗ IK computation timed out and exit")
            return

    # Verify IK solution by computing FK and comparing with original end-effector pose
    print("\nVerifying IK solutions...")
    for name, point3d in [("home", home), ("target", target)]:
        # Prepare FK request
        fk_request = GetPositionFK.Request()
        fk_request.header.frame_id = 'link_base'
        fk_request.fk_link_names = [client.end_effector_link]

        joint_state = JointState()
        joint_state.name = client.joint_names
        joint_state.position = list(point3d['joints'])
        fk_request.robot_state.joint_state = joint_state

        # Call FK service
        future = client._fk_client.call_async(fk_request)
        rclpy.spin_until_future_complete(client, future, timeout_sec=2.0)

        if future.done():
            result = future.result()
            if result.error_code.val == 1:  # SUCCESS
                pose = result.pose_stamped[0].pose

                # Compare with original end-effector pose
                original_pos = point3d['end_effector']['position']
                calculated_pos = [pose.position.x, pose.position.y, pose.position.z]

                pos_diff = [abs(o - c) for o, c in zip(original_pos, calculated_pos)]
                max_pos_error = max(pos_diff)

                # Compare quaternion orientation
                # Convert original orientation to quaternion if needed
                if len(point3d['end_effector']['orientation']) == 3:
                    from scipy.spatial.transform import Rotation
                    original_quat = Rotation.from_euler('xyz', point3d['end_effector']['orientation']).as_quat()
                else:
                    original_quat = point3d['end_effector']['orientation']

                calculated_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

                # Calculate quaternion distance (angular error)
                # Use dot product: angle = 2 * arccos(|q1 · q2|)
                dot_product = sum(o * c for o, c in zip(original_quat, calculated_quat))
                angular_error = 2 * math.acos(min(abs(dot_product), 1.0))  # Clamp to avoid numerical issues
                angular_error_deg = math.degrees(angular_error)

                print(f"  {name}: pos_error={max_pos_error:.6f}m, orient_error={angular_error_deg:.3f}°", end="")
                if max_pos_error < 0.001 and angular_error_deg < 1.0:  # 1mm and 1 degree tolerance
                    print(" ✓")
                else:
                    print(f" ✗")
                    if max_pos_error >= 0.001:
                        print(f"    Position exceeds 1mm tolerance:")
                        print(f"      Original: {[f'{p:.4f}' for p in original_pos]}")
                        print(f"      Calculated: {[f'{p:.4f}' for p in calculated_pos]}")
                    if angular_error_deg >= 1.0:
                        print(f"    Orientation exceeds 1° tolerance:")
                        print(f"      Original quat: {[f'{q:.4f}' for q in original_quat]}")
                        print(f"      Calculated quat: {[f'{q:.4f}' for q in calculated_quat]}")
            else:
                print(f"  {name}: FK failed with error code {result.error_code.val}")
        else:
            print(f"  {name}: FK computation timed out")

    if 1:
        # Convert home position from degrees to radians
        home_joints = home["joints"]
        target_joints = target["joints"]
        print(f"Home position: {[f'{math.degrees(j):.1f}°' for j in home_joints]}")
        print(f'target position: {[f"{math.degrees(j):.1f}°" for j in target_joints]}')

        # Example 1: Move to joint positions with home as start state
        print("\nMoving to zero position from home...")
        success = client.send_joint_goal(target_joints, plan_only=True, start_pos=home_joints)

        if success:
            print("✓ Movement succeeded!")
        else:
            print("✗ Movement failed!")

        # Example 2: Move to a pose with home as start state (uncomment to test)
        print("\nMoving to target pose from home...")
        success = client.send_pose_goal(end_pos=target["end_effector"], plan_only=False, start_pos=home_joints)
        if success:
            print("✓ Movement succeeded!")
        else:
            print("✗ Movement failed!")

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
