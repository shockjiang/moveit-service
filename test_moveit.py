#!/usr/bin/env python3
import json
import time
from typing import Sequence, Dict, Any, List, Optional, Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Point, Pose,Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    MoveItErrorCodes,
    RobotTrajectory,
    CollisionObject,
    AttachedCollisionObject,
    PlanningScene,
    RobotState
)
from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive, Mesh
from scipy.spatial.transform import Rotation as R
from pointcloud_to_moveit_convexhull import SceneManager


class XArmGraspExecutor(Node):   
    def __init__(self, execution_mode: bool = True,camera_params: Optional[Dict[str, Any]] = None):
        """
        Args:
            execution_mode: True=规划+执行(仿真), False=仅规划(实际调用)
        """
        super().__init__("grasp_executor_node")

        self.execution_mode = execution_mode
        self.use_pregrasp = True
        self.target_object_index = None
        self.camera_params = camera_params

        self.planning_group = "xarm7"
        self.base_frame = "link_base"
        self.tcp_link = "link_tcp"
        self.gripper_link = "link_tcp"

        self.gripper_width_max = 0.085  # 弧度
        self.gripper_joint_max = 0.085  # 修复：添加缺失的属性
        self.gripper_touch_links = [
            "link_tcp",
            "xarm_gripper_base_link",
            "left_outer_knuckle", 
            "left_finger",
            "right_outer_knuckle",
            "right_finger"
        ]

        # 预抓取点参数
        self.pregrasp_offset = 0.10  # 预抓取点距离目标点的距离(m)
        self.pregrasp_approach_axis = np.array([0.0, 0.0, -1.0])  # 接近方向(z轴向下)

        # 末端活动空间限制
        self.workspace_limits = {
            "x": (-0.5, 0.7),  # x轴范围
            "y": (-0.8, 0.8),  # y轴范围
            "z": (0.0, 0.8)    # z轴范围
        }

        # 当前关节状态跟踪（用于连续规划）
        self.current_joint_state = None

        self.attached_objects = {}
        self.object_meshes = {}

        self.scene_manager = None
        self.init_scene_manager()
        
        # Action clients
        self.move_action_client = ActionClient(self, MoveGroup, "move_action")
        self.execute_client = ActionClient(self, ExecuteTrajectory, "execute_trajectory")
        self.gripper_move_client = ActionClient(self, FollowJointTrajectory,"/xarm_gripper_traj_controller/follow_joint_trajectory")
        
        # Service clients
        self.cartesian_path_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        
        # Publisher
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        
        # Wait for action servers
        if not self.move_action_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveGroup action server not responding")
        if self.execution_mode and not self.execute_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn("ExecuteTrajectory action server not responding")
        if self.execution_mode and not self.gripper_move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn("Gripper action server not responding")
        
        # Wait for service
        if not self.cartesian_path_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn("Cartesian path service not available")

    def init_scene_manager(self):
        """初始化场景管理器"""
        try:
            self.scene_manager = SceneManager(
                frame_id="link_base",
                fx=909.6648559570312,
                fy=909.5330200195312,
                cx=636.739013671875,
                cy=376.3500061035156,
                t_cam2base=np.array([0.32508802, 0.02826776, 0.65804681]),
                q_cam2base=np.array([-0.70315987, 0.71022054, -0.02642658, 0.02132171]),
                depth_scale=0.001,
                max_range_m=3.0,
                min_range_m=0.02,
                stride=1,
                score_thresh=0.3,
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize scene manager: {e}")
            self.scene_manager = None

    def transform_to_base(self, u: float, v: float, depth_img: np.ndarray) -> Tuple[float, float, float]:
        """相机坐标系转基座坐标系"""
        h, w = depth_img.shape
        u_int, v_int = int(round(u)), int(round(v))
        
        if not (0 <= u_int < w and 0 <= v_int < h):
            return None
        
        z = depth_img[v_int, u_int] * self.camera_params["depth_scale"]
        
        if z <= 0 or not np.isfinite(z):
            return None
        
        x = (u - self.camera_params["cx"]) * z / self.camera_params["fx"]
        y = (v - self.camera_params["cy"]) * z / self.camera_params["fy"]
        point_cam = [x, y, z]
        pt = np.array(point_cam, dtype=np.float64).reshape(3)
        pt_base = (pt @ self.scene_manager.R_cam2base.T) + self.scene_manager.t_cam2base
        
        return tuple(pt_base.astype(float))

    def parse_affordances_to_candidates(
        self,
        affordance_data: List[Dict[str, Any]],
        depth_path: str,
        seg_json_path: str,
        target_object_index: Optional[int] = None,
        extra_open: float = 0.020,
        pixel_to_meter: float = 0.0001
    ) -> List[Dict[str, Any]]:
        """解析affordance为候选抓取"""
        import cv2
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            return []
        
        try:
            with open(seg_json_path, 'r', encoding='utf-8') as f:
                seg_data = json.load(f)
            results = seg_data.get("results", [])
            text_prompt = str(results[0].get("text_prompt", "obj")) if results else "obj"
        except:
            text_prompt = "obj"
        
        # 判断要处理哪些物品，是否有目标物品
        all_candidates = []
        if target_object_index is not None:
            obj_indices = [target_object_index]
        else:
            obj_indices = range(len(affordance_data))
        
        for obj_idx in obj_indices:
            affordance_obj = affordance_data[obj_idx]
            affs = affordance_obj.get("affs", [])
            scores = affordance_obj.get("scores", [])
            instance_id = f"{text_prompt}_{obj_idx}"
            
            for aff_idx, (aff, score) in enumerate(zip(affs, scores)):
                u, v, width_px, height_px = aff[:4]
                # 判断是否有3d抓取姿态
                has_grasp_pose = len(aff) >= 8
                grasp_pose_rpy = None
                if has_grasp_pose:
                    grasp_pose_rpy = [float(aff[5]), float(aff[6]), float(aff[7])]  # roll, pitch, yaw
                else:
                    angle = float(aff[4]) 

                center_base = self.transform_to_base(u, v, depth_img)                
                width_m = width_px * pixel_to_meter
                height_m = height_px * pixel_to_meter
                
                candidate = {
                    "id": len(all_candidates),
                    "object_index": obj_idx,
                    "aff_index": aff_idx,
                    "instance_id": instance_id,
                    "score": float(score),
                    "center": list(center_base),
                    "boundingbox": {"width": float(width_m),"height": float(height_m),"length": 0.05},
                    "extra_open": float(extra_open),
                    "pixel_coords": [float(u), float(v)]
                }
                
                # 如果包含抓取姿态，添加到candidate中
                if has_grasp_pose:
                    candidate["grasp_pose_rpy"] = grasp_pose_rpy
                else:
                    candidate["angle"] = angle
                all_candidates.append(candidate)
        return all_candidates

    def add_open_top_basket_to_scene(
        self,
        object_id: str,
        center: Tuple[float, float, float],
        outer_size: Tuple[float, float, float],
        wall_t: float = 0.005,
        bottom_t: float = 0.005,
        rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> bool:
        """添加上开口中空的放置框"""
        L, W, H = map(float, outer_size)
        wall_t = float(wall_t)
        bottom_t = float(bottom_t)

        rot = R.from_euler("xyz", list(rpy), degrees=False)
        c = np.array(center, dtype=float)

        def local_to_world(p_local: np.ndarray) -> Tuple[float, float, float]:
            p_world = c + rot.apply(p_local)
            return float(p_world[0]), float(p_world[1]), float(p_world[2])

        # 底板
        bottom_center_local = np.array([0.0, 0.0, -H/2 + bottom_t/2], dtype=float)
        self.add_box_to_scene(object_id=f"{object_id}__bottom",center=local_to_world(bottom_center_local),size=(L, W, bottom_t),rpy=rpy,frame_id=self.base_frame,)

        wall_h = H - bottom_t
        wall_center_z_local = -H/2 + bottom_t + wall_h/2
        # 两个长边墙
        long_wall_size = (L, wall_t, wall_h)
        for sign in (+1.0, -1.0):
            wall_center_local = np.array([0.0, sign*(W/2 - wall_t/2), wall_center_z_local], dtype=float)
            self.add_box_to_scene(object_id=f"{object_id}__wall_y{int(sign)}",center=local_to_world(wall_center_local),size=long_wall_size,rpy=rpy,frame_id=self.base_frame,)

        # 两个短边墙
        short_wall_size = (wall_t, W, wall_h)
        for sign in (+1.0, -1.0):
            wall_center_local = np.array([sign*(L/2 - wall_t/2), 0.0, wall_center_z_local], dtype=float)
            self.add_box_to_scene(object_id=f"{object_id}__wall_x{int(sign)}",center=local_to_world(wall_center_local),size=short_wall_size,rpy=rpy,frame_id=self.base_frame,)
        return True

    def add_box_to_scene(
        self,
        object_id: str,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        frame_id: Optional[str] = None
    ) -> bool:
        """添加box障碍物"""
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, center)
        qx, qy, qz, qw = R.from_euler("xyz", list(rpy), degrees=False).as_quat()
        pose.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))

        co = CollisionObject()
        co.header.frame_id = frame_id or self.base_frame
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = object_id
        co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(size[0]), float(size[1]), float(size[2])]
        co.primitives.append(box)
        co.primitive_poses.append(pose)

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        return True

    def add_workspace_constraint(self) -> bool:
        """添加末端活动空间限制（虚拟边界墙）"""
        x_min, x_max = self.workspace_limits["x"]
        y_min, y_max = self.workspace_limits["y"]
        z_min, z_max = self.workspace_limits["z"]
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        wall_thickness = 0.01
        
        # 添加6面墙壁（作为障碍物的反向边界）
        walls = [
            # X方向的两面墙
            ("workspace_wall_x_min", (x_min - wall_thickness/2, y_center, z_center), 
             (wall_thickness, y_range, z_range)),
            ("workspace_wall_x_max", (x_max + wall_thickness/2, y_center, z_center), 
             (wall_thickness, y_range, z_range)),
            # Y方向的两面墙
            ("workspace_wall_y_min", (x_center, y_min - wall_thickness/2, z_center), 
             (x_range, wall_thickness, z_range)),
            ("workspace_wall_y_max", (x_center, y_max + wall_thickness/2, z_center), 
             (x_range, wall_thickness, z_range)),
            # Z方向的两面墙
            ("workspace_wall_z_min", (x_center, y_center, z_min - wall_thickness/2), 
             (x_range, y_range, wall_thickness)),
            ("workspace_wall_z_max", (x_center, y_center, z_max + wall_thickness/2), 
             (x_range, y_range, wall_thickness)),
        ]
        
        for wall_id, center, size in walls:
            self.add_box_to_scene(wall_id, center, size)
        
        time.sleep(0.3)
        self.get_logger().info(
            f"Added workspace constraints: "
            f"X[{x_min:.2f}, {x_max:.2f}], "
            f"Y[{y_min:.2f}, {y_max:.2f}], "
            f"Z[{z_min:.2f}, {z_max:.2f}]"
        )
        return True

    def remove_workspace_constraint(self) -> bool:
        """移除末端活动空间限制"""
        wall_ids = [
            "workspace_wall_x_min", "workspace_wall_x_max",
            "workspace_wall_y_min", "workspace_wall_y_max",
            "workspace_wall_z_min", "workspace_wall_z_max"
        ]
        for wall_id in wall_ids:
            self.remove_object_from_scene(wall_id)
        time.sleep(0.3)
        self.get_logger().info("Removed workspace constraints")
        return True

    def remove_basket_from_scene(self, basket_id: str) -> bool:
        """移除放置框的所有部分"""
        parts = [
            f"{basket_id}__bottom",
            f"{basket_id}__wall_y1",
            f"{basket_id}__wall_y-1",
            f"{basket_id}__wall_x1",
            f"{basket_id}__wall_x-1"
        ]
        
        for part_id in parts:
            self.remove_object_from_scene(part_id)
        
        time.sleep(0.3)
        self.get_logger().info(f"✓ Removed basket '{basket_id}' (5 parts)")
        return True


    def clear_octomap(self):
        """清空OctoMap"""
        try:
            from std_srvs.srv import Empty
            
            clear_cli = self.create_client(Empty, "/clear_octomap")
            
            if clear_cli.wait_for_service(timeout_sec=2.0):
                future = clear_cli.call_async(Empty.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                self.get_logger().info("OctoMap cleared")
            else:
                self.get_logger().warn("/clear_octomap service not available")
                
            self.destroy_client(clear_cli)
            
        except Exception as e:
            self.get_logger().error(f"Failed to clear OctoMap: {e}")

    def clear_pointcloud_obstacles(self):
        """清除所有点云凸包障碍物"""
        if self.scene_manager is None:
            return
        try:
            for inst in self.scene_manager.instances:
                inst_id = inst["id"]
                self.scene_manager.remove_instance_hull(inst_id)
            
            self.get_logger().info("Cleared all convex hull obstacles")
            time.sleep(0.5)
        except Exception as e:
            self.get_logger().error(f"Failed to clear obstacles: {e}")

    def set_gripper(self, drive_joint_rad: float) -> bool:
        """设置抓夹"""
        drive_joint_rad = float(max(0.0, min(self.gripper_joint_max, drive_joint_rad)))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["drive_joint"]

        p = JointTrajectoryPoint()
        p.positions = [drive_joint_rad]
        p.time_from_start.sec = 1
        goal.trajectory.points = [p]

        future = self.gripper_move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is None or not future.result().accepted:
            self.get_logger().warn("Gripper move goal rejected")
            return False

        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)
        if result_future.result() is not None and result_future.result().result.error_code == 0:
            self.get_logger().info(f"Gripper set to {drive_joint_rad:.3f} rad")
            return True  
        return False

    def attach_object_mesh(self, instance_id: str) -> bool:
        """将物体凸包附着到抓夹"""
        if instance_id not in self.scene_manager.processed_meshes:
            self.get_logger().error(f"No mesh for '{instance_id}'")
            return False
        
        mesh = self.scene_manager.processed_meshes[instance_id]
        
        try:
            co = CollisionObject()
            co.header.frame_id = self.base_frame
            co.header.stamp = self.get_clock().now().to_msg()
            co.id = instance_id
            co.operation = CollisionObject.ADD
            
            co.meshes.append(mesh)
            
            pose = Pose()
            pose.orientation.w = 1.0
            co.mesh_poses.append(pose)
            
            aco = AttachedCollisionObject()
            aco.link_name = self.gripper_link
            aco.object = co
            aco.touch_links = self.gripper_touch_links
            
            ps = PlanningScene()
            ps.is_diff = True
            ps.robot_state.is_diff = True
            ps.robot_state.attached_collision_objects.append(aco)
            self.scene_pub.publish(ps)
            
            self.attached_objects[instance_id] = aco
            
            self.get_logger().info(f"Attached mesh '{instance_id}' to gripper")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to attach mesh: {e}")
            return False

    def detach_object(self) -> bool:
        """从抓夹分离物体"""
        for object_id in list(self.attached_objects.keys()):
            aco = AttachedCollisionObject()
            aco.object.id = object_id
            aco.object.operation = CollisionObject.REMOVE

            ps = PlanningScene()
            ps.is_diff = True
            ps.robot_state.is_diff = True
            ps.robot_state.attached_collision_objects.append(aco)
            ps.world.collision_objects.append(aco.object)
            self.scene_pub.publish(ps)
            del self.attached_objects[object_id]
        return True
    
    def remove_object_from_scene(self, object_id: str) -> bool:
        """从场景移除障碍物"""
        co = CollisionObject()
        co.id = object_id
        co.operation = CollisionObject.REMOVE

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        return True
    
    def build_goal(
        self,
        xyz: Sequence[float],
        grasp_pose: Sequence[float],
        drive_joint_rad: float,
        plan_only: bool = True,
        pos_tol: float = 0.01,
        ori_tol: float = 0.1,
        allowed_time: float = 10.0,
        planner_id: Optional[str] = None,
        start_joint_state: Optional[List[float]] = None  # 新增：起始关节状态
    ) -> MoveGroup.Goal:
        """
        构建MoveGroup目标
        
        Args:
            start_joint_state: 起始关节状态（7个关节），如果为None则使用当前状态
            planner_id: 规划器ID
        """
        x, y, z = map(float, xyz)
        roll, pitch, yaw = map(float, grasp_pose)
        qx, qy, qz, qw = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_quat()

        goal = MoveGroup.Goal()
        req = goal.request

        req.group_name = self.planning_group
        req.allowed_planning_time = float(allowed_time)
        
        # 设置规划器
        if planner_id:
            req.planner_id = planner_id

        # 设置起始状态
        js = JointState()
        js.name = ["drive_joint"]
        js.position = [float(max(0.0, min(self.gripper_joint_max, drive_joint_rad)))]
        
        # 如果提供了起始关节状态，使用它而不是is_diff
        if start_joint_state is not None:
            req.start_state.is_diff = False
            arm_js = JointState()
            arm_js.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
            arm_js.position = [float(j) for j in start_joint_state]
            req.start_state.joint_state.name = arm_js.name + js.name
            req.start_state.joint_state.position = list(arm_js.position) + list(js.position)
        else:
            req.start_state.is_diff = True
            req.start_state.joint_state = js

        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position = Point(x=x, y=y, z=z)
        ps.pose.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))

        pos_c = PositionConstraint()
        pos_c.header = ps.header
        pos_c.link_name = self.tcp_link
        pos_c.weight = 1.0

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(pos_tol), float(pos_tol), float(pos_tol)]

        bv = BoundingVolume()
        bv.primitives = [prim]
        bv.primitive_poses = [ps.pose]
        pos_c.constraint_region = bv

        ori_c = OrientationConstraint()
        ori_c.header = ps.header
        ori_c.link_name = self.tcp_link
        ori_c.orientation = ps.pose.orientation
        ori_c.weight = 1.0
        ori_c.absolute_x_axis_tolerance = float(ori_tol)
        ori_c.absolute_y_axis_tolerance = float(ori_tol)
        ori_c.absolute_z_axis_tolerance = float(ori_tol)

        cons = Constraints()
        cons.position_constraints = [pos_c]
        cons.orientation_constraints = [ori_c]
        req.goal_constraints = [cons]

        goal.planning_options.plan_only = bool(plan_only)
        return goal

    def extract_final_joint_state(self, trajectory: RobotTrajectory) -> Optional[List[float]]:
        """从轨迹中提取最终关节状态"""
        try:
            jt = trajectory.joint_trajectory
            if not jt.points:
                return None
            last_point = jt.points[-1]
            # 返回机械臂的7个关节（不包括gripper的drive_joint）
            return list(last_point.positions[:7])
        except Exception as e:
            self.get_logger().error(f"Failed to extract joint state: {e}")
            return None

    def plan_cartesian_path(
        self,
        waypoints: List[Pose],
        start_joint_state: Optional[List[float]],
        drive_joint_rad: float = 0.085,
        max_step: float = 0.01,
        jump_threshold: float = 0.0,
        avoid_collisions: bool = True
    ) -> Optional[Tuple[RobotTrajectory, List[float]]]:
        """笛卡尔路径规划"""
        if not self.cartesian_path_client.service_is_ready():
            self.get_logger().error("Cartesian path service not available")
            return None
        
        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = self.planning_group
        req.waypoints = waypoints
        req.max_step = float(max_step)
        req.jump_threshold = float(jump_threshold)
        req.avoid_collisions = bool(avoid_collisions)

        req.start_state = RobotState()
        req.start_state.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "drive_joint"]
        req.start_state.joint_state.position = list(start_joint_state) + [float(drive_joint_rad)]

        try:
            future = self.cartesian_path_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            if future.result() is None:
                self.get_logger().error("Cartesian path service call failed")
                return None
            
            fraction = float(future.result().fraction)
            if fraction < 0.95:  # 如果完成度低于95%，认为失败
                # self.get_logger().warn(f"Cartesian path incomplete: {fraction*100:.1f}%")
                return None
            if future.result().error_code.val != MoveItErrorCodes.SUCCESS:
                # self.get_logger().error(f"Cartesian path planning failed: {response.error_code.val}")
                return None
            
            final_state = self.extract_final_joint_state(future.result().solution)
            if final_state is None:
                # self.get_logger().error("Failed to extract final joint state from Cartesian path result")
                return None
            # self.get_logger().info(f"Cartesian path planned: {fraction*100:.1f}% complete")
            return (future.result().solution, final_state)
            
        except Exception as e:
            self.get_logger().error(f"Cartesian path planning error: {e}")
            return None

    def plan_candidate(
        self,
        aff: Dict[str, Any],
        grasp_pose: Sequence[float],
        timeout: float = 15.0,
        use_pregrasp: bool = True,
        planner_id: Optional[str] = None,
        start_joint_state: Optional[List[float]] = None  # 新增：起始关节状态
    ) -> Optional[Dict[str, Any]]:
        """规划单个候选"""
        center = aff["center"]
        gripper_width = float(aff["boundingbox"]["width"]) + float(aff.get("extra_open", 0.0))
        gripper_width = float(max(0.0, min(self.gripper_width_max, gripper_width)))
        drive_joint_rad = gripper_width / self.gripper_width_max * self.gripper_joint_max

        # TODO：将affordance中真实的抓取姿态数据名更新，确定姿态形式
        if "grasp_pose_rpy" in aff:
            adjusted_grasp_pose = aff["grasp_pose_rpy"]
        elif "angle" in aff:
            roll, pitch, _ = grasp_pose
            yaw = float(aff["angle"])
            adjusted_grasp_pose = [roll, pitch, yaw]
        else:
            adjusted_grasp_pose = grasp_pose

        # TODO:check四个id的作用
        result_dict = {
            "id": aff["id"],
            "object_index": aff["object_index"],
            "aff_index": aff["aff_index"],
            "instance_id": aff["instance_id"],
            "score": aff.get("score", 0.0),
            "gripper_width": float(gripper_width),
            "drive_joint_rad": float(drive_joint_rad),
            "center": center,
            "angle": aff.get("angle", 0.0),
            "boundingbox": [
                float(aff["boundingbox"].get("length", 0.05)),
                float(aff["boundingbox"]["width"]),
                float(aff["boundingbox"].get("height", 0.02))
            ]
        }
        current_state = start_joint_state

        
        # 计算预抓取点
        rot = R.from_euler("xyz", list(adjusted_grasp_pose), degrees=False)
        approach_vec = rot.apply(self.pregrasp_approach_axis)
        pregrasp_center = np.array(center) + np.array(approach_vec) * self.pregrasp_offset
        pregrasp_pose = adjusted_grasp_pose
        result_dict["pregrasp_center"] = list(pregrasp_center)
        
        # 第一段：规划到预抓取点
        pregrasp_goal = self.build_goal(xyz=pregrasp_center,grasp_pose=pregrasp_pose,drive_joint_rad=drive_joint_rad,
                                        plan_only=True,allowed_time=10.0,planner_id=planner_id,start_joint_state=current_state)
        
        future = self.move_action_client.send_goal_async(pregrasp_goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.result() or not future.result().accepted:
            return None
        
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if result_future.result().result.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        
        result_dict["pregrasp_trajectory"] = result_future.result().result.planned_trajectory
        current_state = self.extract_final_joint_state(result_future.result().result.planned_trajectory)
        if current_state is None:
            return None

        if use_pregrasp:
            # 第二段：从预抓取点到抓取点（使用笛卡尔规划）
            grasp_pose_msg = Pose()
            grasp_pose_msg.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
            qx, qy, qz, qw = R.from_euler("xyz", adjusted_grasp_pose, degrees=False).as_quat()
            grasp_pose_msg.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
            cartesian_result = self.plan_cartesian_path(waypoints=[grasp_pose_msg],start_joint_state=current_state,max_step=0.005,avoid_collisions=True)
            
            if cartesian_result is None:
                grasp_goal = self.build_goal(xyz=center,grasp_pose=adjusted_grasp_pose,drive_joint_rad=drive_joint_rad,
                                            plan_only=True,allowed_time=10.0,planner_id=planner_id,start_joint_state=current_state)
                
                future2 = self.move_action_client.send_goal_async(grasp_goal)
                rclpy.spin_until_future_complete(self, future2, timeout_sec=timeout)
                if not future2.result() or not future2.result().accepted:
                    return None
                
                result_future2 = future2.result().get_result_async()
                rclpy.spin_until_future_complete(self, result_future2, timeout_sec=timeout)
                if result_future2.result().result.error_code.val != MoveItErrorCodes.SUCCESS:
                    return None
                
                result_dict["grasp_trajectory"] = result_future2.result().result.planned_trajectory
                result_dict["final_joint_state"] = self.extract_final_joint_state(result_future2.result().result.planned_trajectory)
                if result_dict["final_joint_state"] is None:
                    return None
            else:
                grasp_traj, final_state = cartesian_result
                result_dict["grasp_trajectory"] = grasp_traj
                result_dict["final_joint_state"] = final_state
        return result_dict

    def trajectory_cost(
        self,
        obj,
        w_len: float = 30.0,
        w_time: float = 20.0,
        w_j: float = 0.01,
        eps: float = 1e-12
    ) -> Dict[str, Any]:
        """计算轨迹成本"""
        traj = getattr(obj, "planned_trajectory", obj)
        jt = getattr(traj, "joint_trajectory", None)
        pts = getattr(jt, "points", [])

        pos = np.asarray([p.positions for p in pts], dtype=np.float64)
        times = np.asarray([p.time_from_start.sec + 1e-9 * p.time_from_start.nanosec for p in pts],dtype=np.float64)

        L = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        T = float(times[-1] - times[0])

        has_acc = any(getattr(p, "accelerations", None) for p in pts)
        J = 0.0
        if has_acc:
            m = pos.shape[1]
            acc = np.asarray([p.accelerations if p.accelerations else [0.0] * m for p in pts],dtype=np.float64)
            dt = np.diff(times)
            valid = dt > eps
            if np.any(valid):
                da = np.diff(acc, axis=0)
                J = float(np.sum(np.sum(da**2, axis=1)[valid] / dt[valid]))

        cost = float(w_len * L + w_time * T + (w_j * J if has_acc else 0.0))
        return {"cost": cost,
                "metrics": {"ok": True, "path_len": L, "time": T, "jerk": J}}

    def execute_trajectory(self, traj: RobotTrajectory, timeout_sec: float = 60.0) -> bool:
        """执行轨迹（仅在execution_mode=True时实际执行）"""
        if traj is None:
            return False

        if not self.execution_mode:
            self.get_logger().info("[PLAN ONLY] Skipping trajectory execution")
            return True

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        future = self.execute_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.result().accepted:
            return False

        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        if result_future.result() is not None and result_future.result().result.error_code.val == MoveItErrorCodes.SUCCESS:
            return True
        return False

    def plan_rank_all_candidates(
        self,
        grasp_candidates: List[Dict[str, Any]],
        grasp_pose: Sequence[float],
        use_pregrasp: bool = True,
        planner_id: str = "RRTConnect",
        start_joint_state: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """规划并排序所有候选(approaching阶段)"""
        results = []
        t0 = time.perf_counter()
        
        for aff in grasp_candidates:
            r = self.plan_candidate(aff, grasp_pose=grasp_pose,use_pregrasp=use_pregrasp,
                planner_id=planner_id,start_joint_state=start_joint_state)
            if r is not None:
                main_traj = r["pregrasp_trajectory"]
                cost_info = self.trajectory_cost(main_traj)
                r['cost'] = cost_info['cost']
                r['cost_metrics'] = cost_info['metrics']
                results.append(r)
                
        planning_time = time.perf_counter() - t0
        self.get_logger().info(f"[Approaching] Planning complete: {len(results)}/{len(grasp_candidates)} succeeded "
                               f"in {planning_time:.2f}s")
        
        if not results:
            self.get_logger().error("All candidates failed planning")
            return []
        
        results.sort(key=lambda r: r['cost'])
        return results

    def plan_to_pose(
        self,
        xyz: Sequence[float],
        rpy: Sequence[float],
        drive_joint_rad: float,
        start_joint_state: Optional[List[float]],
        planner_id: str = "RRTConnect",
        plan_only: bool = True,
        allowed_time: float = 15.0,
        pos_tol: float = 0.01,
        ori_tol: float = 0.1
    ) -> Optional[Tuple[RobotTrajectory, List[float]]]:
        """
        规划到指定位姿，返回(轨迹, 终点关节状态)
        """
        goal = self.build_goal(
            xyz=xyz,
            grasp_pose=rpy,
            drive_joint_rad=drive_joint_rad,
            plan_only=plan_only,
            pos_tol=pos_tol,
            ori_tol=ori_tol,
            allowed_time=allowed_time,
            planner_id=planner_id,
            start_joint_state=start_joint_state
        )
        
        future = self.move_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=allowed_time + 5.0)
        goal_handle = future.result()
        
        if not goal_handle or not goal_handle.accepted:
            return None
        
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        result = result_future.result()
        
        if result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        
        trajectory = result.result.planned_trajectory
        final_state = self.extract_final_joint_state(trajectory)
        if final_state is None:
            return None
        
        return (trajectory, final_state)

    def execute_complete_grasp_sequence(
        self,
        best_candidate: Dict[str, Any],
        basket: Dict[str, Any],
        gripper_close_tightness: float = 0.02,
        place_clearance: float = 0.05,
        grasp_pose_base: Sequence[float] = (-np.pi, 0.0, 0.0)
    ) -> Dict[str, Any]:
        """执行完整抓取序列并生成轨迹"""
        instance_id = best_candidate["instance_id"]
        grasp_z = best_candidate["center"][2]
        
        result = {
            "success": False,
            "trajectories": {
                "approaching": None,
                "grasp": None,
                "carrying": None,
                "returning": None
            }
        }
        
        current_joint_state = best_candidate.get("final_joint_state")
        try:
            self.remove_object_from_scene(instance_id)
            time.sleep(0.5)
            self.set_gripper(best_candidate["drive_joint_rad"])
            if self.execution_mode:
                time.sleep(0.5)
            
            if self.use_pregrasp and "pregrasp_trajectory" in best_candidate:
                if not self.execute_trajectory(best_candidate["pregrasp_trajectory"]):
                    self.get_logger().error("[Approaching] Failed to reach pregrasp")
                    return result
                result["trajectories"]["approaching"] = best_candidate["pregrasp_trajectory"]
                # self.get_logger().info("[Approaching] ✓ Reached pregrasp")
                
                if self.execution_mode:
                    time.sleep(0.3)
                
                # 到抓取点
                if "grasp_trajectory" in best_candidate:
                    if not self.execute_trajectory(best_candidate["grasp_trajectory"]):
                        self.get_logger().error("[Grasp] Failed to reach grasp")
                        return result
                    result["trajectories"]["grasp"] = best_candidate["grasp_trajectory"]
                    self.get_logger().info("[Grasp] ✓ Reached grasp point")
            else:
                # 直接到抓取点
                if not self.execute_trajectory(best_candidate["planned_trajectory"]):
                    self.get_logger().error("[Approaching] Failed")
                    return result
                result["trajectories"]["approaching"] = best_candidate["planned_trajectory"]
                self.get_logger().info("[Approaching] ✓ Complete")
            
            if self.execution_mode:
                time.sleep(0.5)
            
            # 闭合抓夹
            close_width = max(0.0, best_candidate["gripper_width"] - gripper_close_tightness)
            close_rad = close_width / self.gripper_width_max * self.gripper_joint_max
            self.set_gripper(close_rad)
            if self.execution_mode:
                time.sleep(0.5)
            
            # 附着物体
            self.attach_object_mesh(instance_id)
            time.sleep(0.5)
            
            # ========== Phase 2: Carrying ==========
            self.get_logger().info(f"\n{'='*60}")
            self.get_logger().info(f"[Phase 2] CARRYING to basket")
            self.get_logger().info(f"{'='*60}")
            
            # Step 2a: 笛卡尔返回预抓取点
            if self.use_pregrasp and "pregrasp_center" in best_candidate:
                pregrasp_center = best_candidate["pregrasp_center"]
                
                # 构造预抓取点pose
                retreat_pose = Pose()
                retreat_pose.position.x = float(pregrasp_center[0])
                retreat_pose.position.y = float(pregrasp_center[1])
                retreat_pose.position.z = float(pregrasp_center[2])
                # 保持抓取时的姿态（使用抓取时实际用的姿态）
                actual_grasp_rpy = best_candidate.get("grasp_pose_rpy")
                if actual_grasp_rpy is None and "angle" in best_candidate:
                    actual_grasp_rpy = [grasp_pose_base[0], grasp_pose_base[1], best_candidate["angle"]]
                elif actual_grasp_rpy is None:
                    actual_grasp_rpy = grasp_pose_base
                grasp_quat = R.from_euler("xyz", actual_grasp_rpy, degrees=False).as_quat()
                retreat_pose.orientation = Quaternion(x=float(grasp_quat[0]), y=float(grasp_quat[1]), z=float(grasp_quat[2]), w=float(grasp_quat[3]))
                
                self.get_logger().info("[Carrying] Retreating to pregrasp via Cartesian...")
                retreat_result = self.plan_cartesian_path(
                    waypoints=[retreat_pose],
                    start_joint_state=current_joint_state,
                    max_step=0.005,
                    avoid_collisions=True
                )
                
                if retreat_result is None:
                    self.get_logger().error("[Carrying] Cartesian retreat failed")
                    return result
                
                retreat_traj, current_joint_state = retreat_result
                
                # 执行退回到预抓取点
                if not self.execute_trajectory(retreat_traj):
                    self.get_logger().error("[Carrying] Retreat execution failed")
                    return result
                
                self.get_logger().info("[Carrying] ✓ Retreated to pregrasp")
                if self.execution_mode:
                    time.sleep(0.3)
            
            # Step 2b: 从预抓取点规划到放置点
            place_z = grasp_z + basket["size"][2] + place_clearance
            place_position = [basket["center"][0], basket["center"][1], place_z]
            
            self.get_logger().info(f"[Carrying] Planning to place: [{place_position[0]:.3f}, {place_position[1]:.3f}, {place_z:.3f}]")
            
            carry_result = self.plan_to_pose(
                xyz=place_position,
                rpy=(-np.pi, 0.0, 0.0),
                drive_joint_rad=close_rad,
                start_joint_state=current_joint_state,
                planner_id="RRTstar",
                allowed_time=15.0,
                pos_tol=0.02,
                ori_tol=0.2
            )
            
            if carry_result is None:
                self.get_logger().error("[Carrying] Planning to place failed")
                return result
            
            carry_traj, current_joint_state = carry_result
            result["trajectories"]["carrying"] = carry_traj
            
            # 执行到放置点
            if not self.execute_trajectory(carry_traj):
                self.get_logger().error("[Carrying] Execution failed")
                return result
            
            self.get_logger().info("[Carrying] ✓ Complete")
            if self.execution_mode:
                time.sleep(0.5)
            
            # 松开抓夹
            self.set_gripper(self.gripper_joint_max)
            if self.execution_mode:
                time.sleep(0.5)
            
            # 分离物体
            self.detach_object()
            time.sleep(0.5)
            
            # ========== Phase 3: Returning ==========
            self.get_logger().info(f"\n{'='*60}")
            self.get_logger().info(f"[Phase 3] RETURNING to home")
            self.get_logger().info(f"{'='*60}")
            
            # 规划returning轨迹（从放置点到home）
            return_result = self.plan_to_pose(
                xyz=[0.270, 0.0, 0.307],
                rpy=(-np.pi, 0.0, 0.0),
                drive_joint_rad=self.gripper_joint_max,
                start_joint_state=current_joint_state,
                planner_id="RRTstar",
                allowed_time=10.0
            )
            
            if return_result is None:
                self.get_logger().error("[Returning] Planning failed")
                return result
            
            return_traj, current_joint_state = return_result
            result["trajectories"]["returning"] = return_traj
            
            # 执行returning
            if not self.execute_trajectory(return_traj):
                self.get_logger().error("[Returning] Execution failed")
                return result
            
            self.get_logger().info("[Returning] ✓ Complete")
            
            # 全部成功
            result["success"] = True
            self.get_logger().info(f"\n{'='*60}")
            self.get_logger().info("✓✓✓ ALL PHASES COMPLETED ✓✓✓")
            self.get_logger().info(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            self.get_logger().error(f"Error in grasp sequence: {e}")
            import traceback
            traceback.print_exc()
            return result


def main():
    rclpy.init()
    
    # ========== 配置参数 ==========
    EXECUTION_MODE = True  # True=规划+执行(仿真), False=仅规划
    USE_PREGRASP = True
    PREGRASP_OFFSET = 0.20
    
    # 文件路径
    depth_path = "/home/bb/下载/3DGrasp-BMv1/grasp-wrist-dpt_opt.png"
    seg_json_path = "/home/bb/桌面/rgb检测分割结果wrist"
    affordance_path = "/home/bb/桌面/affordance"
    
    # Home点位姿
    home_pose = ([0.270, 0.0, 0.307], (-np.pi, 0.0, 0.0))
    home_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # 放置框参数
    basket = {}
    basket["id"] = "target_basket"
    basket["center"] = (0.3, -0.6, 0.0)
    basket["size"] = (0.50, 0.30, 0.15)
    
    # 其他参数
    grasp_pose_base = (-np.pi, 0.0, 0.0)
    gripper_close_tightness = 0.02
    pixel_to_meter = 0.0001
    place_clearance = -0.02
    
    with open("camera_config.json", 'r') as f:
        camera_params = json.load(f)
    executor = XArmGraspExecutor(execution_mode=EXECUTION_MODE,camera_params=camera_params)
    executor.pregrasp_offset = PREGRASP_OFFSET
    executor.current_joint_state = home_joints  # 初始化关节状态

    try:
        # ========== Step 1: 移到home点 ==========
        executor.get_logger().info("=== Step 1: Moving to HOME ===")
        if EXECUTION_MODE:
            executor.plan_to_pose(
                xyz=home_pose[0],rpy=home_pose[1],
                drive_joint_rad=executor.gripper_joint_max,
                start_joint_state=None,
                planner_id="RRTConnect",
                plan_only = not EXECUTION_MODE)

        # ========== Step 2: 添加workspace限制 ==========
        executor.get_logger().info("=== Step 2: Adding workspace constraints ===")
        # executor.add_workspace_constraint()

        # ========== Step 3: 加载affordance ==========
        executor.get_logger().info("=== Step 3: Loading affordance data ===")
        try:
            with open(affordance_path, 'r', encoding='utf-8') as f:
                    affordance_data = json.load(f)
        except Exception as e:
            executor.get_logger().error(f"Failed to load affordance data: {e}")
            raise RuntimeError("Failed to load affordance data")

        # ========== Step 4: 场景重建 ==========
        executor.get_logger().info("=== Step 4: Scene reconstruction ===")
        executor.scene_manager.update_scene(depth_path, seg_json_path)
        time.sleep(1.0)  # 等待场景更新稳定

        # ========== Step 5: 添加终点框 ==========
        executor.get_logger().info("=== Step 5: Adding target basket ===")
        executor.add_open_top_basket_to_scene(object_id=basket["id"],center=basket["center"],outer_size=basket["size"],wall_t=0.005,bottom_t=0.005)
        time.sleep(0.5)

        # ========== Step 6: 解析候选 ==========
        executor.get_logger().info(f"=== Step 6: Parsing candidates")
        all_candidates = executor.parse_affordances_to_candidates(affordance_data,depth_path,seg_json_path,target_object_index=executor.target_object_index,extra_open=0.020,pixel_to_meter=pixel_to_meter)
        if not all_candidates:
            raise RuntimeError("No valid grasp candidates")

        # ========== Step 7: Approaching规划排序 ==========
        executor.get_logger().info(f"=== Step 7: Planning approaching phase (pregrasp={USE_PREGRASP}) ===")
        ranked_results = executor.plan_rank_all_candidates(all_candidates,grasp_pose=grasp_pose_base,
            use_pregrasp=USE_PREGRASP,planner_id="RRTConnect",start_joint_state=executor.current_joint_state)
        if not ranked_results:
            raise RuntimeError("All candidates failed planning")
        
        best = ranked_results[0]
        executor.get_logger().info(f"\n=== Best candidate ===\n"
                                   f"  ID: {best['instance_id']}\n"
                                   f"  Cost: {best['cost']:.2f}\n")

        # ========== Step 8: 执行完整序列 ==========
        executor.get_logger().info("=== Step 8: Executing complete grasp sequence ===")
        result = executor.execute_complete_grasp_sequence(best_candidate=best,basket=basket,
            gripper_close_tightness=gripper_close_tightness,place_clearance=place_clearance,grasp_pose_base=grasp_pose_base)
        
        if result["success"]:
            executor.get_logger().info("\n✓✓✓ SUCCESS ✓✓✓")
            for phase, traj in result['trajectories'].items():
                if traj is not None:
                    executor.get_logger().info(f"  {phase}: ✓")
        else:
            executor.get_logger().error("\n✗✗✗ FAILED ✗✗✗")

    except Exception as e:
        executor.get_logger().error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        executor.remove_workspace_constraint()
        executor.clear_octomap()
        executor.clear_pointcloud_obstacles()
        executor.remove_basket_from_scene(basket["id"])
        executor.detach_object()
        
        if executor.scene_manager:
            executor.scene_manager.destroy_node()
        executor.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()