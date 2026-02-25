#!/usr/bin/env python3
import json
import time
import threading
from typing import Sequence, Dict, Any, List, Optional, Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    MoveItErrorCodes,
    RobotTrajectory,
    CollisionObject,
    AttachedCollisionObject,
    PlanningScene,
    RobotState,
    ObjectColor
)
from std_msgs.msg import ColorRGBA
from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive, Mesh
from scipy.spatial.transform import Rotation as R
from pointcloud_to_moveit_convexhull import SceneManager


class XArmGraspExecutor(Node):
    def __init__(self, execution_mode: bool = True, camera_params: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__("grasp_executor_node")

        self.config = config or {}
        self.execution_mode = execution_mode
        self.use_pregrasp = self.config.get("execution", {}).get("use_pregrasp", True)
        self.target_object_index = None
        self.camera_params = camera_params or self.config.get("camera", {})

        robot_cfg = self.config.get("robot", {})
        self.planning_group = robot_cfg.get("planning_group", "xarm7")
        self.base_frame = robot_cfg.get("base_frame", "link_base")
        self.tcp_link = robot_cfg.get("tcp_link", "link_tcp")
        self.gripper_link = robot_cfg.get("gripper_link", "link_tcp")

        gripper_cfg = self.config.get("gripper", {})
        self.gripper_width_max = gripper_cfg.get("width_max", 0.085)
        self.gripper_joint_max = gripper_cfg.get("joint_max", 0.085)
        self.gripper_touch_links = gripper_cfg.get("touch_links", [
            "link_tcp", "xarm_gripper_base_link",
            "left_outer_knuckle", "left_finger",
            "right_outer_knuckle", "right_finger"
        ])

        grasp_cfg = self.config.get("grasp", {})
        self.pregrasp_offset = grasp_cfg.get("pregrasp_offset", 0.10)
        pregrasp_axis = grasp_cfg.get("pregrasp_approach_axis", [0.0, 0.0, -1.0])
        self.pregrasp_approach_axis = np.array(pregrasp_axis)

        ws_limits = self.config.get("workspace_limits", {})
        self.workspace_limits = {
            "x": tuple(ws_limits.get("x", [-0.5, 0.7])),
            "y": tuple(ws_limits.get("y", [-0.8, 0.8])),
            "z": tuple(ws_limits.get("z", [0.0, 0.8]))
        }

        self.current_joint_state = None
        self.attached_objects = {}
        self.object_meshes = {}
        self.scene_manager = None
        self.init_scene_manager()

        self.move_action_client = ActionClient(self, MoveGroup, "move_action")
        self.execute_client = ActionClient(self, ExecuteTrajectory, "execute_trajectory")
        self.gripper_move_client = ActionClient(self, FollowJointTrajectory,"/xarm_gripper_traj_controller/follow_joint_trajectory")
        self.cartesian_path_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
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

    def _spin_wait(self, timeout_sec=0.1):
        """等待回调处理，兼容两种模式：
        - CLI模式：rclpy.spin_once 处理回调
        - HTTP服务模式：节点已被MultiThreadedExecutor管理，spin_once 会抛异常，fallback 到 sleep
        """
        try:
            rclpy.spin_once(self, timeout_sec=timeout_sec)
        except Exception:
            time.sleep(timeout_sec)

    def wait_for_future(self, future, timeout_sec=10.0):
        """等待future完成"""
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < timeout_sec:
            self._spin_wait(0.1)
        return future.done()

    def add_workspace_walls(self):
        """在工作空间边界添加碰撞墙壁，强制规划器不超出范围"""
        ws = self.workspace_limits
        x_min, x_max = ws["x"]
        y_min, y_max = ws["y"]
        z_min, z_max = ws["z"]

        # 墙壁厚度和向外扩展的尺寸
        t = 0.02  # 2cm 厚
        # 每面墙的范围要覆盖得比工作空间大一些，避免边角漏洞
        pad = 1.0

        walls = []

        # +X 墙 (前方)
        walls.append(self._make_box_collision_object(
            "ws_wall_x_pos",
            center=(x_max + t / 2, (y_min + y_max) / 2, (z_min + z_max) / 2),
            size=(t, y_max - y_min + pad, z_max - z_min + pad),
        ))
        # -X 墙 (后方)
        walls.append(self._make_box_collision_object(
            "ws_wall_x_neg",
            center=(x_min - t / 2, (y_min + y_max) / 2, (z_min + z_max) / 2),
            size=(t, y_max - y_min + pad, z_max - z_min + pad),
        ))
        # +Y 墙 (左侧)
        walls.append(self._make_box_collision_object(
            "ws_wall_y_pos",
            center=((x_min + x_max) / 2, y_max + t / 2, (z_min + z_max) / 2),
            size=(x_max - x_min + pad, t, z_max - z_min + pad),
        ))
        # -Y 墙 (右侧)
        walls.append(self._make_box_collision_object(
            "ws_wall_y_neg",
            center=((x_min + x_max) / 2, y_min - t / 2, (z_min + z_max) / 2),
            size=(x_max - x_min + pad, t, z_max - z_min + pad),
        ))
        # +Z 墙 (顶部)
        walls.append(self._make_box_collision_object(
            "ws_wall_z_pos",
            center=((x_min + x_max) / 2, (y_min + y_max) / 2, z_max + t / 2),
            size=(x_max - x_min + pad, y_max - y_min + pad, t),
        ))
        # -Z 墙 (底部/地面)
        walls.append(self._make_box_collision_object(
            "ws_wall_z_neg",
            center=((x_min + x_max) / 2, (y_min + y_max) / 2, z_min - t / 2),
            size=(x_max - x_min + pad, y_max - y_min + pad, t),
        ))

        # 一次性发布所有墙壁（透明，RViz中不可见，但碰撞检测仍生效）
        ps = PlanningScene()
        ps.is_diff = True
        for wall in walls:
            ps.world.collision_objects.append(wall)
            oc = ObjectColor()
            oc.id = wall.id
            oc.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.0)
            ps.object_colors.append(oc)
        self.scene_pub.publish(ps)

        self.get_logger().info(
            f"[Workspace] Added 6 walls: "
            f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]"
        )

    def remove_workspace_walls(self):
        """移除工作空间碰撞墙壁"""
        wall_ids = [
            "ws_wall_x_pos", "ws_wall_x_neg",
            "ws_wall_y_pos", "ws_wall_y_neg",
            "ws_wall_z_pos", "ws_wall_z_neg",
        ]
        ps = PlanningScene()
        ps.is_diff = True
        for wid in wall_ids:
            co = CollisionObject()
            co.id = wid
            co.operation = CollisionObject.REMOVE
            ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        self.get_logger().info("[Workspace] Removed workspace walls")

    def init_scene_manager(self):
        try:
            scene_cfg = self.config.get("scene_manager", {})
            cam_cfg = self.camera_params or {}
            cam_intr = cam_cfg.get("intrinsics", {})
            cam_extr = cam_cfg.get("extrinsics", {})

            self.scene_manager = SceneManager(
                node=self,
                frame_id=scene_cfg.get("frame_id", self.base_frame),
                fx=cam_intr.get("fx", 909.6648559570312),
                fy=cam_intr.get("fy", 909.5330200195312),
                cx=cam_intr.get("cx", 636.739013671875),
                cy=cam_intr.get("cy", 376.3500061035156),
                t_cam2base=np.array(cam_extr.get("translation", [0.32508802, 0.02826776, 0.65804681])),
                q_cam2base=np.array(cam_extr.get("quaternion", [-0.70315987, 0.71022054, -0.02642658, 0.02132171])),
                depth_scale=cam_cfg.get("depth_scale", 0.001),
                max_range_m=cam_cfg.get("max_range_m", 3.0),
                min_range_m=cam_cfg.get("min_range_m", 0.02),
                stride=scene_cfg.get("stride", 1),
                score_thresh=scene_cfg.get("score_thresh", 0.3)
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize scene manager: {e}")
            self.scene_manager = None

    def transform_to_base(self, u: float, v: float, depth_img: np.ndarray) -> Tuple[float, float, float]:
        """相机坐标系转基座坐标系"""
        if self.scene_manager is None:
            self.get_logger().error("Scene manager not initialized")
            return None

        h, w = depth_img.shape
        u_int, v_int = int(round(u)), int(round(v))

        if not (0 <= u_int < w and 0 <= v_int < h):
            self.get_logger().warn(f"Pixel coords out of bounds: u={u_int}, v={v_int}, image size={w}x{h}")
            return None

        # 从嵌套的intrinsics字典中获取参数
        intrinsics = self.camera_params.get("intrinsics", self.camera_params)
        depth_scale = self.camera_params.get("depth_scale", 0.001)

        depth_raw = depth_img[v_int, u_int]
        z = depth_raw * depth_scale

        if z <= 0 or not np.isfinite(z):
            self.get_logger().warn(f"Invalid depth at ({u_int}, {v_int}): raw={depth_raw}, z={z}, depth_scale={depth_scale}")
            return None

        x = (u - intrinsics["cx"]) * z / intrinsics["fx"]
        y = (v - intrinsics["cy"]) * z / intrinsics["fy"]
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
            self.get_logger().error(f"Failed to load depth image from {depth_path}")
            return []

        self.get_logger().info(f"Loaded depth image: shape={depth_img.shape}, dtype={depth_img.dtype}, min={depth_img.min()}, max={depth_img.max()}")

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

        self.get_logger().info(f"Processing {len(obj_indices)} objects from affordance data")

        for obj_idx in obj_indices:
            affordance_obj = affordance_data[obj_idx]
            affs = affordance_obj.get("affs", [])
            scores = affordance_obj.get("scores", [])
            instance_id = f"{text_prompt}_{obj_idx}"

            self.get_logger().info(f"Object {obj_idx}: {len(affs)} affordances found")

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
                if center_base is None:
                    self.get_logger().warn(f"Skipping candidate {obj_idx}-{aff_idx}: transform_to_base returned None for u={u}, v={v}")
                    continue

                self.get_logger().info(f"Candidate {obj_idx}-{aff_idx}: center_base={center_base}")
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

        self.get_logger().info(f"Total candidates generated: {len(all_candidates)}")
        return all_candidates

    def _make_box_collision_object(
        self,
        object_id: str,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        frame_id: Optional[str] = None
    ) -> CollisionObject:
        """构造一个 box CollisionObject（不发布）"""
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
        return co

    def add_open_top_basket_to_scene(
        self,
        object_id: str,
        center: Tuple[float, float, float],
        outer_size: Tuple[float, float, float],
        wall_t: float = 0.005,
        bottom_t: float = 0.005,
        rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> bool:
        """添加上开口中空的放置框（5个零件打包为一次发布，避免丢消息）"""
        L, W, H = map(float, outer_size)
        wall_t = float(wall_t)
        bottom_t = float(bottom_t)

        rot = R.from_euler("xyz", list(rpy), degrees=False)
        c = np.array(center, dtype=float)

        def local_to_world(p_local: np.ndarray) -> Tuple[float, float, float]:
            p_world = c + rot.apply(p_local)
            return float(p_world[0]), float(p_world[1]), float(p_world[2])

        parts = []

        # 底板
        bottom_center_local = np.array([0.0, 0.0, -H/2 + bottom_t/2], dtype=float)
        parts.append(self._make_box_collision_object(
            f"{object_id}__bottom", local_to_world(bottom_center_local),
            (L, W, bottom_t), rpy=rpy, frame_id=self.base_frame))

        wall_h = H - bottom_t
        wall_center_z_local = -H/2 + bottom_t + wall_h/2

        # 两个长边墙
        for sign in (+1.0, -1.0):
            wall_center_local = np.array([0.0, sign*(W/2 - wall_t/2), wall_center_z_local], dtype=float)
            parts.append(self._make_box_collision_object(
                f"{object_id}__wall_y{int(sign)}", local_to_world(wall_center_local),
                (L, wall_t, wall_h), rpy=rpy, frame_id=self.base_frame))

        # 两个短边墙
        for sign in (+1.0, -1.0):
            wall_center_local = np.array([sign*(L/2 - wall_t/2), 0.0, wall_center_z_local], dtype=float)
            parts.append(self._make_box_collision_object(
                f"{object_id}__wall_x{int(sign)}", local_to_world(wall_center_local),
                (wall_t, W, wall_h), rpy=rpy, frame_id=self.base_frame))

        # 一次性发布所有5个零件
        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects = parts
        self.scene_pub.publish(ps)
        self.get_logger().info(f"Basket '{object_id}' published ({len(parts)} parts in 1 msg)")
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
        co = self._make_box_collision_object(object_id, center, size, rpy=rpy, frame_id=frame_id)
        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        return True

    def remove_basket_from_scene(self, basket_id: str) -> bool:
        """移除篮子所有零件（打包为一次发布）"""
        part_ids = [
            f"{basket_id}__bottom",
            f"{basket_id}__wall_y1",
            f"{basket_id}__wall_y-1",
            f"{basket_id}__wall_x1",
            f"{basket_id}__wall_x-1"
        ]

        ps = PlanningScene()
        ps.is_diff = True
        for part_id in part_ids:
            co = CollisionObject()
            co.id = part_id
            co.operation = CollisionObject.REMOVE
            ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)

        time.sleep(0.3)
        self.get_logger().info(f"✓ Removed basket '{basket_id}' ({len(part_ids)} parts in 1 msg)")
        return True


    def clear_octomap(self):
        """清空OctoMap"""
        try:
            from std_srvs.srv import Empty

            clear_cli = self.create_client(Empty, "/clear_octomap")

            if clear_cli.wait_for_service(timeout_sec=2.0):
                future = clear_cli.call_async(Empty.Request())
                self.wait_for_future(future, timeout_sec=2.0)
                self.get_logger().info("OctoMap cleared")
            else:
                self.get_logger().warn("/clear_octomap service not available")

            self.destroy_client(clear_cli)

        except Exception as e:
            self.get_logger().error(f"Failed to clear OctoMap: {e}")

    def clear_pointcloud_obstacles(self):
        """清除所有点云凸包障碍物（通过 topic 发布，兼容 Flask 线程调用）"""
        if self.scene_manager is None:
            return
        try:
            ids_to_remove = [inst["id"] for inst in self.scene_manager.instances]
            if not ids_to_remove:
                return

            # 标记为disabled
            for inst_id in ids_to_remove:
                self.scene_manager.disabled_instance_ids.add(inst_id)

            # 通过 topic 一次性批量移除（与 remove_object_from_scene / remove_basket_from_scene 一致）
            ps = PlanningScene()
            ps.is_diff = True
            for inst_id in ids_to_remove:
                co = CollisionObject()
                co.id = inst_id
                co.header.frame_id = self.scene_manager.frame_id
                co.operation = CollisionObject.REMOVE
                ps.world.collision_objects.append(co)
            self.scene_pub.publish(ps)

            # 清空disabled标记，让下一次 update_scene 可以正常重建凸包
            self.scene_manager.disabled_instance_ids.clear()

            self.get_logger().info(f"Cleared {len(ids_to_remove)} convex hull obstacles (1 msg)")
            time.sleep(0.5)
        except Exception as e:
            self.get_logger().error(f"Failed to clear obstacles: {e}")

    def set_gripper(self, drive_joint_rad: float) -> bool:
        """设置抓夹"""
        # 如果不在执行模式，直接返回成功（只规划不执行）
        if not self.execution_mode:
            self.get_logger().info(f"[Plan-only] Gripper would be set to {drive_joint_rad:.3f} rad")
            return True

        drive_joint_rad = float(max(0.0, min(self.gripper_joint_max, drive_joint_rad)))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["drive_joint"]

        p = JointTrajectoryPoint()
        p.positions = [drive_joint_rad]
        p.time_from_start.sec = 1
        goal.trajectory.points = [p]

        future = self.gripper_move_client.send_goal_async(goal)

        # 使用spin_once等待
        start_time = time.time()
        while not future.done():
            self._spin_wait(0.1)
            if time.time() - start_time > 2.0:
                self.get_logger().warn("Gripper move goal timeout")
                return False

        if future.result() is None or not future.result().accepted:
            self.get_logger().warn("Gripper move goal rejected")
            return False

        result_future = future.result().get_result_async()
        start_time = time.time()
        while not result_future.done():
            self._spin_wait(0.1)
            if time.time() - start_time > 5.0:
                self.get_logger().warn("Gripper execution timeout")
                return False

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

    def reset_planning_scene(self):
        """重置规划场景（用于多次规划测试）"""
        self.get_logger().info("[Scene] Resetting planning scene")

        # 分离所有附着的物体
        self.detach_object()

        # 清空octomap
        self.clear_octomap()

        # 清空点云障碍物
        self.clear_pointcloud_obstacles()

        # 移除工作空间墙壁
        self.remove_workspace_walls()

        self.get_logger().info("[Scene] Planning scene reset complete")

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
        rpy: Sequence[float],
        drive_joint_rad: float,
        pos_tol: float = 0.01,
        ori_tol: float = 0.1,
        allowed_time: float = 10.0,
        planner_id: Optional[str] = None,
        start_joint_state: Optional[List[float]] = None,
        plan_only: bool = True
    ) -> MoveGroup.Goal:
        x, y, z = map(float, xyz)
        roll, pitch, yaw = map(float, rpy)
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

        goal.planning_options.plan_only = plan_only
        return goal

    def build_joint_goal(
        self,
        joint_positions: List[float],
        drive_joint_rad: float,
        tolerance: float = 0.001,
        allowed_time: float = 30.0,
        planner_id: Optional[str] = None,
        start_joint_state: Optional[List[float]] = None,
        plan_only: bool = True
    ) -> MoveGroup.Goal:
        """构建关节空间目标（用于精确到达指定关节角度）"""
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.planning_group
        req.allowed_planning_time = float(allowed_time)

        if planner_id:
            req.planner_id = planner_id

        js = JointState()
        js.name = ["drive_joint"]
        js.position = [float(max(0.0, min(self.gripper_joint_max, drive_joint_rad)))]

        if start_joint_state is not None:
            req.start_state.is_diff = False
            req.start_state.joint_state.name = joint_names + js.name
            req.start_state.joint_state.position = [float(j) for j in start_joint_state] + list(js.position)
        else:
            req.start_state.is_diff = True
            req.start_state.joint_state = js

        cons = Constraints()
        for name, pos in zip(joint_names, joint_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = float(tolerance)
            jc.tolerance_below = float(tolerance)
            jc.weight = 1.0
            cons.joint_constraints.append(jc)
        req.goal_constraints = [cons]

        goal.planning_options.plan_only = plan_only
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
            # 使用spin_once等待结果
            start_time = time.time()
            while not future.done():
                self._spin_wait(0.1)
                if time.time() - start_time > 320.0:  # 5分钟超时
                    self.get_logger().error("Cartesian path service call timeout")
                    return None

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
        use_pregrasp: bool = True,
        planner_id: Optional[str] = None,
        start_joint_state: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        surface_center = aff["center"]
        gripper_width = float(aff["boundingbox"]["width"]) + float(aff.get("extra_open", 0.0))
        gripper_width = float(max(0.0, min(self.gripper_width_max, gripper_width)))
        drive_joint_rad = gripper_width / self.gripper_width_max * self.gripper_joint_max

        if "grasp_pose_rpy" in aff:
            adjusted_grasp_pose = aff["grasp_pose_rpy"]
        elif "angle" in aff:
            roll, pitch, _ = grasp_pose
            yaw = float(aff["angle"])
            adjusted_grasp_pose = [roll, pitch, yaw]
        else:
            adjusted_grasp_pose = grasp_pose

        # 抓取点沿接近方向深入0.015m（检测点在物体表面，实际抓取需更深）
        grasp_depth = self.config.get("grasp", {}).get("grasp_depth_offset", 0.015)
        rot = R.from_euler("xyz", list(adjusted_grasp_pose), degrees=False)
        depth_vec = -rot.apply(self.pregrasp_approach_axis)  # approach_axis指向接近方向，取反即为深入方向
        center = list(np.array(surface_center) + np.array(depth_vec) * grasp_depth)

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

        pregrasp_goal = self.build_goal(
            xyz=pregrasp_center,
            rpy=pregrasp_pose,
            drive_joint_rad=drive_joint_rad,
            allowed_time=300.0,  # 5分钟规划时间
            planner_id=planner_id,
            start_joint_state=current_state
        )

        # 检查action server是否可用
        if not self.move_action_client.server_is_ready():
            self.get_logger().warn(f"Action server not ready for candidate {aff.get('id', '?')}")
            return None

        # 发送goal并使用rclpy的内置等待机制
        future = self.move_action_client.send_goal_async(pregrasp_goal)

        # 在当前线程中处理回调，直到future完成
        start_time = time.time()
        while not future.done():
            self._spin_wait(0.1)
            if time.time() - start_time > 15.0:
                self.get_logger().warn(f"Goal send timeout for candidate {aff.get('id', '?')}")
                return None

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn(f"Goal not accepted for candidate {aff.get('id', '?')}")
            return None

        self.get_logger().info(f"Goal accepted for candidate {aff.get('id', '?')}, waiting for planning...")

        # 等待planning结果（5分钟超时）
        result_future = goal_handle.get_result_async()
        start_time = time.time()
        last_log = start_time
        while not result_future.done():
            self._spin_wait(0.1)
            # 每10秒打印一次进度
            if time.time() - last_log > 10.0:
                elapsed = time.time() - start_time
                self.get_logger().info(f"Candidate {aff.get('id', '?')}: still planning... ({elapsed:.1f}s)")
                last_log = time.time()
            if time.time() - start_time > 310.0:  # 5分钟+10秒缓冲
                self.get_logger().warn(f"Planning timeout (5min) for candidate {aff.get('id', '?')}")
                return None

        result = result_future.result()
        if result is None:
            self.get_logger().warn(f"Planning result is None for candidate {aff.get('id', '?')}")
            return None

        error_code = result.result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            # 记录详细错误信息
            error_code_name = {
                1: "SUCCESS", -1: "FAILURE", -2: "PLANNING_FAILED",
                -3: "INVALID_MOTION_PLAN", -4: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
                -5: "CONTROL_FAILED", -6: "UNABLE_TO_AQUIRE_SENSOR_DATA",
                -7: "TIMED_OUT", -10: "PREEMPTED", -11: "START_STATE_IN_COLLISION",
                -12: "START_STATE_VIOLATES_PATH_CONSTRAINTS", -13: "GOAL_IN_COLLISION",
                -14: "GOAL_VIOLATES_PATH_CONSTRAINTS", -15: "GOAL_CONSTRAINTS_VIOLATED",
                -16: "INVALID_GROUP_NAME", -21: "NO_IK_SOLUTION"
            }.get(error_code, f"UNKNOWN({error_code})")
            self.get_logger().warn(f"Planning failed for candidate {aff.get('id', '?')}: {error_code_name}")
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
                grasp_goal = self.build_goal(
                    xyz=center,
                    rpy=adjusted_grasp_pose,
                    drive_joint_rad=drive_joint_rad,
                    allowed_time=300.0,  # 5分钟规划时间
                    planner_id=planner_id,
                    start_joint_state=current_state
                )

                future2 = self.move_action_client.send_goal_async(grasp_goal)
                self.wait_for_future(future2, timeout_sec=320.0)
                if not future2.result() or not future2.result().accepted:
                    return None

                result_future2 = future2.result().get_result_async()
                self.wait_for_future(result_future2, timeout_sec=320.0)
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
            return True

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        future = self.execute_client.send_goal_async(goal)
        self.wait_for_future(future, timeout_sec=5.0)

        if future.result() is None or not future.result().accepted:
            return False

        result_future = future.result().get_result_async()
        self.wait_for_future(result_future, timeout_sec=timeout_sec)

        if result_future.result() is None:
            return False

        error_code = result_future.result().result.error_code.val
        return error_code == MoveItErrorCodes.SUCCESS

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
            t_cand = time.perf_counter()
            r = self.plan_candidate(aff, grasp_pose=grasp_pose,use_pregrasp=use_pregrasp,
                planner_id=planner_id,start_joint_state=start_joint_state)
            if r is not None:
                r['candidate_planning_time'] = time.perf_counter() - t_cand
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

    def execute_complete_grasp_sequence(
        self,
        best_candidate: Dict[str, Any],
        basket: Dict[str, Any],
        place_clearance: float = 0.05,
        grasp_pose_base: Sequence[float] = (-np.pi, 0.0, 0.0),
        gripper_close_tightness: float = 0.02,
        home_joints: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """执行完整抓取序列并生成轨迹"""
        instance_id = best_candidate["instance_id"]
        grasp_z = best_candidate["center"][2]

        result = {
            "success": False,
            "instance_id": instance_id,
            "trajectories": {
                "approaching": None,
                "grasp": None,
                "retreat": None,
                "carrying": None,
                "returning": None
            },
            "gripper_commands": {
                "open_rad": best_candidate.get("drive_joint_rad", self.gripper_joint_max),
                "close_rad": None,
                "release_rad": self.gripper_joint_max
            }
        }

        current_joint_state = best_candidate.get("final_joint_state")

        # 记录场景状态，用于plan_only模式的回滚
        scene_snapshot = {
            "object_removed": False,
            "object_attached": False
        }

        try:
            # Step 1: 移除抓取目标物体（为接近做准备）
            self.get_logger().info(f"[Scene] Removing object '{instance_id}' from scene for grasping")
            self.remove_object_from_scene(instance_id)
            scene_snapshot["object_removed"] = True
            time.sleep(0.5)

            self.set_gripper(best_candidate["drive_joint_rad"])
            if self.execution_mode:
                time.sleep(0.5)

            if self.use_pregrasp and "pregrasp_trajectory" in best_candidate:
                if not self.execute_trajectory(best_candidate["pregrasp_trajectory"]):
                    self.get_logger().error("[Approaching] Failed")
                    return result

                result["trajectories"]["approaching"] = best_candidate["pregrasp_trajectory"]

                if self.execution_mode:
                    time.sleep(0.3)

                # 到抓取点
                if "grasp_trajectory" in best_candidate:
                    if not self.execute_trajectory(best_candidate["grasp_trajectory"]):
                        self.get_logger().error("[Grasp] Failed")
                        return result
                    result["trajectories"]["grasp"] = best_candidate["grasp_trajectory"]
            else:
                # 直接到抓取点
                if not self.execute_trajectory(best_candidate["planned_trajectory"]):
                    self.get_logger().error("[Approaching] Failed")
                    return result
                result["trajectories"]["approaching"] = best_candidate["planned_trajectory"]

            if self.execution_mode:
                time.sleep(0.5)

            # Step 2: 闭合夹爪（模拟抓取）
            close_width = max(0.0, best_candidate["gripper_width"] - gripper_close_tightness)
            close_rad = close_width / self.gripper_width_max * self.gripper_joint_max
            result["gripper_commands"]["close_rad"] = close_rad
            self.set_gripper(close_rad)
            if self.execution_mode:
                time.sleep(0.5)

            # Step 3: 附着物体到夹爪（重要：这影响后续的碰撞检测）
            self.get_logger().info(f"[Scene] Attaching object '{instance_id}' to gripper for carrying phase")
            self.attach_object_mesh(instance_id)
            scene_snapshot["object_attached"] = True
            time.sleep(0.5)

            if self.use_pregrasp and "pregrasp_center" in best_candidate:
                pregrasp_center = best_candidate["pregrasp_center"]

                retreat_pose = Pose()
                retreat_pose.position.x = float(pregrasp_center[0])
                retreat_pose.position.y = float(pregrasp_center[1])
                retreat_pose.position.z = float(pregrasp_center[2])

                actual_grasp_rpy = best_candidate.get("grasp_pose_rpy")
                if actual_grasp_rpy is None and "angle" in best_candidate:
                    actual_grasp_rpy = [grasp_pose_base[0], grasp_pose_base[1], best_candidate["angle"]]
                elif actual_grasp_rpy is None:
                    actual_grasp_rpy = grasp_pose_base
                grasp_quat = R.from_euler("xyz", actual_grasp_rpy, degrees=False).as_quat()
                retreat_pose.orientation = Quaternion(x=float(grasp_quat[0]), y=float(grasp_quat[1]), z=float(grasp_quat[2]), w=float(grasp_quat[3]))

                retreat_result = self.plan_cartesian_path(
                    waypoints=[retreat_pose],
                    start_joint_state=current_joint_state,
                    max_step=0.005,
                    avoid_collisions=True
                )

                if retreat_result is None:
                    self.get_logger().error("[Carrying] Retreat planning failed")
                    return result

                retreat_traj, current_joint_state = retreat_result
                result["trajectories"]["retreat"] = retreat_traj

                # 执行退回到预抓取点
                if not self.execute_trajectory(retreat_traj):
                    self.get_logger().error("[Carrying] Retreat execution failed")
                    return result

                if self.execution_mode:
                    time.sleep(0.3)

            # 从预抓取点规划到放置点
            place_z = grasp_z + basket["size"][2] + place_clearance
            place_position = [basket["center"][0], basket["center"][1], place_z]

            goal = self.build_goal(
                xyz=place_position,
                rpy=(-np.pi, 0.0, 0.0),
                drive_joint_rad=close_rad,
                start_joint_state=current_joint_state,
                planner_id="RRTConnect",
                allowed_time=300.0,  # 5分钟规划时间
                pos_tol=0.02,
                ori_tol=0.2
            )

            future = self.move_action_client.send_goal_async(goal)
            start_time = time.time()
            while not future.done():
                self._spin_wait(0.1)
                if time.time() - start_time > 320.0:
                    self.get_logger().warn("[Carrying] Goal send timeout")
                    return result

            goal_handle = future.result()

            if not goal_handle or not goal_handle.accepted:
                self.get_logger().error("[Carrying] Planning failed")
                return result

            result_future = goal_handle.get_result_async()
            start_time = time.time()
            while not result_future.done():
                self._spin_wait(0.1)
                if time.time() - start_time > 320.0:
                    self.get_logger().warn("[Carrying] Result timeout")
                    return result

            result_msg = result_future.result()

            if result_msg.result.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error("[Carrying] Planning failed")
                return result

            carry_traj = result_msg.result.planned_trajectory
            current_joint_state = self.extract_final_joint_state(carry_traj)
            if current_joint_state is None:
                self.get_logger().error("[Carrying] Failed to extract final state")
                return result

            result["trajectories"]["carrying"] = carry_traj

            # 执行到放置点
            if not self.execute_trajectory(carry_traj):
                self.get_logger().error("[Carrying] Execution failed")
                return result

            if self.execution_mode:
                time.sleep(0.5)

            self.set_gripper(self.gripper_joint_max)
            if self.execution_mode:
                time.sleep(0.5)

            self.detach_object()
            time.sleep(0.5)

            return_joints = home_joints if home_joints is not None else [0.0] * 7
            goal = self.build_joint_goal(
                joint_positions=return_joints,
                drive_joint_rad=self.gripper_joint_max,
                tolerance=0.001,
                allowed_time=300.0,
                planner_id="RRTConnect",
                start_joint_state=current_joint_state,
                plan_only=True
            )

            future = self.move_action_client.send_goal_async(goal)
            start_time = time.time()
            while not future.done():
                self._spin_wait(0.1)
                if time.time() - start_time > 320.0:
                    self.get_logger().warn("[Returning] Goal send timeout")
                    return result

            goal_handle = future.result()

            if not goal_handle or not goal_handle.accepted:
                self.get_logger().error("[Returning] Planning failed")
                return result

            result_future = goal_handle.get_result_async()
            start_time = time.time()
            while not result_future.done():
                self._spin_wait(0.1)
                if time.time() - start_time > 320.0:
                    self.get_logger().warn("[Returning] Result timeout")
                    return result

            result_msg = result_future.result()

            if result_msg.result.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error("[Returning] Planning failed")
                return result

            return_traj = result_msg.result.planned_trajectory
            result["trajectories"]["returning"] = return_traj

            if not self.execute_trajectory(return_traj):
                self.get_logger().error("[Returning] Execution failed")
                return result

            # 全部成功
            result["success"] = True
            self.get_logger().info("✓ Grasp sequence completed")

            return result

        except Exception as e:
            self.get_logger().error(f"Error in grasp sequence: {e}")
            import traceback
            traceback.print_exc()
            return result

        finally:
            # 清理场景状态（特别是在plan_only模式下）
            if not self.execution_mode:
                self.get_logger().info("[Scene] Cleaning up scene state (plan_only mode)")

                # 如果物体被附着，需要分离
                if scene_snapshot["object_attached"]:
                    self.get_logger().info(f"[Scene] Detaching object '{instance_id}'")
                    self.detach_object()

                # 注意：不恢复被移除的物体，因为在实际执行中它确实会被抓走
                # 如果需要多次规划测试，应该在外层重新加载场景


def _trajectory_to_json_safe(traj) -> Optional[Dict[str, Any]]:
    """将RobotTrajectory转换为JSON安全的格式"""
    if traj is None:
        return None
    try:
        return {
            "num_points": len(traj.joint_trajectory.points),
            "duration": traj.joint_trajectory.points[-1].time_from_start.sec if traj.joint_trajectory.points else 0
        }
    except:
        return {"info": "trajectory_data"}


def _trajectory_to_full_json(traj) -> Optional[Dict[str, Any]]:
    """将RobotTrajectory转换为包含完整路径点数据的JSON格式（供客户端执行）"""
    if traj is None:
        return None
    jt = traj.joint_trajectory
    return {
        "joint_names": list(jt.joint_names),
        "points": [{
            "positions": list(p.positions),
            "velocities": list(p.velocities) if p.velocities else [],
            "accelerations": list(p.accelerations) if p.accelerations else [],
            "time_from_start": {
                "sec": p.time_from_start.sec,
                "nanosec": p.time_from_start.nanosec
            }
        } for p in jt.points]
    }


def _build_execution_steps(full_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """根据完整抓取结果构建按执行顺序排列的步骤列表"""
    steps = []
    trajs = full_result["trajectories"]
    gc = full_result["gripper_commands"]
    instance_id = full_result.get("instance_id")

    # 1. 从场景中移除目标物体（为接近做准备）
    if instance_id:
        steps.append({"action": "remove_object", "instance_id": instance_id, "label": "remove_target"})

    # 2. 打开夹爪
    steps.append({"action": "set_gripper", "position": gc["open_rad"], "label": "open_gripper"})

    # 3. 接近（预抓取点）
    if trajs.get("approaching") is not None:
        steps.append({"action": "execute_trajectory", "trajectory": _trajectory_to_full_json(trajs["approaching"]), "label": "approaching"})

    # 3. 抓取接近（预抓取→抓取点）
    if trajs.get("grasp") is not None:
        steps.append({"action": "execute_trajectory", "trajectory": _trajectory_to_full_json(trajs["grasp"]), "label": "grasp_approach"})

    # 4. 闭合夹爪
    if gc["close_rad"] is not None:
        steps.append({"action": "set_gripper", "position": gc["close_rad"], "label": "close_gripper"})

    # 5. 附着物体到夹爪
    if instance_id:
        steps.append({"action": "attach_object", "instance_id": instance_id, "label": "attach_object"})

    # 6. 退回到预抓取点
    if trajs.get("retreat") is not None:
        steps.append({"action": "execute_trajectory", "trajectory": _trajectory_to_full_json(trajs["retreat"]), "label": "retreat"})

    # 7. 搬运到放置点
    if trajs.get("carrying") is not None:
        steps.append({"action": "execute_trajectory", "trajectory": _trajectory_to_full_json(trajs["carrying"]), "label": "carrying"})

    # 8. 释放
    steps.append({"action": "set_gripper", "position": gc["release_rad"], "label": "release"})

    # 9. 从夹爪分离物体
    if instance_id:
        steps.append({"action": "detach_object", "instance_id": instance_id, "label": "detach_object"})

    # 10. 返回HOME
    if trajs.get("returning") is not None:
        steps.append({"action": "execute_trajectory", "trajectory": _trajectory_to_full_json(trajs["returning"]), "label": "returning"})

    return steps


def _execute_grasp_core(
    executor: XArmGraspExecutor,
    depth_path: str,
    seg_json_path: str,
    affordance_path: str,
    config: Dict[str, Any],
    target_object_index: Optional[int] = None,
    return_full_trajectories: bool = False
) -> List[Dict[str, Any]]:
    """核心抓取逻辑（可被命令行和HTTP服务复用）"""
    results = []

    home_config = config.get("home", {})
    basket_cfg = config.get("basket", {})
    grasp_cfg = config.get("grasp", {})

    basket = {
        "id": basket_cfg.get("default_id", "basket_1"),
        "center": basket_cfg.get("center", [0.4, 0.3, 0.1]),
        "size": basket_cfg.get("outer_size", [0.3, 0.2, 0.15])
    }

    try:
        # Step 1: 移到 home 点（使用关节目标，分两步：先规划，再执行）
        executor.get_logger().info("=== Step 1: Moving to HOME ===")
        if executor.execution_mode:
            home_joints = home_config.get("joints", [0.0] * 7)

            goal = executor.build_joint_goal(
                joint_positions=home_joints,
                drive_joint_rad=executor.gripper_joint_max,
                tolerance=0.001,
                allowed_time=30.0,
                planner_id="RRTConnect",
                start_joint_state=None,
                plan_only=True
            )

            future = executor.move_action_client.send_goal_async(goal)

            start_time = time.time()
            while not future.done():
                executor._spin_wait(0.1)
                if time.time() - start_time > 30.0:
                    executor.get_logger().error("[HOME] Goal send timeout (30s)")
                    return results

            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                executor.get_logger().error("[HOME] Goal not accepted")
                return results

            executor.get_logger().info("[HOME] Goal accepted, waiting for planning...")

            result_future = goal_handle.get_result_async()
            start_time = time.time()
            while not result_future.done():
                executor._spin_wait(0.1)
                if time.time() - start_time > 60.0:
                    executor.get_logger().error("[HOME] Planning timeout (60s)")
                    return results

            result_msg = result_future.result()
            if result_msg is None or result_msg.result.error_code.val != MoveItErrorCodes.SUCCESS:
                error_val = result_msg.result.error_code.val if result_msg else "None"
                executor.get_logger().error(f"[HOME] Planning failed, error_code={error_val}")
                return results

            home_traj = result_msg.result.planned_trajectory
            executor.get_logger().info(
                f"[HOME] Planning OK ({result_msg.result.planning_time:.2f}s, "
                f"{len(home_traj.joint_trajectory.points)} points)")

            # 1b: 执行轨迹
            executor.get_logger().info("[HOME] Executing trajectory...")
            if not executor.execute_trajectory(home_traj, timeout_sec=30.0):
                executor.get_logger().error("[HOME] Execution failed")
                return results

            executor.get_logger().info("[HOME] Done")

        # 设置当前关节状态为HOME位置（无论execution_mode如何）
        executor.current_joint_state = home_config.get("joints", [0.0] * 7)

        # Step 2: 加载 affordance
        executor.get_logger().info("=== Step 2: Loading affordance data ===")
        with open(affordance_path, 'r', encoding='utf-8') as f:
            affordance_data = json.load(f)

        # Step 3: 场景重建
        executor.get_logger().info("=== Step 3: Scene reconstruction ===")
        if executor.scene_manager is None:
            raise RuntimeError("Scene manager failed to initialize")
        executor.scene_manager.update_scene(depth_path, seg_json_path)
        time.sleep(1.0)

        # Step 3.5: 添加工作空间碰撞墙壁
        executor.get_logger().info("=== Step 3.5: Adding workspace walls ===")
        executor.add_workspace_walls()
        time.sleep(0.5)

        # Step 4: 添加终点框
        executor.get_logger().info("=== Step 4: Adding target basket ===")
        executor.add_open_top_basket_to_scene(
            object_id=basket["id"],
            center=basket["center"],
            outer_size=basket["size"],
            wall_t=basket_cfg.get("wall_thickness", 0.005),
            bottom_t=basket_cfg.get("bottom_thickness", 0.005)
        )
        time.sleep(0.5)

        # Step 5: 解析候选
        executor.get_logger().info("=== Step 5: Parsing candidates ===")
        all_candidates = executor.parse_affordances_to_candidates(
            affordance_data, depth_path, seg_json_path,
            target_object_index=target_object_index,
            extra_open=grasp_cfg.get("extra_open", 0.020),
            pixel_to_meter=grasp_cfg.get("pixel_to_meter", 0.0001)
        )

        if not all_candidates:
            executor.get_logger().error("No valid grasp candidates")
            return results

        # Step 6: Approaching 规划排序
        executor.get_logger().info("=== Step 6: Planning approaching phase ===")
        grasp_pose_base = tuple(home_config["orientation"])
        ranked_results = executor.plan_rank_all_candidates(
            all_candidates,
            grasp_pose=grasp_pose_base,
            use_pregrasp=executor.use_pregrasp,
            planner_id="RRTConnect",
            start_joint_state=executor.current_joint_state
        )

        if not ranked_results:
            executor.get_logger().error("All candidates failed planning")
            return results

        # Step 7: 执行抓取序列
        place_clearance = grasp_cfg.get("place_clearance", -0.02)

        if target_object_index is not None:
            # 单个目标物体
            best_candidate = ranked_results[0]
            t0 = time.perf_counter()

            full_result = executor.execute_complete_grasp_sequence(
                best_candidate=best_candidate,
                basket=basket,
                place_clearance=place_clearance,
                grasp_pose_base=grasp_pose_base,
                home_joints=home_config.get("joints", [0.0] * 7)
            )

            planning_time = time.perf_counter() - t0
            result = {
                "instance_id": best_candidate["instance_id"],
                "success": full_result["success"],
                "planning_time": planning_time,
                "approaching_trajectory": _trajectory_to_json_safe(full_result["trajectories"].get("approaching")),
                "carrying_trajectory": _trajectory_to_json_safe(full_result["trajectories"].get("carrying")),
                "returning_trajectory": _trajectory_to_json_safe(full_result["trajectories"].get("returning"))
            }
            if return_full_trajectories:
                result["execution_steps"] = _build_execution_steps(full_result)
            results.append(result)
        else:
            # 多个物体：去重并排序
            # 先按 instance_id 汇总 approaching 阶段的规划时间
            instance_planning_time = {}
            for candidate in ranked_results:
                iid = candidate["instance_id"]
                t = candidate.get("candidate_planning_time", 0.0)
                instance_planning_time[iid] = instance_planning_time.get(iid, 0.0) + t

            seen_instances = set()
            filtered_results = []
            for candidate in ranked_results:
                instance_id = candidate["instance_id"]
                if instance_id not in seen_instances:
                    filtered_results.append(candidate)
                    seen_instances.add(instance_id)

            filtered_results.sort(key=lambda r: r['cost'])

            for rank, candidate in enumerate(filtered_results, start=1):
                iid = candidate["instance_id"]

                if rank == 1:
                    # 只执行排序最好的那个
                    t0 = time.perf_counter()
                    full_result = executor.execute_complete_grasp_sequence(
                        best_candidate=candidate,
                        basket=basket,
                        place_clearance=place_clearance,
                        grasp_pose_base=grasp_pose_base,
                        home_joints=home_config.get("joints", [0.0] * 7)
                    )
                    seq_time = time.perf_counter() - t0

                    # approaching 时间 + 完整序列执行时间
                    planning_time = instance_planning_time.get(iid, 0.0) + seq_time
                    result = {
                        "instance_id": iid,
                        "success": full_result["success"],
                        "planning_time": planning_time,
                        "rank": rank,
                        "approaching_trajectory": _trajectory_to_json_safe(full_result["trajectories"].get("approaching")),
                        "carrying_trajectory": _trajectory_to_json_safe(full_result["trajectories"].get("carrying")),
                        "returning_trajectory": _trajectory_to_json_safe(full_result["trajectories"].get("returning"))
                    }
                    if return_full_trajectories:
                        result["execution_steps"] = _build_execution_steps(full_result)
                else:
                    # 其他候选只记录规划结果（使用 approaching 阶段汇总时间）
                    planning_time = instance_planning_time.get(iid, 0.0)
                    result = {
                        "instance_id": iid,
                        "success": True,
                        "planning_time": planning_time,
                        "rank": rank,
                        "approaching_trajectory": _trajectory_to_json_safe(candidate.get("pregrasp_trajectory"))
                    }

                results.append(result)

        success_count = sum(1 for r in results if r["success"])
        executor.get_logger().info(f"Planning complete: {success_count}/{len(results)} succeeded")

    finally:
        if return_full_trajectories:
            # 客户端模式：场景保留，由客户端执行完毕后调用 /cleanup 清理
            executor.get_logger().info("[Cleanup] Skipped (return_full_trajectories=True, client will cleanup)")
        else:
            # 清理场景（每步之间留间隔，确保 PlanningScene 更新生效）
            try:
                executor.detach_object()
                time.sleep(0.3)
                executor.clear_pointcloud_obstacles()
                executor.remove_basket_from_scene(basket["id"])
                executor.clear_octomap()
                time.sleep(0.5)
                executor.get_logger().info("[Cleanup] Scene cleanup done")
            except Exception as e:
                executor.get_logger().warn(f"Cleanup error: {e}")

    return results


def plan_grasps(
    robot_name: str = "xarm7",
    target_object_index: Optional[int] = None,
    execution_mode: bool = False,
    depth_path: str = "test_data/grasp-wrist-dpt_opt.png",
    seg_json_path: str = "test_data/rgb_detection_wrist.json",
    affordance_path: str = "test_data/affordance.json"
) -> List[Dict[str, Any]]:
    """抓取规划主函数（命令行模式）"""
    # 初始化ROS
    if not rclpy.ok():
        rclpy.init()

    # 加载配置
    config_path = f"config/{robot_name}_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 创建临时 executor
    executor = XArmGraspExecutor(
        execution_mode=execution_mode,
        camera_params=config.get("camera"),
        config=config
    )

    try:
        return _execute_grasp_core(
            executor=executor,
            depth_path=depth_path,
            seg_json_path=seg_json_path,
            affordance_path=affordance_path,
            config=config,
            target_object_index=target_object_index
        )
    except Exception as e:
        executor.get_logger().error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        try:
            executor.destroy_node()
        except Exception as e:
            print(f"Destroy node error: {e}")

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"Shutdown error: {e}")


def main():
    import sys
    plan_grasps(
        robot_name="xarm7",
        target_object_index=None,
        execution_mode=True,
        depth_path="test_data/grasp-wrist-dpt_opt.png",
        seg_json_path="test_data/rgb_detection_wrist.json",
        affordance_path="test_data/affordance.json"
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
