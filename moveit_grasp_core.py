#!/usr/bin/env python3
import json
import time
from typing import Sequence, Dict, Any, List, Optional, Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

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
from moveit_msgs.srv import GetCartesianPath, ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from scipy.spatial.transform import Rotation as R
from pointcloud_to_moveit_convexhull import SceneManager

_MOVEIT_ERROR_NAMES = {
    1: "SUCCESS", -1: "FAILURE", -2: "PLANNING_FAILED",
    -3: "INVALID_MOTION_PLAN", -4: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    -5: "CONTROL_FAILED", -6: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    -7: "TIMED_OUT", -10: "PREEMPTED", -11: "START_STATE_IN_COLLISION",
    -12: "START_STATE_VIOLATES_PATH_CONSTRAINTS", -13: "GOAL_IN_COLLISION",
    -14: "GOAL_VIOLATES_PATH_CONSTRAINTS", -15: "GOAL_CONSTRAINTS_VIOLATED",
    -16: "INVALID_GROUP_NAME", -21: "NO_IK_SOLUTION"
}


class GraspExecutor(Node):

    def __init__(self, execution_mode: bool = True, camera_params: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__("grasp_executor_node")

        self.config = config or {}
        self.execution_mode = execution_mode
        self.camera_params = camera_params or self.config.get("camera", {})

        robot_cfg = self.config.get("robot", {})
        self.planning_group = robot_cfg.get("planning_group", "xarm7")
        self.base_frame = robot_cfg.get("base_frame", "link_base")
        self.tcp_link = robot_cfg.get("tcp_link", "link_tcp")
        self.gripper_link = robot_cfg.get("gripper_link", "link_tcp")
        self.arm_joint_names = robot_cfg.get("arm_joint_names", ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"])

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

        ws_limits = self.config.get("workspace_limits", {})
        self.workspace_limits = {
            "x": tuple(ws_limits.get("x", [-0.5, 0.7])),
            "y": tuple(ws_limits.get("y", [-0.8, 0.8])),
            "z": tuple(ws_limits.get("z", [0.0, 0.8]))
        }

        self.current_joint_state = None
        self.attached_objects = {}
        self.scene_manager = None
        self.init_scene_manager()

        self.move_action_client = ActionClient(self, MoveGroup, "move_action")
        self.execute_client = ActionClient(self, ExecuteTrajectory, "execute_trajectory")
        self.gripper_move_client = ActionClient(self, FollowJointTrajectory,"/xarm_gripper_traj_controller/follow_joint_trajectory")
        self.cartesian_path_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.apply_scene_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)

        if not self.move_action_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveGroup action server not responding")
        if self.execution_mode and not self.execute_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn("ExecuteTrajectory action server not responding")
        if self.execution_mode and not self.gripper_move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn("Gripper action server not responding")
        if not self.cartesian_path_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn("Cartesian path service not available")

    def _spin_wait(self, timeout_sec=0.1):
        """等待回调处理（CLI 用 spin_once，HTTP 服务 fallback 到 sleep）"""
        try:
            rclpy.spin_once(self, timeout_sec=timeout_sec)
        except Exception:
            time.sleep(timeout_sec)

    def _send_action_goal(self, action_client, goal,
                          send_timeout=15.0, result_timeout=310.0,
                          label="", log_interval=0.0) -> Optional[Any]:
        """发送 action goal 并等待结果。返回 result wrapper 或 None。"""
        future = action_client.send_goal_async(goal)

        start_time = time.time()
        while not future.done():
            self._spin_wait(0.1)
            if time.time() - start_time > send_timeout:
                if label:
                    self.get_logger().warn(f"{label} Goal send timeout ({send_timeout}s)")
                return None

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            if label:
                self.get_logger().warn(f"{label} Goal not accepted")
            return None

        result_future = goal_handle.get_result_async()
        start_time = time.time()
        last_log = start_time
        while not result_future.done():
            self._spin_wait(0.1)
            if log_interval > 0 and time.time() - last_log > log_interval:
                elapsed = time.time() - start_time
                self.get_logger().info(f"{label} still waiting... ({elapsed:.1f}s)")
                last_log = time.time()
            if time.time() - start_time > result_timeout:
                if label:
                    self.get_logger().warn(f"{label} Result timeout ({result_timeout}s)")
                return None

        return result_future.result()

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

    def add_workspace_walls(self):
        """添加碰撞墙壁，强制规划器不超出工作空间"""
        ws = self.workspace_limits
        x_min, x_max = ws["x"]
        y_min, y_max = ws["y"]
        z_min, z_max = ws["z"]

        t = 0.02
        pad = 1.0  # 覆盖范围 padding，避免边角漏洞

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

        # 透明发布（RViz 不可见，碰撞检测仍生效）
        ps = PlanningScene()
        ps.is_diff = True
        for wall in walls:
            ps.world.collision_objects.append(wall)
            oc = ObjectColor()
            oc.id = wall.id
            oc.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.0)
            ps.object_colors.append(oc)
        self.scene_pub.publish(ps)

        self.get_logger().info(f"[Workspace] Added 6 walls")

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

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects = parts
        self.scene_pub.publish(ps)
        self.get_logger().info(f"Basket '{object_id}' published")
        return True

    def remove_basket_from_scene(self, basket_id: str) -> bool:
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
        self.get_logger().info(f"Removed basket '{basket_id}'")
        return True

    def remove_object_from_scene(self, object_id: str) -> bool:
        co = CollisionObject()
        co.id = object_id
        co.operation = CollisionObject.REMOVE

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        return True

    def clear_octomap(self):
        try:
            from std_srvs.srv import Empty

            clear_cli = self.create_client(Empty, "/clear_octomap")

            if clear_cli.wait_for_service(timeout_sec=2.0):
                future = clear_cli.call_async(Empty.Request())
                start = time.time()
                while not future.done() and time.time() - start < 2.0:
                    self._spin_wait(0.1)
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
            ids_to_remove = [inst["id"] for inst in self.scene_manager.instances]
            if not ids_to_remove:
                return

            for inst_id in ids_to_remove:
                self.scene_manager.disabled_instance_ids.add(inst_id)

            ps = PlanningScene()
            ps.is_diff = True
            for inst_id in ids_to_remove:
                co = CollisionObject()
                co.id = inst_id
                co.header.frame_id = self.scene_manager.frame_id
                co.operation = CollisionObject.REMOVE
                ps.world.collision_objects.append(co)
            self.scene_pub.publish(ps)

            # 清空 disabled 标记，使下次 update_scene 能正常重建凸包
            self.scene_manager.disabled_instance_ids.clear()

            self.get_logger().info(f"Cleared {len(ids_to_remove)} convex hull obstacles")
            time.sleep(0.5)
        except Exception as e:
            self.get_logger().error(f"Failed to clear obstacles: {e}")

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

        intrinsics = self.camera_params.get("intrinsics", self.camera_params)
        depth_scale = self.camera_params.get("depth_scale", 0.001)

        depth_raw = depth_img[v_int, u_int]
        z = depth_raw * depth_scale

        if z <= 0 or not np.isfinite(z):
            self.get_logger().warn(f"Invalid depth at ({u_int}, {v_int}): raw={depth_raw}, z={z}, depth_scale={depth_scale}")
            return None

        pt_cam = np.array([
            (u - intrinsics["cx"]) * z / intrinsics["fx"],
            (v - intrinsics["cy"]) * z / intrinsics["fy"],
            z
        ], dtype=np.float64)
        pt_base = pt_cam @ self.scene_manager.R_cam2base.T + self.scene_manager.t_cam2base
        return tuple(pt_base.astype(float))

    def _compute_width3d(
        self,
        tp1: List[float],
        tp2: List[float],
        center_depth: float
    ) -> float:
        """用接触点反投影到3D空间计算夹爪开度"""
        intrinsics = self.camera_params.get("intrinsics", self.camera_params)
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        p1 = np.array([
            (tp1[0] - cx) * center_depth / fx,
            (tp1[1] - cy) * center_depth / fy,
            center_depth
        ])
        p2 = np.array([
            (tp2[0] - cx) * center_depth / fx,
            (tp2[1] - cy) * center_depth / fy,
            center_depth
        ])
        return float(np.linalg.norm(p1 - p2))

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

        self.get_logger().debug(f"Depth image: shape={depth_img.shape}, dtype={depth_img.dtype}, range=[{depth_img.min()}, {depth_img.max()}]")

        intrinsics = self.camera_params.get("intrinsics", self.camera_params)
        depth_scale = self.camera_params.get("depth_scale", 0.001)

        try:
            with open(seg_json_path, 'r', encoding='utf-8') as f:
                seg_data = json.load(f)
            results = seg_data.get("results", [])
            text_prompt = str(results[0].get("text_prompt", "obj")) if results else "obj"
        except:
            text_prompt = "obj"

        all_candidates = []
        if target_object_index is not None:
            obj_indices = [target_object_index]
        else:
            obj_indices = range(len(affordance_data))

        for obj_idx in obj_indices:
            affordance_obj = affordance_data[obj_idx]
            affs = affordance_obj.get("affs", [])
            scores = affordance_obj.get("scores", [])
            touching_points_list = affordance_obj.get("touching_points", [])
            instance_id = f"{text_prompt}_{obj_idx}"

            for aff_idx, (aff, score) in enumerate(zip(affs, scores)):
                u, v, width_px, height_px = aff[:4]
                has_grasp_pose = len(aff) >= 8
                grasp_pose_rpy = None
                if has_grasp_pose:
                    grasp_pose_rpy = [float(aff[5]), float(aff[6]), float(aff[7])]  # roll, pitch, yaw
                else:
                    angle = float(aff[4])

                center_base = self.transform_to_base(u, v, depth_img)
                if center_base is None:
                    self.get_logger().debug(f"Skipping candidate {obj_idx}-{aff_idx}: transform_to_base returned None")
                    continue

                # 用接触点计算3D夹爪开度（width3d）
                u_int, v_int = int(round(u)), int(round(v))
                center_depth = depth_img[v_int, u_int] * depth_scale

                if aff_idx < len(touching_points_list) and center_depth > 1e-3:
                    tp1, tp2 = touching_points_list[aff_idx]
                    width_m = self._compute_width3d(tp1, tp2, center_depth)
                    self.get_logger().debug(
                        f"Candidate {obj_idx}-{aff_idx}: width3d={width_m:.4f}m "
                        f"(touching_points={tp1}, {tp2}, depth={center_depth:.3f}m)")
                else:
                    width_m = width_px * pixel_to_meter
                    self.get_logger().debug(
                        f"Candidate {obj_idx}-{aff_idx}: fallback width={width_m:.4f}m (no touching_points)")

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

                if has_grasp_pose:
                    candidate["grasp_pose_rpy"] = grasp_pose_rpy
                else:
                    candidate["angle"] = angle
                all_candidates.append(candidate)

        self.get_logger().info(f"Total candidates generated: {len(all_candidates)}")
        return all_candidates


    def set_gripper(self, drive_joint_rad: float) -> bool:
        if not self.execution_mode:
            return True

        drive_joint_rad = float(max(0.0, min(self.gripper_joint_max, drive_joint_rad)))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["drive_joint"]

        p = JointTrajectoryPoint()
        p.positions = [drive_joint_rad]
        p.time_from_start.sec = 1
        goal.trajectory.points = [p]

        result = self._send_action_goal(
            self.gripper_move_client, goal,
            send_timeout=2.0, result_timeout=5.0, label="[Gripper]")

        if result is not None and result.result.error_code == 0:
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
            self.get_logger().info(f"Attached '{instance_id}' to '{self.gripper_link}'")
            time.sleep(0.5)
            return True

        except Exception as e:
            self.get_logger().error(f"Failed to attach mesh: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detach_object(self) -> bool:
        if not self.attached_objects:
            return True

        for object_id in list(self.attached_objects.keys()):
            aco = AttachedCollisionObject()
            aco.object.id = object_id
            aco.object.operation = CollisionObject.REMOVE

            ps = PlanningScene()
            ps.is_diff = True
            ps.robot_state.is_diff = True
            ps.robot_state.attached_collision_objects.append(aco)
            ps.world.collision_objects.append(aco.object)

            if self.apply_scene_client.wait_for_service(timeout_sec=2.0):
                req = ApplyPlanningScene.Request()
                req.scene = ps
                future = self.apply_scene_client.call_async(req)
                deadline = time.monotonic() + 5.0
                while not future.done():
                    if time.monotonic() > deadline:
                        self.get_logger().warn(f"Timeout detaching '{object_id}', falling back to topic")
                        self.scene_pub.publish(ps)
                        break
                    self._spin_wait(0.05)
            else:
                self.scene_pub.publish(ps)

            del self.attached_objects[object_id]
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
        if planner_id:
            req.planner_id = planner_id

        js = JointState()
        js.name = ["drive_joint"]
        js.position = [float(max(0.0, min(self.gripper_joint_max, drive_joint_rad)))]

        if start_joint_state is not None:
            req.start_state.is_diff = False
            req.start_state.joint_state.name = self.arm_joint_names + js.name
            req.start_state.joint_state.position = [float(j) for j in start_joint_state] + list(js.position)
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
            req.start_state.joint_state.name = self.arm_joint_names + js.name
            req.start_state.joint_state.position = [float(j) for j in start_joint_state] + list(js.position)
        else:
            req.start_state.is_diff = True
            req.start_state.joint_state = js

        cons = Constraints()
        for name, pos in zip(self.arm_joint_names, joint_positions):
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
        try:
            jt = trajectory.joint_trajectory
            if not jt.points:
                return None
            n = len(self.arm_joint_names)
            return list(jt.points[-1].positions[:n])
        except Exception as e:
            self.get_logger().error(f"Failed to extract joint state: {e}")
            return None

    def trajectory_cost(
        self,
        obj,
        w_len: float = 30.0,
        w_time: float = 20.0,
        w_j: float = 0.01,
        eps: float = 1e-12
    ) -> Dict[str, Any]:
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

    def plan_cartesian_path(
        self,
        waypoints: List[Pose],
        start_joint_state: Optional[List[float]],
        drive_joint_rad: float = 0.085,
        max_step: float = 0.01,
        jump_threshold: float = 0.0,
        avoid_collisions: bool = True
    ) -> Optional[Tuple[RobotTrajectory, List[float]]]:
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
        req.start_state.joint_state.name = self.arm_joint_names + ["drive_joint"]
        req.start_state.joint_state.position = list(start_joint_state) + [float(drive_joint_rad)]

        try:
            future = self.cartesian_path_client.call_async(req)
            start_time = time.time()
            while not future.done():
                self._spin_wait(0.1)
                if time.time() - start_time > 320.0:  # 5分钟超时
                    self.get_logger().error("Cartesian path service call timeout")
                    return None

            resp = future.result()
            if resp is None:
                self.get_logger().error("Cartesian path service call failed")
                return None
            if float(resp.fraction) < 0.95 or resp.error_code.val != MoveItErrorCodes.SUCCESS:
                return None

            final_state = self.extract_final_joint_state(resp.solution)
            return (resp.solution, final_state) if final_state else None

        except Exception as e:
            self.get_logger().error(f"Cartesian path planning error: {e}")
            return None

    def plan_candidate(
        self,
        aff: Dict[str, Any],
        grasp_pose: Sequence[float],
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

        grasp_depth = self.config.get("grasp", {}).get("grasp_depth_offset", 0.015)
        rot = R.from_euler("xyz", list(adjusted_grasp_pose), degrees=False)
        approach_vec = rot.apply([0.0, 0.0, -1.0])
        center = list(np.array(surface_center) - approach_vec * grasp_depth)

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
        pregrasp_center = np.array(center) + approach_vec * self.pregrasp_offset
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

        if not self.move_action_client.server_is_ready():
            self.get_logger().warn(f"Action server not ready for candidate {aff.get('id', '?')}")
            return None

        label = f"[Candidate {aff.get('id', '?')}]"
        result = self._send_action_goal(
            self.move_action_client, pregrasp_goal,
            send_timeout=15.0, result_timeout=310.0,
            label=label, log_interval=10.0)

        if result is None:
            return None

        error_code = result.result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            name = _MOVEIT_ERROR_NAMES.get(error_code, f"UNKNOWN({error_code})")
            self.get_logger().warn(f"Planning failed for candidate {aff.get('id', '?')}: {name}")
            return None

        result_dict["pregrasp_trajectory"] = result.result.planned_trajectory
        current_state = self.extract_final_joint_state(result.result.planned_trajectory)
        if current_state is None:
            return None

        # 预抓取→抓取：优先笛卡尔规划，失败则 fallback 到关节空间
        grasp_pose_msg = Pose()
        grasp_pose_msg.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        qx, qy, qz, qw = R.from_euler("xyz", adjusted_grasp_pose, degrees=False).as_quat()
        grasp_pose_msg.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
        cartesian_result = self.plan_cartesian_path(
            waypoints=[grasp_pose_msg], start_joint_state=current_state,
            max_step=0.005, avoid_collisions=True)

        if cartesian_result is None:
            grasp_goal = self.build_goal(
                xyz=center, rpy=adjusted_grasp_pose,
                drive_joint_rad=drive_joint_rad,
                allowed_time=300.0, planner_id=planner_id,
                start_joint_state=current_state
            )
            grasp_result = self._send_action_goal(
                self.move_action_client, grasp_goal,
                send_timeout=15.0, result_timeout=310.0,
                label=f"[Candidate {aff.get('id', '?')} grasp]")
            if grasp_result is None or grasp_result.result.error_code.val != MoveItErrorCodes.SUCCESS:
                return None

            result_dict["grasp_trajectory"] = grasp_result.result.planned_trajectory
            result_dict["final_joint_state"] = self.extract_final_joint_state(grasp_result.result.planned_trajectory)
            if result_dict["final_joint_state"] is None:
                return None
        else:
            result_dict["grasp_trajectory"] = cartesian_result[0]
            result_dict["final_joint_state"] = cartesian_result[1]

        return result_dict

    def plan_rank_all_candidates(
        self,
        grasp_candidates: List[Dict[str, Any]],
        grasp_pose: Sequence[float],
        planner_id: str = "RRTConnect",
        start_joint_state: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        results = []
        t0 = time.perf_counter()

        for aff in grasp_candidates:
            t_cand = time.perf_counter()
            r = self.plan_candidate(aff, grasp_pose=grasp_pose,
                planner_id=planner_id, start_joint_state=start_joint_state)
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


    def execute_trajectory(self, traj: RobotTrajectory, timeout_sec: float = 60.0) -> bool:
        """执行轨迹（仅在execution_mode=True时实际执行）"""
        if traj is None:
            return False

        if not self.execution_mode:
            return True

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        result = self._send_action_goal(
            self.execute_client, goal,
            send_timeout=5.0, result_timeout=timeout_sec, label="[Execute]")

        if result is None:
            return False
        return result.result.error_code.val == MoveItErrorCodes.SUCCESS

    def execute_complete_grasp_sequence(
        self,
        best_candidate: Dict[str, Any],
        basket: Dict[str, Any],
        place_clearance: float = 0.05,
        grasp_pose_base: Sequence[float] = (-np.pi, 0.0, 0.0),
        gripper_close_tightness: float = 0.02,
        home_joints: Optional[List[float]] = None,
        end_pos: Optional[List[float]] = None
    ) -> Dict[str, Any]:
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

        object_attached = False

        try:
            # --- 准备阶段 ---
            close_width = max(0.0, best_candidate["gripper_width"] - gripper_close_tightness)
            close_rad = close_width / self.gripper_width_max * self.gripper_joint_max
            result["gripper_commands"]["close_rad"] = close_rad
            result["trajectories"]["approaching"] = best_candidate["pregrasp_trajectory"]
            result["trajectories"]["grasp"] = best_candidate["grasp_trajectory"]

            # 移除抓取目标物体（为接近做准备）
            self.get_logger().info(f"[Scene] Removing object '{instance_id}' from scene for grasping")
            self.remove_object_from_scene(instance_id)
            time.sleep(0.5)

            # --- 执行接近+抓取 ---
            if self.execution_mode:
                self.set_gripper(best_candidate["drive_joint_rad"])
                time.sleep(0.5)

                if not self.execute_trajectory(best_candidate["pregrasp_trajectory"]):
                    self.get_logger().error("[Approaching] Failed")
                    return result

                if not self.execute_trajectory(best_candidate["grasp_trajectory"]):
                    self.get_logger().error("[Grasp] Failed")
                    return result

                self.set_gripper(close_rad)
                time.sleep(0.5)

            # 附着物体到夹爪（影响后续碰撞检测）
            self.get_logger().info(f"[Scene] Attaching object '{instance_id}' to gripper for carrying phase")
            if not self.attach_object_mesh(instance_id):
                self.get_logger().error(f"[Attach] Failed to attach '{instance_id}', aborting sequence")
                return result
            object_attached = True

            # retreat: 回退到预抓取点
            pregrasp_center = best_candidate["pregrasp_center"]
            actual_rpy = best_candidate.get("grasp_pose_rpy") or (
                [grasp_pose_base[0], grasp_pose_base[1], best_candidate["angle"]]
                if "angle" in best_candidate else list(grasp_pose_base)
            )
            grasp_quat = R.from_euler("xyz", actual_rpy, degrees=False).as_quat()
            retreat_pose = Pose()
            retreat_pose.position = Point(x=float(pregrasp_center[0]), y=float(pregrasp_center[1]), z=float(pregrasp_center[2]))
            retreat_pose.orientation = Quaternion(x=float(grasp_quat[0]), y=float(grasp_quat[1]), z=float(grasp_quat[2]), w=float(grasp_quat[3]))

            retreat_result = self.plan_cartesian_path(
                waypoints=[retreat_pose], start_joint_state=current_joint_state,
                max_step=0.005, avoid_collisions=True)
            if retreat_result is None:
                self.get_logger().error("[Retreat] Planning failed")
                return result
            retreat_traj, current_joint_state = retreat_result
            result["trajectories"]["retreat"] = retreat_traj

            # 执行退回到预抓取点
            if not self.execute_trajectory(retreat_traj):
                self.get_logger().error("[Retreat] Execution failed")
                return result

            if self.execution_mode:
                time.sleep(0.3)

            # carry: 搬运到放置点
            place_z = grasp_z + basket["size"][2] + place_clearance
            carry_goal = self.build_goal(
                xyz=[basket["center"][0], basket["center"][1], place_z],
                rpy=(-np.pi, 0.0, 0.0),
                drive_joint_rad=close_rad,
                start_joint_state=current_joint_state,
                planner_id="RRTConnect", allowed_time=300.0,
                pos_tol=0.02, ori_tol=0.2
            )
            result_msg = self._send_action_goal(
                self.move_action_client, carry_goal,
                send_timeout=15.0, result_timeout=310.0, label="[Carrying]")
            if result_msg is None or result_msg.result.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error("[Carrying] Planning failed")
                return result
            carry_traj = result_msg.result.planned_trajectory
            current_joint_state = self.extract_final_joint_state(carry_traj)
            if current_joint_state is None:
                self.get_logger().error("[Carrying] Failed to extract final state")
                return result
            result["trajectories"]["carrying"] = carry_traj

            # 执行搬运轨迹
            if not self.execute_trajectory(carry_traj):
                self.get_logger().error("[Carrying] Execution failed")
                return result

            if self.execution_mode:
                time.sleep(0.5)

            # 释放夹爪
            self.set_gripper(self.gripper_joint_max)
            if self.execution_mode:
                time.sleep(0.5)

            # detach 后规划 return
            self.detach_object()
            object_attached = False
            time.sleep(0.3)

            # return: end_pos(笛卡尔) > home_joints(关节) > default
            if end_pos:
                return_goal = self.build_goal(
                    xyz=end_pos[:3], rpy=end_pos[3:],
                    drive_joint_rad=self.gripper_joint_max,
                    start_joint_state=current_joint_state,
                    planner_id="RRTConnect", allowed_time=300.0)
            else:
                return_goal = self.build_joint_goal(
                    joint_positions=home_joints or [0.0] * 7,
                    drive_joint_rad=self.gripper_joint_max,
                    allowed_time=300.0, planner_id="RRTConnect",
                    start_joint_state=current_joint_state)
            result_msg = self._send_action_goal(
                self.move_action_client, return_goal,
                send_timeout=15.0, result_timeout=310.0, label="[Returning]")
            if result_msg is None or result_msg.result.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error("[Returning] Planning failed")
                return result
            result["trajectories"]["returning"] = result_msg.result.planned_trajectory

            if not self.execute_trajectory(result["trajectories"]["returning"]):
                self.get_logger().error("[Returning] Execution failed")
                return result

            result["success"] = True
            return result

        except Exception as e:
            self.get_logger().error(f"Error in grasp sequence: {e}")
            import traceback
            traceback.print_exc()
            return result

        finally:
            if not self.execution_mode and object_attached:
                self.detach_object()


def _trajectory_to_json(traj, full=False) -> Optional[Dict[str, Any]]:
    """将 RobotTrajectory 转为 JSON。full=True 时包含完整路径点数据。"""
    if traj is None:
        return None
    if not full:
        try:
            pts = traj.joint_trajectory.points
            return {"num_points": len(pts),
                    "duration": pts[-1].time_from_start.sec if pts else 0}
        except:
            return {"info": "trajectory_data"}
    jt = traj.joint_trajectory
    return {
        "joint_names": list(jt.joint_names),
        "points": [{
            "positions": list(p.positions),
            "velocities": list(p.velocities) if p.velocities else [],
            "accelerations": list(p.accelerations) if p.accelerations else [],
            "time_from_start": {"sec": p.time_from_start.sec, "nanosec": p.time_from_start.nanosec}
        } for p in jt.points]
    }

def _build_execution_steps(full_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """构建按执行顺序排列的步骤列表，供客户端回放"""
    steps = []
    trajs = full_result["trajectories"]
    gc = full_result["gripper_commands"]
    instance_id = full_result.get("instance_id")

    if instance_id:
        steps.append({"action": "remove_object", "instance_id": instance_id, "label": "remove_target"})
    steps.append({"action": "set_gripper", "position": gc["open_rad"], "label": "open_gripper"})
    if trajs.get("approaching") is not None:
        steps.append({"action": "execute_trajectory", "trajectory": _trajectory_to_json(trajs["approaching"], full=True), "label": "approaching"})
    if trajs.get("grasp") is not None:
        steps.append({"action": "execute_trajectory", "trajectory": _trajectory_to_json(trajs["grasp"], full=True), "label": "grasp_approach"})
    if gc["close_rad"] is not None:
        steps.append({"action": "set_gripper", "position": gc["close_rad"], "label": "close_gripper"})
    if instance_id:
        steps.append({"action": "attach_object", "instance_id": instance_id, "label": "attach_object"})
    if trajs.get("retreat") is not None:
        steps.append({"action": "execute_trajectory", "trajectory": _trajectory_to_json(trajs["retreat"], full=True), "label": "retreat"})
    if trajs.get("carrying") is not None:
        steps.append({"action": "execute_trajectory", "trajectory": _trajectory_to_json(trajs["carrying"], full=True), "label": "carrying"})
    steps.append({"action": "set_gripper", "position": gc["release_rad"], "label": "release"})
    if instance_id:
        steps.append({"action": "detach_object", "instance_id": instance_id, "label": "detach_object"})
    if trajs.get("returning") is not None:
        steps.append({"action": "execute_trajectory", "trajectory": _trajectory_to_json(trajs["returning"], full=True), "label": "returning"})

    return steps

def interpolate_trajectory(points, dt=0.02):
    """将稀疏 MoveIt 轨迹点用三次 Hermite 样条插值，按固定 dt 重采样。
    Args:
        points: _trajectory_to_json 输出的 points 列表，含 positions/velocities/time_from_start
        dt: 目标时间步长（秒），默认 0.02s = 50Hz
    Returns:
        dict with positions, velocities, accelerations, dt, duration
    """
    from scipy.interpolate import CubicHermiteSpline

    if len(points) < 2:
        return {
            "positions":     [p["positions"] for p in points],
            "velocities":    [p.get("velocities", []) for p in points],
            "accelerations": [p.get("accelerations", []) for p in points],
            "dt": dt}

    # 提取时间轴
    times = np.array([p["time_from_start"]["sec"] + p["time_from_start"]["nanosec"] * 1e-9
                       for p in points])
    n_joints = len(points[0]["positions"])
    pos = np.array([p["positions"] for p in points])
    vel = np.array([p.get("velocities", [0.0] * n_joints) for p in points])

    # 按固定 dt 生成新时间轴
    t_new = np.arange(times[0], times[-1], dt)
    if len(t_new) == 0 or t_new[-1] < times[-1]:
        t_new = np.append(t_new, times[-1])

    # 逐关节插值
    new_pos = np.zeros((len(t_new), n_joints))
    new_vel = np.zeros((len(t_new), n_joints))
    new_acc = np.zeros((len(t_new), n_joints))
    for j in range(n_joints):
        cs = CubicHermiteSpline(times, pos[:, j], vel[:, j])
        new_pos[:, j] = cs(t_new)
        new_vel[:, j] = cs(t_new, 1)
        new_acc[:, j] = cs(t_new, 2)

    return {
        "positions":     new_pos.tolist(),
        "velocities":    new_vel.tolist(),
        "accelerations": new_acc.tolist(),
        "dt": dt,
        "duration": float(times[-1] - times[0])}

def format_grasp_result(results, dt=None):
    """Format _execute_grasp_core output for HTTP API response.
    dt 不为 None 时，对每段轨迹做插值重采样（单位：秒，推荐 0.02 = 50Hz）。
    """
    if not results or not results[0].get("success"):
        return {"success": False, "message": "Grasp planning failed"}
    best = results[0]
    traj = {}
    for s in best.get("execution_steps", []):
        if s.get("action") == "execute_trajectory":
            pts = s["trajectory"].get("points", [])
            if dt and len(pts) >= 2:
                traj[s["label"]] = interpolate_trajectory(pts, dt)
            else:
                traj[s["label"]] = {
                    "positions":     [p["positions"] for p in pts],
                    "velocities":    [p.get("velocities", []) for p in pts],
                    "accelerations": [p.get("accelerations", []) for p in pts]}
    iid = best.get("instance_id", "")
    try:    idx = int(iid.split("_")[-1])
    except: idx = -1
    return {"success": True, "obj_index": idx, "instance_id": iid,
            "trajectory": traj, "planning_time": best.get("planning_time", 0)}

def _execute_grasp_core(
    executor: GraspExecutor,
    depth_path: str,
    seg_json_path: str,
    affordance_path: str,
    config: Dict[str, Any],
    target_object_index: Optional[int] = None,
    return_full_trajectories: bool = False,
    camera: Optional[Dict[str, Any]] = None,
    start_pos: Optional[List[float]] = None,
    end_pos: Optional[List[float]] = None,
    target_pos: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """核心抓取逻辑（可被命令行和HTTP服务复用）
    参数优先级：直接传入 > config 字典 > 默认值
    """
    results = []

    home_config = config.get("home", {})
    basket_cfg = config.get("basket", {})
    grasp_cfg = config.get("grasp", {})

    # camera: 传入 > executor 已有
    if camera:
        executor.camera_params.update(camera)
        executor.config.setdefault("camera", {}).update(camera)

    # basket center: target_pos(框子中心) > config > default
    basket = {
        "id": basket_cfg.get("default_id", "basket_1"),
        "center": target_pos[:3] if target_pos else basket_cfg.get("center", [0.4, 0.3, 0.1]),
        "size": basket_cfg.get("outer_size", [0.3, 0.2, 0.15])
    }

    home_joints = home_config.get("joints", [0.0] * 7)

    try:
        if executor.execution_mode:
            goal = executor.build_joint_goal(
                joint_positions=home_joints,
                drive_joint_rad=executor.gripper_joint_max,
                allowed_time=30.0, planner_id="RRTConnect"
            )
            result_msg = executor._send_action_goal(
                executor.move_action_client, goal,
                send_timeout=30.0, result_timeout=60.0, label="[HOME]")
            if result_msg is None or result_msg.result.error_code.val != MoveItErrorCodes.SUCCESS:
                executor.get_logger().error("[HOME] Planning failed")
                return results
            if not executor.execute_trajectory(result_msg.result.planned_trajectory, timeout_sec=30.0):
                executor.get_logger().error("[HOME] Execution failed")
                return results

        executor.current_joint_state = home_joints

        with open(affordance_path, 'r', encoding='utf-8') as f:
            affordance_data = json.load(f)

        if executor.scene_manager is None:
            raise RuntimeError("Scene manager failed to initialize")
        executor.scene_manager.update_scene(depth_path, seg_json_path)
        time.sleep(1.0)

        executor.add_workspace_walls()
        time.sleep(0.5)

        executor.add_open_top_basket_to_scene(
            object_id=basket["id"],
            center=basket["center"],
            outer_size=basket["size"],
            wall_t=basket_cfg.get("wall_thickness", 0.005),
            bottom_t=basket_cfg.get("bottom_thickness", 0.005)
        )
        time.sleep(0.5)

        all_candidates = executor.parse_affordances_to_candidates(
            affordance_data, depth_path, seg_json_path,
            target_object_index=target_object_index,
            extra_open=grasp_cfg.get("extra_open", 0.020),
            pixel_to_meter=grasp_cfg.get("pixel_to_meter", 0.0001)
        )

        if not all_candidates:
            executor.get_logger().error("No valid grasp candidates")
            return results

        # grasp_pose_base: 传入 > config > default
        grasp_pose_base = tuple(start_pos[3:]) if start_pos \
            else tuple(home_config.get("orientation", (-np.pi, 0.0, 0.0)))
        ranked_results = executor.plan_rank_all_candidates(
            all_candidates,
            grasp_pose=grasp_pose_base,
            planner_id="RRTConnect",
            start_joint_state=executor.current_joint_state
        )

        if not ranked_results:
            executor.get_logger().error("All candidates failed planning")
            return results

        place_clearance = grasp_cfg.get("place_clearance", -0.02)

        # 按 instance_id 去重，保留最优候选
        seen = set()
        unique_candidates = []
        for c in ranked_results:
            if c["instance_id"] not in seen:
                unique_candidates.append(c)
                seen.add(c["instance_id"])

        best = unique_candidates[0]
        t0 = time.perf_counter()
        full_result = executor.execute_complete_grasp_sequence(
            best_candidate=best, basket=basket,
            place_clearance=place_clearance,
            grasp_pose_base=grasp_pose_base,
            home_joints=home_joints,
            end_pos=end_pos
        )
        seq_time = time.perf_counter() - t0

        trajs = full_result["trajectories"]
        result = {
            "instance_id": best["instance_id"],
            "success": full_result["success"],
            "planning_time": seq_time,
            "rank": 1,
            "approaching_trajectory": _trajectory_to_json(trajs.get("approaching")),
            "carrying_trajectory": _trajectory_to_json(trajs.get("carrying")),
            "returning_trajectory": _trajectory_to_json(trajs.get("returning"))
        }
        if return_full_trajectories:
            result["execution_steps"] = _build_execution_steps(full_result)
        results.append(result)

        for rank, candidate in enumerate(unique_candidates[1:], start=2):
            results.append({
                "instance_id": candidate["instance_id"],
                "success": True,
                "planning_time": candidate.get("candidate_planning_time", 0.0),
                "rank": rank,
                "approaching_trajectory": _trajectory_to_json(candidate.get("pregrasp_trajectory"))
            })

        executor.get_logger().info(f"Planning complete: {sum(r['success'] for r in results)}/{len(results)} succeeded")

    finally:
        if not return_full_trajectories:
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

def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="MoveIt Grasp Core")
    parser.add_argument("--robot", default="xarm7", help="Robot name, determines config file (default: xarm7)")
    parser.add_argument("--depth", default="test_data/grasp-wrist-dpt_opt.png", help="Depth image path")
    parser.add_argument("--seg", default="test_data/rgb_detection_wrist.json", help="Segmentation JSON path")
    parser.add_argument("--affordance", default="test_data/affordance.json", help="Affordance JSON path")
    args = parser.parse_args()

    robot_name = args.robot
    depth_path = args.depth
    seg_json_path = args.seg
    affordance_path = args.affordance

    if not rclpy.ok():
        rclpy.init()

    config_path = f"config/{robot_name}_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    executor = GraspExecutor(
        execution_mode=True,
        camera_params=config.get("camera"),
        config=config
    )

    try:
        _execute_grasp_core(
            executor=executor,
            depth_path=depth_path,
            seg_json_path=seg_json_path,
            affordance_path=affordance_path,
            config=config,
        )
    except Exception as e:
        executor.get_logger().error(f"Error: {e}")
        import traceback
        traceback.print_exc()
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

    sys.exit(0)


if __name__ == "__main__":
    main()
