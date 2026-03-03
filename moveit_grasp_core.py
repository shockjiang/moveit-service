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
    1: "SUCCESS", 0: "UNDEFINED", 99999: "FAILURE",
    -1: "PLANNING_FAILED", -2: "INVALID_MOTION_PLAN",
    -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    -4: "CONTROL_FAILED", -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    -6: "TIMED_OUT", -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION", -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION", -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED", -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS", -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME", -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE", -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE", -24: "SENSOR_INFO_STALE",
    -25: "COMMUNICATION_FAILURE", -26: "START_STATE_INVALID",
    -27: "GOAL_STATE_INVALID", -28: "UNRECOGNIZED_GOAL_TYPE",
    -29: "CRASH", -30: "ABORT",
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
        self.get_logger().info("[Init] Initializing scene manager...")
        self.init_scene_manager()
        self.get_logger().info("[Init] Scene manager done, connecting to action servers...")

        self.move_action_client = ActionClient(self, MoveGroup, "move_action")
        self.execute_client = ActionClient(self, ExecuteTrajectory, "execute_trajectory")
        self.gripper_move_client = ActionClient(self, FollowJointTrajectory,"/xarm_gripper_traj_controller/follow_joint_trajectory")
        self.cartesian_path_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.apply_scene_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)

        self.get_logger().info("[Init] Waiting for MoveGroup action server...")
        if not self.move_action_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveGroup action server not responding")
        self.get_logger().info("[Init] MoveGroup ready")

        if self.execution_mode:
            self.get_logger().info("[Init] Waiting for ExecuteTrajectory action server...")
            if not self.execute_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().warn("ExecuteTrajectory action server not responding")
            else:
                self.get_logger().info("[Init] ExecuteTrajectory ready")

            self.get_logger().info("[Init] Waiting for Gripper action server...")
            if not self.gripper_move_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().warn("Gripper action server not responding")
            else:
                self.get_logger().info("[Init] Gripper ready")

        self.get_logger().info("[Init] Waiting for CartesianPath service...")
        if not self.cartesian_path_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn("Cartesian path service not available")
        else:
            self.get_logger().info("[Init] CartesianPath ready")

        self.get_logger().info("[Init] All services connected")

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
        pad = 0.05  # 覆盖范围 padding，避免边角漏洞

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
            oc.color = ColorRGBA(r=55.0, g=1.0, b=1.0, a=0.0)
            ps.object_colors.append(oc)
        self.scene_pub.publish(ps)

        self.get_logger().info(f"[Workspace] Added 6 walls")

    def remove_workspace_walls(self):
        """移除 add_workspace_walls 添加的 6 面碰撞墙壁"""
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
            co.header.frame_id = self.base_frame
            co.operation = CollisionObject.REMOVE
            ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        self.get_logger().info("[Workspace] Removed 6 walls")

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

        if self.apply_scene_client.wait_for_service(timeout_sec=2.0):
            req = ApplyPlanningScene.Request()
            req.scene = ps
            future = self.apply_scene_client.call_async(req)
            deadline = time.monotonic() + 5.0
            while not future.done():
                if time.monotonic() > deadline:
                    self.get_logger().warn(
                        f"Timeout removing '{object_id}', falling back to topic")
                    self.scene_pub.publish(ps)
                    break
                self._spin_wait(0.05)
        else:
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

    def attach_object_mesh(self, instance_id: str,
                           tcp_xyz: Optional[Sequence[float]] = None,
                           tcp_rpy: Optional[Sequence[float]] = None) -> bool:
        """将物体凸包附着到夹爪。
        tcp_xyz/tcp_rpy: 附着时 TCP 在 base_frame 中的位姿。
        提供后会将 mesh 顶点转换到 gripper_link 相对坐标，
        确保后续规划中 mesh 始终正确跟随 gripper。
        """
        if instance_id not in self.scene_manager.processed_meshes:
            self.get_logger().error(f"No mesh for '{instance_id}'")
            return False

        mesh_orig = self.scene_manager.processed_meshes[instance_id]
        self.get_logger().info(
            f"Attaching '{instance_id}' mesh ({len(mesh_orig.vertices)} vertices)")

        try:
            # 将 mesh 顶点从 base_frame 绝对坐标转换为 gripper_link 相对坐标
            # 这样在 build_goal 中用不同的 start_joint_state 时，
            # mesh 始终保持正确的 gripper-relative 偏移，不会出现位姿漂移
            if tcp_xyz is not None and tcp_rpy is not None:
                from copy import deepcopy
                R_grip = R.from_euler("xyz", list(tcp_rpy), degrees=False).as_matrix()
                t_grip = np.array(tcp_xyz, dtype=np.float64)
                mesh = deepcopy(mesh_orig)
                for vertex in mesh.vertices:
                    v_base = np.array([vertex.x, vertex.y, vertex.z], dtype=np.float64)
                    v_grip = R_grip.T @ (v_base - t_grip)
                    vertex.x, vertex.y, vertex.z = float(v_grip[0]), float(v_grip[1]), float(v_grip[2])
                frame_id = self.gripper_link
                self.get_logger().info(
                    f"Transformed mesh to gripper-relative coords (tcp={list(tcp_xyz)})")
            else:
                mesh = mesh_orig
                frame_id = self.base_frame

            co = CollisionObject()
            co.header.frame_id = frame_id
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

            # 使用 service 调用确保 MoveIt 确实应用了 attach
            attached = False
            if self.apply_scene_client.wait_for_service(timeout_sec=2.0):
                req = ApplyPlanningScene.Request()
                req.scene = ps
                future = self.apply_scene_client.call_async(req)
                deadline = time.monotonic() + 5.0
                while not future.done():
                    if time.monotonic() > deadline:
                        self.get_logger().warn(
                            f"Timeout attaching '{instance_id}', falling back to topic")
                        self.scene_pub.publish(ps)
                        attached = True
                        break
                    self._spin_wait(0.05)
                else:
                    resp = future.result()
                    attached = resp.success if resp else False
                    if not attached:
                        self.get_logger().warn(
                            f"ApplyPlanningScene returned failure for attach, "
                            f"falling back to topic")
                        self.scene_pub.publish(ps)
                        attached = True
            else:
                self.get_logger().warn(
                    "/apply_planning_scene not available, using topic")
                self.scene_pub.publish(ps)
                attached = True

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
        plan_only: bool = True,
        num_planning_attempts: int = 1
    ) -> MoveGroup.Goal:
        x, y, z = map(float, xyz)
        roll, pitch, yaw = map(float, rpy)
        qx, qy, qz, qw = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_quat()

        goal = MoveGroup.Goal()
        req = goal.request

        req.group_name = self.planning_group
        req.allowed_planning_time = float(allowed_time)
        req.num_planning_attempts = int(num_planning_attempts)
        if planner_id:
            req.planner_id = planner_id

        js = JointState()
        js.name = ["drive_joint"]
        js.position = [float(max(0.0, min(self.gripper_joint_max, drive_joint_rad)))]

        if start_joint_state is not None:
            req.start_state.is_diff = False
            req.start_state.joint_state.name = self.arm_joint_names + js.name
            req.start_state.joint_state.position = [float(j) for j in start_joint_state] + list(js.position)
            # 将已 attach 的物品写入 start_state，否则规划器不知道附着物
            for aco in self.attached_objects.values():
                req.start_state.attached_collision_objects.append(aco)
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
        plan_only: bool = True,
        num_planning_attempts: int = 1
    ) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.planning_group
        req.allowed_planning_time = float(allowed_time)
        req.num_planning_attempts = int(num_planning_attempts)

        if planner_id:
            req.planner_id = planner_id

        js = JointState()
        js.name = ["drive_joint"]
        js.position = [float(max(0.0, min(self.gripper_joint_max, drive_joint_rad)))]

        if start_joint_state is not None:
            req.start_state.is_diff = False
            req.start_state.joint_state.name = self.arm_joint_names + js.name
            req.start_state.joint_state.position = [float(j) for j in start_joint_state] + list(js.position)
            for aco in self.attached_objects.values():
                req.start_state.attached_collision_objects.append(aco)
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
            
            last_positions = jt.points[-1].positions
            name_to_pos = dict(zip(jt.joint_names, last_positions))
            
            # 按 arm_joint_names 的顺序提取，忽略 drive_joint
            result = []
            for name in self.arm_joint_names:
                if name not in name_to_pos:
                    self.get_logger().error(
                        f"[Extract] joint '{name}' not found in trajectory. "
                        f"Available: {list(jt.joint_names)}")
                    return None
                result.append(name_to_pos[name])
            return result
        except Exception as e:
            self.get_logger().error(f"Failed to extract joint state: {e}")
            return None

    def trajectory_cost(
        self,
        obj,
        w_len: float = 30.0,
        w_time: float = 2.0,
        w_j: float = 1.0,
        eps: float = 1e-12,
        cartesian_dist: float = 0.0
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

        # 路径长度项：有笛卡尔直线距离时用比值（归一化），否则用绝对值
        L_norm = L / max(cartesian_dist, eps) if cartesian_dist > 0 else L
        cost = float(w_len * L_norm + w_time * T + (w_j * J if has_acc else 0.0))
        return {"cost": cost,
                "metrics": {"ok": True, "path_len": L, "time": T, "jerk": J}}

    def _rdp_mask(self, positions: np.ndarray, epsilon: float) -> np.ndarray:
        """Ramer-Douglas-Peucker 关节空间简化，返回保留点的布尔掩码。"""
        n = len(positions)
        if n <= 2:
            return np.ones(n, dtype=bool)

        mask = np.zeros(n, dtype=bool)
        mask[0] = True
        mask[-1] = True

        stack = [(0, n - 1)]
        while stack:
            start, end = stack.pop()
            if end - start < 2:
                continue

            seg = positions[end] - positions[start]
            seg_len_sq = float(np.dot(seg, seg))

            max_dist = 0.0
            max_idx = start
            for i in range(start + 1, end):
                if seg_len_sq < 1e-30:
                    dist = float(np.linalg.norm(positions[i] - positions[start]))
                else:
                    t = float(np.dot(positions[i] - positions[start], seg)) / seg_len_sq
                    t = max(0.0, min(1.0, t))
                    proj = positions[start] + t * seg
                    dist = float(np.linalg.norm(positions[i] - proj))
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

            if max_dist > epsilon:
                mask[max_idx] = True
                stack.append((start, max_idx))
                stack.append((max_idx, end))

        return mask

    def shortcut_trajectory(
        self,
        trajectory: RobotTrajectory,
        epsilon: float = 0.03
    ) -> RobotTrajectory:
        """RDP 简化 + 三次样条重参数化，减少冗余路径点并生成平滑速度。

        Args:
            trajectory: MoveIt 规划器输出的 RobotTrajectory
            epsilon: RDP 容差（弧度），默认 0.03 ≈ 1.7°
        Returns:
            简化后的 RobotTrajectory
        """
        from scipy.interpolate import CubicSpline

        jt = trajectory.joint_trajectory
        pts = jt.points
        joint_names = list(jt.joint_names)

        if len(pts) < 3:
            return trajectory

        positions = np.array([list(p.positions) for p in pts], dtype=np.float64)
        times_orig = np.array(
            [p.time_from_start.sec + 1e-9 * p.time_from_start.nanosec for p in pts],
            dtype=np.float64)

        n_orig = len(positions)

        # RDP 简化
        mask = self._rdp_mask(positions, epsilon)
        pos_simp = positions[mask]
        n_simp = len(pos_simp)

        if n_simp >= n_orig - 1:
            return trajectory

        self.get_logger().info(
            f"[Shortcut] RDP: {n_orig} -> {n_simp} waypoints (epsilon={epsilon:.3f})")

        # 按累积关节空间距离重新分配时间
        seg_lengths = np.linalg.norm(np.diff(pos_simp, axis=0), axis=1)
        cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_dist = cum_dist[-1]

        if total_dist < 1e-12:
            return trajectory

        total_time = times_orig[-1] - times_orig[0]
        times_new = cum_dist / total_dist * total_time

        # 三次样条拟合，零速边界条件
        n_joints = pos_simp.shape[1]
        velocities = np.zeros_like(pos_simp)

        if n_simp >= 3:
            for j in range(n_joints):
                cs = CubicSpline(
                    times_new, pos_simp[:, j],
                    bc_type=((1, 0.0), (1, 0.0)))
                velocities[:, j] = cs(times_new, 1)

        # 构建新轨迹
        new_traj = RobotTrajectory()
        new_traj.joint_trajectory.joint_names = joint_names

        for i in range(n_simp):
            pt = JointTrajectoryPoint()
            pt.positions = list(pos_simp[i])
            pt.velocities = list(velocities[i])
            t = times_new[i]
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            new_traj.joint_trajectory.points.append(pt)

        return new_traj

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
        for aco in self.attached_objects.values():
            req.start_state.attached_collision_objects.append(aco)

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

    def _restore_neighbors(self, neighbor_ids: List[str]):
        """恢复之前因冲突而临时移除的邻居凸包"""
        if not neighbor_ids or self.scene_manager is None:
            return
        meshes_to_restore = []
        for nid in neighbor_ids:
            if nid in self.scene_manager.processed_meshes:
                meshes_to_restore.append((nid, self.scene_manager.processed_meshes[nid]))
        if meshes_to_restore:
            self.scene_manager._apply_collision_meshes(meshes_to_restore)
            self.get_logger().info(f"Restored {len(meshes_to_restore)} neighbor hulls: {neighbor_ids}")
            time.sleep(0.2)

    def _find_conflicting_neighbors(
        self,
        point: Sequence[float],
        exclude_id: str,
        margin: float = 0.02
    ) -> List[str]:
        """找出凸包与给定3D点冲突的邻居物体ID。
        用简单的AABB包围盒+margin快速检测，不需要精确的mesh内点测试。
        """
        if self.scene_manager is None:
            return []
        conflicting = []
        pt = np.array(point, dtype=np.float64)
        for inst_id, mesh in self.scene_manager.processed_meshes.items():
            if inst_id == exclude_id:
                continue
            verts = np.array([[v.x, v.y, v.z] for v in mesh.vertices])
            bbox_min = verts.min(axis=0) - margin
            bbox_max = verts.max(axis=0) + margin
            if np.all(pt >= bbox_min) and np.all(pt <= bbox_max):
                conflicting.append(inst_id)
        return conflicting

    def plan_candidate(
        self,
        aff: Dict[str, Any],
        grasp_pose: Sequence[float],
        planner_id: Optional[str] = None,
        start_joint_state: Optional[List[float]] = None,
        pos_tol: float = 0.01,
        ori_tol: float = 0.1,
        allowed_time: float = 15.0,
        num_planning_attempts: int = 3,
        pregrasp_only: bool = False
    ) -> Optional[Dict[str, Any]]:
        surface_center = aff["center"]
        gripper_width = float(aff["boundingbox"]["width"]) + float(aff.get("extra_open", 0.0))
        gripper_width = float(max(0.0, min(self.gripper_width_max, gripper_width)))
        drive_joint_rad = gripper_width / self.gripper_width_max * self.gripper_joint_max

        if "grasp_pose_rpy" in aff:
            adjusted_grasp_pose = aff["grasp_pose_rpy"]
        elif "angle" in aff:
            roll, pitch, start_yaw = grasp_pose
            angle = float(aff["angle"])
            if angle > 90:
                angle -= 180
            yaw = start_yaw - np.radians(angle)
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
            "adjusted_grasp_pose": list(adjusted_grasp_pose),
            "boundingbox": [
                float(aff["boundingbox"].get("length", 0.05)),
                float(aff["boundingbox"]["width"]),
                float(aff["boundingbox"].get("height", 0.02))
            ]
        }
        # 按来源正确存储方向信息（修复原 "angle": yaw 在 grasp_pose_rpy 分支的 NameError）
        if "grasp_pose_rpy" in aff:
            result_dict["grasp_pose_rpy"] = aff["grasp_pose_rpy"]
        elif "angle" in aff:
            result_dict["angle"] = yaw
        current_state = start_joint_state

        # 计算预抓取点
        pregrasp_center = np.array(center) + approach_vec * self.pregrasp_offset
        pregrasp_pose = adjusted_grasp_pose
        result_dict["pregrasp_center"] = list(pregrasp_center)

        pregrasp_goal = self.build_goal(
            xyz=pregrasp_center,
            rpy=pregrasp_pose,
            drive_joint_rad=drive_joint_rad,
            allowed_time=allowed_time,
            planner_id=planner_id,
            start_joint_state=current_state,
            pos_tol=pos_tol,
            ori_tol=ori_tol,
            num_planning_attempts=num_planning_attempts
        )

        if not self.move_action_client.server_is_ready():
            self.get_logger().warn(f"Action server not ready for candidate {aff.get('id', '?')}")
            return None

        label = f"[Candidate {aff.get('id', '?')}]"
        instance_id = aff.get("instance_id", "")
        removed_neighbors = []  # 记录因冲突而临时移除的邻居

        # ── pregrasp 规划（GOAL_IN_COLLISION 时自动移除冲突邻居重试）──
        result = self._send_action_goal(
            self.move_action_client, pregrasp_goal,
            send_timeout=15.0, result_timeout=allowed_time + 30.0,
            label=label, log_interval=10.0)

        if result is not None and result.result.error_code.val in (
            MoveItErrorCodes.GOAL_IN_COLLISION,
            MoveItErrorCodes.START_STATE_IN_COLLISION
        ):
            # 检测哪些邻居凸包与预抓取点/抓取点冲突
            conflicts = self._find_conflicting_neighbors(
                pregrasp_center, exclude_id=instance_id)
            conflicts += [c for c in self._find_conflicting_neighbors(
                center, exclude_id=instance_id) if c not in conflicts]

            if conflicts:
                self.get_logger().info(
                    f"{label} Goal in collision, removing {len(conflicts)} "
                    f"conflicting neighbors: {conflicts}")
                for cid in conflicts:
                    self.remove_object_from_scene(cid)
                    removed_neighbors.append(cid)
                time.sleep(0.3)

                # 重试规划
                result = self._send_action_goal(
                    self.move_action_client, pregrasp_goal,
                    send_timeout=15.0, result_timeout=allowed_time + 30.0,
                    label=f"{label} retry", log_interval=10.0)

        if result is None:
            self._restore_neighbors(removed_neighbors)
            return None

        error_code = result.result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            name = _MOVEIT_ERROR_NAMES.get(error_code, f"UNKNOWN({error_code})")
            self.get_logger().warn(f"Planning failed for candidate {aff.get('id', '?')}: {name}")
            self._restore_neighbors(removed_neighbors)
            return None

        result_dict["pregrasp_trajectory"] = result.result.planned_trajectory
        result_dict["removed_neighbors"] = removed_neighbors
        current_state = self.extract_final_joint_state(result.result.planned_trajectory)
        if current_state is None:
            self._restore_neighbors(removed_neighbors)
            return None

        if pregrasp_only:
            result_dict["pregrasp_end_state"] = current_state
            return result_dict

        # ── grasp 规划 ──
        grasp_goal = self.build_goal(
            xyz=center, rpy=adjusted_grasp_pose,
            drive_joint_rad=drive_joint_rad,
            allowed_time=allowed_time, planner_id=planner_id,
            start_joint_state=current_state,
            pos_tol=pos_tol, ori_tol=ori_tol,
            num_planning_attempts=num_planning_attempts
        )
        grasp_result = self._send_action_goal(
            self.move_action_client, grasp_goal,
            send_timeout=15.0, result_timeout=allowed_time + 30.0,
            label=f"[Candidate {aff.get('id', '?')} grasp]")
        if grasp_result is None or grasp_result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._restore_neighbors(removed_neighbors)
            return None

        result_dict["grasp_trajectory"] = grasp_result.result.planned_trajectory
        result_dict["final_joint_state"] = self.extract_final_joint_state(grasp_result.result.planned_trajectory)
        if result_dict["final_joint_state"] is None:
            self._restore_neighbors(removed_neighbors)
            return None

        return result_dict

    def _manage_scene_for_candidate(self, iid: str, cur_removed: Optional[str]) -> str:
        """移除当前候选物体、恢复上一个，返回新的 cur_removed"""
        if iid != cur_removed:
            if cur_removed and self.scene_manager and cur_removed in self.scene_manager.processed_meshes:
                self.scene_manager._apply_collision_meshes(
                    [(cur_removed, self.scene_manager.processed_meshes[cur_removed])])
            if iid:
                self.remove_object_from_scene(iid)
                time.sleep(0.2)
        return iid

    def _restore_last_removed(self, cur_removed: Optional[str]):
        if cur_removed and self.scene_manager and cur_removed in self.scene_manager.processed_meshes:
            self.scene_manager._apply_collision_meshes(
                [(cur_removed, self.scene_manager.processed_meshes[cur_removed])])

    def plan_rank_all_candidates(
        self,
        grasp_candidates: List[Dict[str, Any]],
        grasp_pose: Sequence[float],
        planner_id: str = "RRTConnect",
        start_joint_state: Optional[List[float]] = None,
        start_xyz: Optional[Sequence[float]] = None,
        pos_tol: float = 0.01,
        ori_tol: float = 0.1,
        fast_filter_time: float = 5.0,
        refine_time: float = 15.0,
        top_n: int = 1,
        score_weight: float = 0.3,
        basket: Optional[Dict[str, Any]] = None,
        place_clearance: float = 0.05,
        carry_filter_k: int = 3,
        carry_filter_time: float = 3.0,
        gripper_close_tightness: float = 0.02
    ) -> List[Dict[str, Any]]:
        """三阶段规划：
        Phase 1   - RRTConnect 只规划 pregrasp，快速筛选全部候选
        Phase 1.5 - 对 top-K 检查 carry 可行性（需 basket 参数）
        Phase 2   - 对 carry 可行的候选补充 grasp 规划 + RDP shortcutting
        """
        t0 = time.perf_counter()

        # ── Phase 1: pregrasp 可达性快速筛选（跳过 grasp 段）──
        self.get_logger().info(
            f"[Phase1] Filtering {len(grasp_candidates)} candidates "
            f"(pregrasp only, RRTConnect {fast_filter_time}s each)")
        feasible = []
        cur_removed = None

        for aff in grasp_candidates:
            iid = aff.get("instance_id", "")
            cur_removed = self._manage_scene_for_candidate(iid, cur_removed)

            t_cand = time.perf_counter()
            r = self.plan_candidate(
                aff, grasp_pose=grasp_pose,
                planner_id="RRTConnect",
                start_joint_state=start_joint_state,
                pos_tol=pos_tol, ori_tol=ori_tol,
                allowed_time=fast_filter_time,
                num_planning_attempts=1,
                pregrasp_only=True
            )
            if r is not None:
                self._restore_neighbors(r.get("removed_neighbors", []))
                r['candidate_planning_time'] = time.perf_counter() - t_cand
                main_traj = r["pregrasp_trajectory"]
                if start_xyz is not None and "pregrasp_center" in r:
                    straight = float(np.linalg.norm(
                        np.array(r["pregrasp_center"]) - np.array(start_xyz)))
                else:
                    straight = 0.0
                cost_info = self.trajectory_cost(main_traj, cartesian_dist=straight)
                r['cost'] = cost_info['cost']
                r['cost_metrics'] = cost_info['metrics']
                r['_aff'] = aff
                feasible.append(r)

        self._restore_last_removed(cur_removed)

        phase1_time = time.perf_counter() - t0
        self.get_logger().info(
            f"[Phase1] {len(feasible)}/{len(grasp_candidates)} feasible in {phase1_time:.2f}s")

        if not feasible:
            self.get_logger().error("All candidates failed planning")
            return []

        def _rank_key(r):
            return r['cost'] - score_weight * r.get('score', 0.0)

        feasible.sort(key=_rank_key)

        # ── Phase 1.5: carry 可行性快筛（top-K）──
        if basket is not None:
            basket_rim_z = basket["center"][2] + basket["size"][2] / 2
            place_z = basket_rim_z + place_clearance
            carry_xyz = [float(basket["center"][0]),
                         float(basket["center"][1]),
                         float(place_z)]

            k = min(carry_filter_k, len(feasible))
            carry_candidates = feasible[:k]
            self.get_logger().info(
                f"[Phase1.5] Carry feasibility check for top-{k} candidates "
                f"(target={carry_xyz}, timeout={carry_filter_time}s)")

            carry_feasible = []
            for fast_r in carry_candidates:
                cand_id = fast_r.get("id", "?")
                instance_id = fast_r.get("instance_id", "")
                pregrasp_end_state = fast_r.get("pregrasp_end_state")
                if pregrasp_end_state is None:
                    self.get_logger().warn(
                        f"[Phase1.5] No pregrasp_end_state for {cand_id}, skip carry check")
                    carry_feasible.append(fast_r)  # 不确定则保留
                    continue

                # 计算 close_rad 和 carry_rpy
                close_width = max(0.0, fast_r["gripper_width"] - gripper_close_tightness)
                close_rad = close_width / self.gripper_width_max * self.gripper_joint_max
                grasp_yaw = float(fast_r["adjusted_grasp_pose"][2])
                carry_rpy = (-np.pi, 0.0, grasp_yaw)

                # attach 物体到夹爪（用于碰撞检测）
                attach_ok = self.attach_object_mesh(
                    instance_id,
                    tcp_xyz=fast_r["center"],
                    tcp_rpy=fast_r["adjusted_grasp_pose"])
                if attach_ok:
                    self._spin_wait(0.5)

                # 短超时 carry 规划
                carry_goal = self.build_goal(
                    xyz=carry_xyz, rpy=carry_rpy,
                    drive_joint_rad=close_rad,
                    start_joint_state=pregrasp_end_state,
                    planner_id="RRTConnect",
                    allowed_time=carry_filter_time,
                    num_planning_attempts=2,
                    pos_tol=0.05, ori_tol=0.3)
                carry_result = self._send_action_goal(
                    self.move_action_client, carry_goal,
                    send_timeout=10.0,
                    result_timeout=carry_filter_time + 10.0,
                    label=f"[Phase1.5 carry {cand_id}]")

                # detach 并恢复物体为场景碰撞体
                self.detach_object()
                if (instance_id and self.scene_manager
                        and instance_id in self.scene_manager.processed_meshes):
                    self.scene_manager._apply_collision_meshes(
                        [(instance_id, self.scene_manager.processed_meshes[instance_id])])
                self._spin_wait(0.2)

                if (carry_result is not None and
                        carry_result.result.error_code.val == MoveItErrorCodes.SUCCESS):
                    carry_traj = carry_result.result.planned_trajectory
                    carry_cost_info = self.trajectory_cost(carry_traj)
                    fast_r['carry_cost'] = carry_cost_info['cost']
                    fast_r['carry_trajectory'] = carry_traj
                    carry_feasible.append(fast_r)
                    self.get_logger().info(
                        f"[Phase1.5] {cand_id} carry OK "
                        f"(cost={carry_cost_info['cost']:.3f})")
                else:
                    err = (carry_result.result.error_code.val
                           if carry_result else "no response")
                    self.get_logger().warn(
                        f"[Phase1.5] {cand_id} carry FAILED: "
                        f"{_MOVEIT_ERROR_NAMES.get(err, err)}")

            phase15_time = time.perf_counter() - t0 - phase1_time
            self.get_logger().info(
                f"[Phase1.5] {len(carry_feasible)}/{k} carry-feasible "
                f"in {phase15_time:.2f}s")

            if not carry_feasible:
                self.get_logger().error(
                    "[Phase1.5] All top-K carry checks failed, "
                    "falling back to approach-only ranking")
                carry_feasible = feasible[:top_n]

            # 用 approach_cost + carry_cost 综合排序
            def _combined_key(r):
                return (r['cost'] + r.get('carry_cost', 0.0)
                        - score_weight * r.get('score', 0.0))
            carry_feasible.sort(key=_combined_key)
            to_refine = carry_feasible
        else:
            to_refine = feasible[:top_n]

        # ── Phase 2: grasp 规划 + pregrasp shortcutting ──
        self.get_logger().info(
            f"[Phase2] Grasp planning + shortcutting for top-{len(to_refine)} candidates")
        results = []
        cur_removed = None

        for fast_r in to_refine:
            aff = fast_r['_aff']
            iid = aff.get("instance_id", "")
            cur_removed = self._manage_scene_for_candidate(iid, cur_removed)

            # 重新移除 Phase 1 中因冲突而临时移除的邻居
            removed_neighbors = fast_r.get("removed_neighbors", [])
            for nid in removed_neighbors:
                self.remove_object_from_scene(nid)
            if removed_neighbors:
                time.sleep(0.3)

            # 规划 grasp 段（pregrasp 末端 → 抓取点，~10cm）
            pregrasp_end_state = fast_r.get("pregrasp_end_state")
            if pregrasp_end_state is None:
                self.get_logger().warn(
                    f"[Phase2] No pregrasp_end_state for candidate {aff.get('id', '?')}, skipping")
                self._restore_neighbors(removed_neighbors)
                continue

            grasp_goal = self.build_goal(
                xyz=fast_r["center"],
                rpy=fast_r["adjusted_grasp_pose"],
                drive_joint_rad=fast_r["drive_joint_rad"],
                allowed_time=fast_filter_time,
                planner_id="RRTConnect",
                start_joint_state=pregrasp_end_state,
                pos_tol=pos_tol, ori_tol=ori_tol,
                num_planning_attempts=3
            )

            label = f"[Phase2 grasp {aff.get('id', '?')}]"
            grasp_result = self._send_action_goal(
                self.move_action_client, grasp_goal,
                send_timeout=15.0,
                result_timeout=fast_filter_time + 30.0,
                label=label)

            self._restore_neighbors(removed_neighbors)

            if grasp_result is None or grasp_result.result.error_code.val != MoveItErrorCodes.SUCCESS:
                err = grasp_result.result.error_code.val if grasp_result else "no response"
                self.get_logger().warn(
                    f"[Phase2] Grasp failed for candidate {aff.get('id', '?')}: "
                    f"{_MOVEIT_ERROR_NAMES.get(err, err)}, skipping")
                continue

            # 补充 grasp 字段
            fast_r["grasp_trajectory"] = grasp_result.result.planned_trajectory
            fast_r["final_joint_state"] = self.extract_final_joint_state(
                grasp_result.result.planned_trajectory)
            if fast_r["final_joint_state"] is None:
                self.get_logger().warn(
                    f"[Phase2] Failed to extract final state for candidate {aff.get('id', '?')}")
                continue

            # RDP shortcutting 优化 pregrasp 轨迹
            fast_r["pregrasp_trajectory"] = self.shortcut_trajectory(
                fast_r["pregrasp_trajectory"])

            # 用简化后的轨迹重新计算 cost
            main_traj = fast_r["pregrasp_trajectory"]
            if start_xyz is not None and "pregrasp_center" in fast_r:
                straight = float(np.linalg.norm(
                    np.array(fast_r["pregrasp_center"]) - np.array(start_xyz)))
            else:
                straight = 0.0
            cost_info = self.trajectory_cost(main_traj, cartesian_dist=straight)
            fast_r['cost'] = cost_info['cost']
            fast_r['cost_metrics'] = cost_info['metrics']
            results.append(fast_r)

        self._restore_last_removed(cur_removed)

        # 清理内部字段，区分完整规划 vs 备选
        attempted_ids = {r.get('id') for r in to_refine}
        refined_ids = {r.get('id') for r in results}

        # 追加未进入 Phase 2 的候选（仅有 pregrasp，作为备选信息）
        for r in feasible:
            rid = r.get('id')
            if rid not in refined_ids and rid not in attempted_ids:
                r.pop('_aff', None)
                results.append(r)

        for r in results:
            r.pop('_aff', None)

        planning_time = time.perf_counter() - t0
        self.get_logger().info(
            f"[Approaching] Planning complete: {len(results)}/{len(grasp_candidates)} succeeded "
            f"in {planning_time:.2f}s (phase1={phase1_time:.1f}s)")

        # 完整规划的候选排在前面，备选排在后面
        def _final_key(r):
            has_grasp = 0 if "grasp_trajectory" in r else 1
            combined = (r['cost'] + r.get('carry_cost', 0.0)
                        - score_weight * r.get('score', 0.0))
            return (has_grasp, combined)
        results.sort(key=_final_key)
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
        end_pos: Optional[List[float]] = None,
        pos_tol: float = 0.01,
        ori_tol: float = 0.1
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
            time.sleep(0.1)

            # --- 执行接近+抓取 ---
            if self.execution_mode:
                self.set_gripper(best_candidate["drive_joint_rad"])
                time.sleep(0.3)

                if not self.execute_trajectory(best_candidate["pregrasp_trajectory"]):
                    self.get_logger().error("[Approaching] Failed")
                    return result

                if not self.execute_trajectory(best_candidate["grasp_trajectory"]):
                    self.get_logger().error("[Grasp] Failed")
                    return result

                self.set_gripper(close_rad)
                time.sleep(0.3)

            # 计算实际抓取姿态（提前计算，供 attach 和 retreat 使用）
            # 优先使用 plan_candidate 已计算好的 adjusted_grasp_pose
            actual_rpy = best_candidate.get("adjusted_grasp_pose") or (
                best_candidate.get("grasp_pose_rpy") or (
                    [grasp_pose_base[0], grasp_pose_base[1], best_candidate["angle"]]
                    if "angle" in best_candidate else list(grasp_pose_base)
                )
            )

            # 附着物体到夹爪（影响后续碰撞检测）
            # 传入 TCP 位姿，将 mesh 转换为 gripper-relative 坐标
            self.get_logger().info(f"[Scene] Attaching object '{instance_id}' to gripper for carrying phase")
            attach_ok = self.attach_object_mesh(
                instance_id,
                tcp_xyz=best_candidate["center"],
                tcp_rpy=actual_rpy
            )
            if attach_ok:
                object_attached = True
                self.get_logger().info(f"[Attach] '{instance_id}' attached, "
                                       f"attached_objects={list(self.attached_objects.keys())}")
            else:
                self.get_logger().warn(f"[Attach] Failed to attach '{instance_id}', "
                                       f"continuing without attached object")

            # 等待 ACO 状态通过 ROS topic 传播到 MoveIt 规划场景
            self._spin_wait(0.5)

            # retreat: 笛卡尔直线回退到预抓取点
            pregrasp_center = best_candidate["pregrasp_center"]
            grasp_quat = R.from_euler("xyz", actual_rpy, degrees=False).as_quat()
            retreat_pose = Pose()
            retreat_pose.position = Point(x=float(pregrasp_center[0]), y=float(pregrasp_center[1]), z=float(pregrasp_center[2]))
            retreat_pose.orientation = Quaternion(x=float(grasp_quat[0]), y=float(grasp_quat[1]), z=float(grasp_quat[2]), w=float(grasp_quat[3]))
            retreat_result = self.plan_cartesian_path(
                waypoints=[retreat_pose], start_joint_state=current_joint_state,
                drive_joint_rad=close_rad, max_step=0.005, avoid_collisions=True)
            if retreat_result is not None:
                retreat_traj, current_joint_state = retreat_result
            else:
                self.get_logger().warn("[Retreat] Cartesian failed, falling back to sampling planner")
                retreat_traj = None
                for planner, t, attempts in [("RRTConnect", 15.0, 5)]:
                    self.get_logger().info(f"[Retreat] Trying {planner} t={t}s")
                    goal = self.build_goal(
                        xyz=pregrasp_center, rpy=actual_rpy,
                        drive_joint_rad=close_rad,
                        start_joint_state=current_joint_state,
                        planner_id=planner, allowed_time=t,
                        num_planning_attempts=attempts,
                        pos_tol=0.05, ori_tol=0.5)
                    result_msg = self._send_action_goal(
                        self.move_action_client, goal,
                        send_timeout=15.0, result_timeout=t + 15.0, label="[Retreat]")
                    if result_msg is not None and result_msg.result.error_code.val == MoveItErrorCodes.SUCCESS:
                        retreat_traj = result_msg.result.planned_trajectory
                        current_joint_state = self.extract_final_joint_state(retreat_traj)
                        self.get_logger().info(f"[Retreat] {planner} succeeded")
                        break
                    err = result_msg.result.error_code.val if result_msg else "no response"
                    print(f"[Retreat] {planner} failed with error code: {err}")
                    self.get_logger().warn(f"[Retreat] {planner} failed: {_MOVEIT_ERROR_NAMES.get(err, err)}")
                if retreat_traj is None:
                    self.get_logger().error("[Retreat] All planners failed")
                    return result
            result["trajectories"]["retreat"] = retreat_traj

            if not self.execute_trajectory(retreat_traj):
                self.get_logger().error("[Retreat] Execution failed")
                return result

            # carry: 搬运到篮子上方（篮子顶沿 + clearance）
            basket_rim_z = basket["center"][2] + basket["size"][2] / 2
            place_z = basket_rim_z + place_clearance
            carry_xyz = [float(basket["center"][0]), float(basket["center"][1]), float(place_z)]
            carry_rpy = (-np.pi, 0.0, float(actual_rpy[2]))

            self.get_logger().info(f"[Carrying] target={carry_xyz}")
            self._spin_wait(0.8)

            # 优先复用 Phase 1.5 已验证的 carry 轨迹
            # Phase 1.5 使用 pregrasp_end_state 规划，与 retreat 后的
            # current_joint_state 可能不同（7DOF 冗余臂 IK 零空间差异），
            # 直接重新规划可能因不同的关节构型而失败
            preplan_carry = best_candidate.get("carry_trajectory")
            carry_result = None

            if preplan_carry is not None:
                self.get_logger().info("[Carrying] Reusing Phase 1.5 pre-planned trajectory")
                carry_traj = preplan_carry
                carry_final = self.extract_final_joint_state(carry_traj)
                if carry_final is not None:
                    carry_result = (carry_traj, carry_final)
                else:
                    self.get_logger().warn(
                        "[Carrying] Phase 1.5 trajectory has no final state, re-planning")

            # 回退：从当前状态重新规划
            if carry_result is None:
                for planner, t, attempts, p_tol, o_tol in [
                    ("RRTConnect", 15.0, 10, 0.05, 0.3),
                ]:
                    self.get_logger().info(f"[Carrying] {planner} t={t}s pos={p_tol} ori={o_tol}")
                    carry_goal = self.build_goal(
                        xyz=carry_xyz, rpy=carry_rpy, drive_joint_rad=close_rad,
                        start_joint_state=current_joint_state,
                        planner_id=planner, allowed_time=t,
                        num_planning_attempts=attempts,
                        pos_tol=p_tol, ori_tol=o_tol)
                    result_msg = self._send_action_goal(
                        self.move_action_client, carry_goal,
                        send_timeout=15.0, result_timeout=t + 15.0, label="[Carrying]")
                    if result_msg is not None and result_msg.result.error_code.val == MoveItErrorCodes.SUCCESS:
                        carry_result = (result_msg.result.planned_trajectory,
                                        self.extract_final_joint_state(result_msg.result.planned_trajectory))
                        self.get_logger().info(f"[Carrying] {planner} succeeded")
                        break
                    err = result_msg.result.error_code.val if result_msg else "no response"
                    self.get_logger().error(
                        f"[Carrying] error_code={err} ({_MOVEIT_ERROR_NAMES.get(err, 'UNKNOWN')}), "
                        f"start_joints={current_joint_state}, "
                        f"target_xyz={carry_xyz}, "
                        f"attached={list(self.attached_objects.keys())}, "
                        f"place_z={place_z}, basket_rim_z={basket_rim_z}")
                    self.get_logger().warn(f"[Carrying] {planner} failed: {_MOVEIT_ERROR_NAMES.get(err, err)}")

            if carry_result is None:
                self.get_logger().error("[Carrying] All planners failed")
                return result
            carry_traj, current_joint_state = carry_result
            if current_joint_state is None:
                self.get_logger().error("[Carrying] Failed to extract final state")
                return result
            carry_traj = self.shortcut_trajectory(carry_traj)
            result["trajectories"]["carrying"] = carry_traj

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
            time.sleep(0.1)

            # return: 严格按关节角回到 home
            build_fn = lambda plnr, t, att, pt, ot: self.build_joint_goal(
                joint_positions=home_joints or [0.0] * 7,
                drive_joint_rad=self.gripper_joint_max,
                start_joint_state=current_joint_state,
                planner_id=plnr, allowed_time=t,
                num_planning_attempts=att)

            return_traj = None
            for planner, t, attempts, tol_scale in [
                ("RRTConnect", 15.0, 5, 1.0),
            ]:
                self.get_logger().info(f"[Returning] Trying {planner} (time={t}s)")
                return_goal = build_fn(planner, t, attempts,
                                       pos_tol * tol_scale, ori_tol * tol_scale)
                result_msg = self._send_action_goal(
                    self.move_action_client, return_goal,
                    send_timeout=15.0, result_timeout=t + 15.0, label="[Returning]")
                if result_msg is not None and result_msg.result.error_code.val == MoveItErrorCodes.SUCCESS:
                    return_traj = result_msg.result.planned_trajectory
                    self.get_logger().info(f"[Returning] {planner} succeeded")
                    break
                err = result_msg.result.error_code.val if result_msg else "no response"
                self.get_logger().warn(f"[Returning] {planner} failed: {_MOVEIT_ERROR_NAMES.get(err, err)}")

            if return_traj is None:
                self.get_logger().error("[Returning] All planners failed")
                return result
            result["trajectories"]["returning"] = return_traj

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
            "grasp_center_3d": best.get("grasp_center_3d"),
            "pregrasp_center_3d": best.get("pregrasp_center_3d"),
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
    pos_tol: Optional[float] = None,
    ori_tol: Optional[float] = None,
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

    # basket: target_pos 的 x,y 定位框子，z 自动算（底面贴桌面）
    basket_size = basket_cfg.get("outer_size", [0.3, 0.2, 0.15])
    default_center = basket_cfg.get("center", [0.4, 0.3, 0.1])
    if target_pos:
        basket_center = [target_pos[0], target_pos[1], basket_size[2] / 2]
    else:
        basket_center = default_center
    basket = {
        "id": basket_cfg.get("default_id", "basket_1"),
        "center": basket_center,
        "size": basket_size
    }

    home_joints = home_config.get("joints", [0.0] * 7)

    # 容差: 传入 > 默认值
    _pos_tol = pos_tol if pos_tol is not None else 0.01
    _ori_tol = ori_tol if ori_tol is not None else 0.1


    try:
        if executor.execution_mode:
            goal = executor.build_joint_goal(
                joint_positions=home_joints,
                drive_joint_rad=executor.gripper_joint_max,
                allowed_time=5.0, planner_id="RRTConnect"
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
        time.sleep(0.3)

        executor.add_workspace_walls()
        time.sleep(0.2)

        executor.add_open_top_basket_to_scene(
            object_id=basket["id"],
            center=basket["center"],
            outer_size=basket["size"],
            wall_t=basket_cfg.get("wall_thickness", 0.005),
            bottom_t=basket_cfg.get("bottom_thickness", 0.005)
        )
        time.sleep(0.2)

        all_candidates = executor.parse_affordances_to_candidates(
            affordance_data, depth_path, seg_json_path,
            target_object_index=target_object_index,
            extra_open=grasp_cfg.get("extra_open", 0.020),
            pixel_to_meter=grasp_cfg.get("pixel_to_meter", 0.0001)
        )

        if not all_candidates:
            executor.get_logger().error("No valid grasp candidates")
            print(f"[DIAG] parse_affordances_to_candidates returned 0 candidates for obj_index={target_object_index}")
            return results

        print(f"[DIAG] {len(all_candidates)} candidates generated for obj_index={target_object_index}")

        # grasp_pose_base: 传入 > config > default
        grasp_pose_base = tuple(start_pos[3:]) if start_pos \
            else tuple(home_config.get("orientation", (-np.pi, 0.0, 0.0)))
        start_xyz = start_pos[:3] if start_pos else home_config.get("position", [0.27, 0.0, 0.307])
        place_clearance = grasp_cfg.get("place_clearance", 0.05)
        gripper_close_tightness = grasp_cfg.get("gripper_close_tightness", 0.02)

        ranked_results = executor.plan_rank_all_candidates(
            all_candidates,
            grasp_pose=grasp_pose_base,
            start_joint_state=executor.current_joint_state,
            start_xyz=start_xyz,
            pos_tol=_pos_tol, ori_tol=_ori_tol,
            basket=basket,
            place_clearance=place_clearance,
            gripper_close_tightness=gripper_close_tightness
        )

        if not ranked_results:
            executor.get_logger().error("All candidates failed planning")
            print(f"[DIAG] plan_rank_all_candidates returned 0 results (all {len(all_candidates)} candidates failed)")
            return results

        # 按 instance_id 去重，保留最优候选
        seen = set()
        unique_candidates = []
        for c in ranked_results:
            if c["instance_id"] not in seen:
                unique_candidates.append(c)
                seen.add(c["instance_id"])

        best = unique_candidates[0]
        print(f"[DIAG] Best candidate: id={best.get('instance_id')}, "
              f"center={best.get('center')}, score={best.get('score')}")
        t0 = time.perf_counter()
        full_result = executor.execute_complete_grasp_sequence(
            best_candidate=best, basket=basket,
            place_clearance=place_clearance,
            grasp_pose_base=grasp_pose_base,
            home_joints=home_joints,
            end_pos=end_pos,
            pos_tol=_pos_tol, ori_tol=_ori_tol
        )
        seq_time = time.perf_counter() - t0
        print(f"[DIAG] execute_complete_grasp_sequence: success={full_result.get('success')}, "
              f"time={seq_time:.2f}s, steps={len(full_result.get('trajectories', {}))}")

        trajs = full_result["trajectories"]
        result = {
            "instance_id": best["instance_id"],
            "success": full_result["success"],
            "planning_time": seq_time,
            "rank": 1,
            "grasp_center_3d": best.get("center"),
            "pregrasp_center_3d": best.get("pregrasp_center"),
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
                time.sleep(0.1)
                executor.clear_pointcloud_obstacles()
                executor.remove_basket_from_scene(basket["id"])
                executor.remove_workspace_walls()
                executor.clear_octomap()
                time.sleep(0.2)
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
