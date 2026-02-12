#!/usr/bin/env python3
import json
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull, QhullError

from std_srvs.srv import Empty

# MoveIt2 PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Pose


def rle_fr_string(counts_str: str):
    """解码RLE压缩字符串"""
    out = []
    p = 0
    while p < len(counts_str):
        val = 0
        k = 0
        more = 1
        while more:
            c = ord(counts_str[p]) - 48
            val |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if (not more) and (c & 0x10):
                val |= -1 << (5 * k)
        if len(out) > 2:
            val += out[-2]
        out.append(int(val))
    return out

def coco_rle_decode(rle_obj):
    """解码COCO RLE格式的mask"""
    h, w = rle_obj["size"]
    counts = rle_fr_string(rle_obj["counts"])

    total = h * w
    flat = np.zeros(total, dtype=np.uint8)

    idx = 0
    flag = 0
    for run_len in counts:
        if run_len < 0:
            run_len = 0
        if idx + run_len > total:
            run_len = total - idx
        if flag == 1:
            flat[idx:idx + run_len] = 1
        idx += run_len
        flag ^= 1
        if idx >= total:
            break

    return flat.reshape((h, w), order="F").astype(bool)

def pose_identity() -> Pose:
    """返回单位姿态"""
    p = Pose()
    p.position.x = 0.0
    p.position.y = 0.0
    p.position.z = 0.0
    p.orientation.w = 1.0
    return p

class SceneManager:
    def __init__(self, node: Node, config: dict = None, **kwargs):
        """
        Args:
            node: 父ROS2节点
            config: 完整配置字典（优先使用）
            **kwargs: 传统参数方式（向后兼容）
        """
        self.node = node

        # 从config或kwargs读取参数
        if config is not None:
            cam = config["camera"]
            intrinsics = cam["intrinsics"]
            extrinsics = cam["extrinsics"]
            scene_cfg = config["scene_manager"]
            robot_cfg = config["robot"]

            self.frame_id = robot_cfg["base_frame"]
            self.fx = float(intrinsics["fx"])
            self.fy = float(intrinsics["fy"])
            self.cx = float(intrinsics["cx"])
            self.cy = float(intrinsics["cy"])
            self.depth_scale = float(cam["depth_scale"])
            self.max_range_m = float(cam["max_range_m"])
            self.min_range_m = float(cam["min_range_m"])
            self.stride = int(scene_cfg["stride"])
            self.score_thresh = float(scene_cfg["score_thresh"])
            self.t_cam2base = np.array(extrinsics["translation"], dtype=np.float64).reshape(3)
            self.q_cam2base = np.array(extrinsics["quaternion"], dtype=np.float64).reshape(4)
        else:
            # 向后兼容：从kwargs读取
            self.frame_id = kwargs.get("frame_id", "link_base")
            self.fx = float(kwargs.get("fx"))
            self.fy = float(kwargs.get("fy"))
            self.cx = float(kwargs.get("cx"))
            self.cy = float(kwargs.get("cy"))
            self.depth_scale = float(kwargs.get("depth_scale", 0.001))
            self.max_range_m = float(kwargs.get("max_range_m", 3.0))
            self.min_range_m = float(kwargs.get("min_range_m", 0.02))
            self.stride = int(kwargs.get("stride", 1))
            self.score_thresh = float(kwargs.get("score_thresh", 0.3))
            self.t_cam2base = np.asarray(kwargs.get("t_cam2base"), dtype=np.float64).reshape(3)
            self.q_cam2base = np.asarray(kwargs.get("q_cam2base"), dtype=np.float64).reshape(4)

        self.R_cam2base = R.from_quat(self.q_cam2base).as_matrix()

        # 实例管理
        self.instances = []
        self.disabled_instance_ids = set()
        self.processed_meshes: dict[str, Mesh] = {}
        self.last_table_z: float | None = None

        # ROS2 发布器
        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.topic = "/camera/depth/color/points"
        self.pub = self.node.create_publisher(PointCloud2, self.topic, cloud_qos)

        # ROS2 服务客户端
        self._clear_cli = self.node.create_client(Empty, "/clear_octomap")
        self._apply_scene_cli = self.node.create_client(ApplyPlanningScene, "/apply_planning_scene")

    def update_scene(self, depth_path: str, seg_json_path: str):
        # 1. 清除OctoMap
        if self._clear_cli.wait_for_service(timeout_sec=2.0):
            try:
                self._clear_cli.call_async(Empty.Request())
            except Exception as e:
                self.node.get_logger().error(f"Failed to clear OctoMap: {e}")
        else:
            self.node.get_logger().warn("/clear_octomap service not available")

        # 2. 加载分割实例
        self.instances = self._load_instances(seg_json_path, self.score_thresh)
        if not self.instances:
            self.node.get_logger().warn("No instances loaded from segmentation")

        # 3. 读取深度图
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            self.node.get_logger().error(f"Failed to read depth image: {depth_path}")
            return
        z_m = depth_img.astype(np.float32) * self.depth_scale

        # 4. 解码所有mask
        full_masks: list[np.ndarray] = []
        inst_ids: list[str] = []
        for inst in self.instances:
            m = coco_rle_decode(inst["mask_rle"])
            full_masks.append(m)
            inst_ids.append(inst["id"])

        # 5. 构建背景点云mask
        if full_masks:
            union_mask = np.logical_or.reduce(full_masks)
            bg_mask = ~union_mask
            coverage_ratio = union_mask.sum() / union_mask.size
            # self.node.get_logger().info(f"Object mask coverage: {coverage_ratio*100:.1f}%")
        else:
            bg_mask = np.ones_like(z_m, dtype=bool)
        pts_bg = self._build_points(z_m, bg_mask)
        # self.node.get_logger().info(f"Background points generated: {pts_bg.shape[0]}")

        # 6. 估计背景z
        base_z = None
        try:
            base_z = self._estimate_base_z_from_bg(pts_bg)
        except Exception as e:
            self.node.get_logger().warn(f"Base z estimation failed: {e}")

        if base_z is not None:
            table_filter_margin = 0.01
        else:
            table_filter_margin = None

        # 7. 先为每个启用的实例生成凸包
        k = 7 if self.stride <= 2 else 5
        kernel = np.ones((k, k), np.uint8)

        meshes: list[tuple[str, Mesh]] = []
        for inst, m in zip(self.instances, full_masks):
            inst_id = inst["id"]
            if inst_id in self.disabled_instance_ids:
                continue

            m_u8 = m.astype(np.uint8)
            m_eroded = cv2.erode(m_u8, kernel, iterations=1).astype(bool)

            pts_obj = self._build_points(z_m, m_eroded)
            # 滤掉base点（减少凸包底部发散）
            if base_z is not None and table_filter_margin is not None and pts_obj.shape[0] > 0:
                pts_obj = pts_obj[pts_obj[:, 2] > (base_z + table_filter_margin)]

            mesh = self._mesh_from_convex_hull(pts_obj)
            if mesh is None:
                continue
            meshes.append((inst_id, mesh))
            self.processed_meshes[inst_id] = mesh

        # 8. Apply 碰撞场景（凸包先上场景）
        if meshes:
            if not self._apply_scene_cli.wait_for_service(timeout_sec=2.0):
                self.node.get_logger().warn("/apply_planning_scene not ready")
            else:
                self._apply_collision_meshes(meshes)
                self.node.get_logger().info(f"Applied {len(meshes)} collision meshes")

        # 9. 发布背景点云给 OctoMap
        if pts_bg.shape[0] == 0:
            self.node.get_logger().warn("Background pointcloud is EMPTY! OctoMap will not update.")
        else:
            # 增加发布次数和时间，确保OctoMap有足够时间处理
            for i in range(30):
                self._publish_pointcloud(pts_bg)
                rclpy.spin_once(self.node, timeout_sec=0.0)
                self.node.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.05))
            self.node.get_logger().info(f"Published background cloud: {pts_bg.shape[0]} points")

    def remove_instance_hull(self, instance_id: str):
        """移除目标物品凸包"""
        self.disabled_instance_ids.add(instance_id)
        if not self._apply_scene_cli.wait_for_service(timeout_sec=3.0):
            return

        scene = PlanningScene()
        scene.is_diff = True

        co = CollisionObject()
        co.id = instance_id
        co.header.frame_id = self.frame_id
        co.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(co)

        req = ApplyPlanningScene.Request()
        req.scene = scene
        self._apply_scene_cli.call_async(req)

    def _load_instances(self, seg_json_path: str, score_thresh: float):
        """加载分割结果JSON"""
        try:
            with open(seg_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.node.get_logger().error(f"Failed to load JSON: {e}")
            return []

        results = data.get("results", [])
        if not results:
            return []

        r0 = results[0]
        objs = r0.get("objects", [])
        text_prompt = str(r0.get("text_prompt", "obj"))
        
        instances = []
        for i, o in enumerate(objs):
            score = float(o.get("score", 1.0))
            if score < score_thresh:
                continue
            
            cat_raw = o.get("category", "obj")
            cat = text_prompt if isinstance(cat_raw, int) else str(cat_raw)

            bbox = o.get("bbox", None)
            mask_rle = o.get("mask", None)
            if mask_rle is None:
                continue
            instances.append({"idx": i,"id": f"{cat}_{i}","category": cat,"score": score,"bbox": bbox,"mask_rle": mask_rle,})
        return instances

    def _estimate_base_z_from_bg(self, pts_bg: np.ndarray) -> float | None:
        """给出背景的均值高度，从而与物品区分"""
        if pts_bg is None or pts_bg.shape[0] < 2000:
            return None
        z = pts_bg[:, 2].astype(np.float64)
        z = z[np.isfinite(z)]
        if z.size < 2000:
            return None

        z_lo, z_hi = np.percentile(z, [2, 98])
        z = z[(z >= z_lo) & (z <= z_hi)]
        if z.size < 2000:
            return None

        hist, edges = np.histogram(z, bins=200)
        i = int(np.argmax(hist))
        table_z = 0.5 * (edges[i] + edges[i + 1])
        return float(table_z)

    def _build_points(self, z_m: np.ndarray, mask_full: np.ndarray):
        """根据深度图和mask构建3D点云"""
        z = z_m[::self.stride, ::self.stride]
        hs, ws = z.shape

        u = (np.arange(ws, dtype=np.float32) * self.stride)
        v = (np.arange(hs, dtype=np.float32) * self.stride)
        uu, vv = np.meshgrid(u, v)

        valid = np.isfinite(z) & (z > self.min_range_m) & (z < self.max_range_m)

        if mask_full is not None:
            mask = mask_full[::self.stride, ::self.stride].astype(bool)
            valid &= mask

        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy

        pts_cam = np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float64)
        if pts_cam.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        pts_world = (pts_cam @ self.R_cam2base.T) + self.t_cam2base.reshape(1, 3)
        return pts_world.astype(np.float32)

    def _mesh_from_convex_hull(self, pts_world: np.ndarray) -> Mesh | None:
        """从点云构建凸包Mesh"""
        if pts_world.shape[0] < 20:
            return None

        # 降采样避免凸包计算过慢
        step = max(1, pts_world.shape[0] // 5000)
        pts = pts_world[::step]

        try:
            hull = ConvexHull(pts)
        except QhullError:
            return None

        verts = pts[hull.vertices]
        index_map = {old_i: new_i for new_i, old_i in enumerate(hull.vertices.tolist())}

        mesh = Mesh()
        mesh.vertices = []
        for v in verts:
            from geometry_msgs.msg import Point
            p = Point()
            p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
            mesh.vertices.append(p)

        mesh.triangles = []
        for tri in hull.simplices:
            try:
                a = index_map[int(tri[0])]
                b = index_map[int(tri[1])]
                c = index_map[int(tri[2])]
            except KeyError:
                continue
            
            t = MeshTriangle()
            t.vertex_indices = [a, b, c]
            mesh.triangles.append(t)

        if len(mesh.triangles) == 0 or len(mesh.vertices) < 4:
            return None
        
        return mesh

    def _publish_pointcloud(self, pts: np.ndarray):
        """发布点云消息"""
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        msg = point_cloud2.create_cloud_xyz32(header, pts.tolist())
        self.pub.publish(msg)

    def _apply_collision_meshes(self, meshes: list[tuple[str, Mesh]]):
        """应用碰撞Mesh到规划场景"""
        if not self._apply_scene_cli.service_is_ready():
            return

        scene = PlanningScene()
        scene.is_diff = True

        for oid, mesh in meshes:
            co = CollisionObject()
            co.id = oid
            co.header.frame_id = self.frame_id
            co.operation = CollisionObject.ADD
            co.meshes.append(mesh)
            co.mesh_poses.append(pose_identity())
            scene.world.collision_objects.append(co)

        req = ApplyPlanningScene.Request()
        req.scene = scene
        fut = self._apply_scene_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=2.0)

# ==================== 示例用法 ====================
def example_with_config():
    """使用config文件初始化SceneManager"""
    rclpy.init()

    # 创建一个简单的ROS2节点作为父节点
    parent_node = Node("scene_manager_example")

    # 加载配置
    with open("config/xarm7_config.json", 'r') as f:
        config = json.load(f)

    # 创建SceneManager，传入父节点
    scene_mgr = SceneManager(node=parent_node, config=config)

    depth_path = "test_data/grasp-wrist-dpt_opt.png"
    seg_json_path = "test_data/rgb_detection_wrist"

    scene_mgr.update_scene(depth_path=depth_path, seg_json_path=seg_json_path)

    try:
        rclpy.spin(parent_node)
    except KeyboardInterrupt:
        pass

    parent_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    example_with_config()