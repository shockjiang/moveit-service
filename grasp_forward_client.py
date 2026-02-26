#!/usr/bin/env python3
"""
GraspForwardClient — 轻量远程抓取规划客户端

仅通过 HTTP 向 moveit_service 发送数据并获取路径规划结果，
不依赖 ROS2，不依赖共享文件系统。

Usage::

    from grasp_forward_client import GraspForwardClient

    client = GraspForwardClient("http://192.168.1.100:14086")
    result = client.forward(depth_bytes, seg_dict, affordance_list)

    if result["success"]:
        for r in result["results"]:
            print(r["instance_id"], len(r["execution_steps"]), "steps")
"""

import json
import os

import requests

# 禁用代理，避免干扰连接
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""


def _to_json_str(data, name: str) -> str:
    """将 dict / list / 文件路径 / 原始 JSON 字符串统一转为 JSON 字符串。"""
    if isinstance(data, (dict, list)):
        return json.dumps(data)
    if isinstance(data, str) and os.path.isfile(data):
        with open(data, "r") as f:
            return f.read()
    if isinstance(data, str):
        return data
    raise TypeError(f"Unsupported {name} type: {type(data)}")


def _to_depth_bytes(depth) -> bytes:
    """将 文件路径 / np.ndarray / bytes 统一转为 PNG bytes。"""
    if isinstance(depth, str):
        with open(depth, "rb") as f:
            return f.read()
    if isinstance(depth, (bytes, bytearray)):
        return bytes(depth)

    # np.ndarray — lazy import，仅在需要时加载
    import numpy as np
    if isinstance(depth, np.ndarray):
        import cv2
        ok, buf = cv2.imencode(".png", depth)
        if not ok:
            raise ValueError("cv2.imencode failed for depth array")
        return buf.tobytes()

    raise TypeError(f"Unsupported depth type: {type(depth)}")


class GraspForwardClient:
    """轻量远程客户端：发送数据到 /forward，获取抓取规划结果。"""

    def __init__(self, base_url: str, robot_name: str = "xarm7", timeout: int = 1800):
        self.base_url = base_url.rstrip("/")
        self.robot_name = robot_name
        self.timeout = timeout

    def health(self) -> dict:
        """检查服务端是否在线。"""
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        return resp.json()

    def forward(self, depth, seg_json, affordance_json,
                robot_name=None, target_object_index=None) -> dict:
        """远程抓取规划。

        Args:
            depth: 深度图 — 文件路径 (str) / np.ndarray / bytes
            seg_json: 分割结果 — 文件路径 (str) / dict / list / JSON 字符串
            affordance_json: affordance — 文件路径 (str) / dict / list / JSON 字符串
            robot_name: 机械臂名称，默认 self.robot_name
            target_object_index: 目标物体索引，None 表示无指定目标

        Returns:
            dict: {"success": bool, "message": str, "results": [...]}
        """
        files = {"depth": ("depth.png", _to_depth_bytes(depth), "image/png")}
        data = {
            "seg_json": _to_json_str(seg_json, "seg_json"),
            "affordance_json": _to_json_str(affordance_json, "affordance_json"),
            "robot_name": robot_name or self.robot_name,
            "target_object_index": target_object_index if target_object_index is not None else -1,
        }

        resp = requests.post(
            f"{self.base_url}/forward",
            files=files, data=data, timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()
