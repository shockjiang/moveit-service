#!/usr/bin/env python3
import json
import time
import threading
from flask import Flask, request, jsonify

import rclpy
from rclpy.executors import MultiThreadedExecutor

from moveit_grasp_core import XArmGraspExecutor, _execute_grasp_core, _trajectory_to_json_safe


# 全局变量
executor_node = None
ros_executor = None
ros_thread = None
ros_worker_thread = None
task_queue = None  # 任务队列
result_queue = None  # 结果队列
app = Flask(__name__)

def ros_worker():
    """ROS2工作线程 - 处理任务队列中的规划请求"""
    global task_queue, result_queue, executor_node

    print("✓ ROS2工作线程已启动")

    while True:
        try:
            # 从队列获取任务
            task = task_queue.get()

            if task is None:  # 退出信号
                break

            task_id = task["task_id"]
            task_type = task["type"]
            params = task["params"]

            print(f"[Worker] 处理任务 {task_id}: {task_type}")

            try:
                if task_type == "grasp":
                    # 执行抓取规划
                    print(f"[Worker] 开始执行抓取任务 {task_id}")
                    result = run_grasp_pipeline(
                        depth_path=params["depth_path"],
                        seg_json_path=params["seg_json_path"],
                        affordance_path=params["affordance_path"],
                        target_object_index=params.get("target_object_index"),
                        return_trajectories=params.get("return_trajectories", False)
                    )
                    print(f"[Worker] 抓取任务完成，结果: {result.get('success')}")
                elif task_type == "update_scene":
                    # 更新场景
                    executor_node.scene_manager.update_scene(
                        params["depth_path"],
                        params["seg_json_path"]
                    )
                    result = {"success": True, "message": "场景更新成功"}
                else:
                    result = {"success": False, "message": f"未知任务类型: {task_type}"}

                # 将结果放入结果队列
                print(f"[Worker] 将结果放入队列: task_id={task_id}, success={result.get('success')}")
                result_queue.put({"task_id": task_id, "result": result})
                print(f"[Worker] 任务 {task_id} 完成，结果已放入队列")

            except Exception as e:
                print(f"[Worker] 任务 {task_id} 执行出错: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "task_id": task_id,
                    "result": {"success": False, "message": f"执行出错: {str(e)}"}
                })

            finally:
                task_queue.task_done()

        except Exception as e:
            print(f"[Worker] 工作线程错误: {e}")
            import traceback
            traceback.print_exc()

def init_ros_service(robot_name="xarm7", execution_mode=True):
    """初始化ROS服务（后台运行）"""
    global executor_node, ros_executor, ros_thread, ros_worker_thread, task_queue, result_queue

    if executor_node is not None:
        return

    # 加载配置
    config_path = f"config/{robot_name}_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    rclpy.init()
    executor_node = XArmGraspExecutor(
        execution_mode=execution_mode,
        camera_params=config["camera"],
        config=config
    )

    # 启动ROS后台线程 - 使用MultiThreadedExecutor支持多线程调用
    ros_executor = MultiThreadedExecutor(num_threads=4)
    ros_executor.add_node(executor_node)

    def ros_spin():
        try:
            ros_executor.spin()
        except Exception as e:
            print(f"ROS spin error: {e}")

    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()

    # 初始化任务队列
    import queue
    task_queue = queue.Queue()
    result_queue = queue.Queue()

    # 启动ROS2工作线程
    ros_worker_thread = threading.Thread(target=ros_worker, daemon=True)
    ros_worker_thread.start()

    # 等待节点完全初始化
    time.sleep(2.0)

    print(f"✓ ROS服务已启动 (robot={robot_name}, execution_mode={execution_mode})")


@app.route('/', methods=['GET'])
def index():
    """服务欢迎页面"""
    return jsonify({
        "service": "MoveIt HTTP Service",
        "version": "1.0",
        "robot": executor_node.planning_group if executor_node else "unknown",
        "endpoints": {
            "health": "GET /health - 健康检查",
            "update_scene": "POST /update_scene - 更新场景",
            "grasp": "POST /grasp - 执行抓取规划",
            "cleanup": "POST /cleanup - 清理场景（客户端执行完后调用）"
        },
        "status": "运行中"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        if executor_node is None:
            return jsonify({"status": "error", "message": "服务未初始化"}), 503
        return jsonify({"status": "healthy", "robot": executor_node.planning_group})
    except Exception as e:
        print(f"Health check error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/remove_object', methods=['POST'])
def remove_object():
    """从场景中移除指定碰撞物体"""
    global executor_node
    if executor_node is None:
        return jsonify({"success": False, "message": "服务未初始化"}), 503
    try:
        data = request.get_json()
        instance_id = data.get("instance_id")
        if not instance_id:
            return jsonify({"success": False, "message": "缺少 instance_id"}), 400
        ok = executor_node.remove_object_from_scene(instance_id)
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/attach_object', methods=['POST'])
def attach_object():
    """将物体附着到夹爪"""
    global executor_node
    if executor_node is None:
        return jsonify({"success": False, "message": "服务未初始化"}), 503
    try:
        data = request.get_json()
        instance_id = data.get("instance_id")
        if not instance_id:
            return jsonify({"success": False, "message": "缺少 instance_id"}), 400
        ok = executor_node.attach_object_mesh(instance_id)
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/detach_object', methods=['POST'])
def detach_object():
    """从夹爪分离物体"""
    global executor_node
    if executor_node is None:
        return jsonify({"success": False, "message": "服务未初始化"}), 503
    try:
        ok = executor_node.detach_object()
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/cleanup', methods=['POST'])
def cleanup_scene():
    """清理场景（客户端执行完毕后调用）"""
    global executor_node

    if executor_node is None:
        return jsonify({"success": False, "message": "服务未初始化"}), 503

    try:
        data = request.get_json() or {}
        basket_id = data.get("basket_id", "basket_1")

        executor_node.detach_object()
        import time as _time
        _time.sleep(0.3)
        executor_node.clear_pointcloud_obstacles()
        executor_node.remove_basket_from_scene(basket_id)
        executor_node.clear_octomap()
        _time.sleep(0.5)

        return jsonify({"success": True, "message": "场景清理完成"})
    except Exception as e:
        return jsonify({"success": False, "message": f"清理失败: {str(e)}"}), 500


@app.route('/update_scene', methods=['POST'])
def update_scene():
    """更新场景（添加碰撞物体）"""
    global executor_node, task_queue, result_queue

    if executor_node is None:
        return jsonify({"success": False, "message": "服务未初始化"}), 503

    try:
        data = request.get_json()
        depth_path = data.get("depth_path")
        seg_json_path = data.get("seg_json_path")

        if not all([depth_path, seg_json_path]):
            return jsonify({"success": False, "message": "缺少必要参数: depth_path, seg_json_path"}), 400

        # 生成任务ID
        import uuid
        task_id = str(uuid.uuid4())

        # 提交任务到队列
        task = {
            "task_id": task_id,
            "type": "update_scene",
            "params": {
                "depth_path": depth_path,
                "seg_json_path": seg_json_path
            }
        }
        task_queue.put(task)

        # 等待结果
        timeout = 30  # 30秒
        start_time = time.time()
        while True:
            try:
                result_item = result_queue.get(timeout=1)
                if result_item["task_id"] == task_id:
                    return jsonify(result_item["result"])
                else:
                    result_queue.put(result_item)
            except:
                pass

            if time.time() - start_time > timeout:
                return jsonify({"success": False, "message": "任务超时"}), 504

    except Exception as e:
        return jsonify({"success": False, "message": f"更新场景失败: {str(e)}"}), 500


@app.route('/grasp', methods=['POST'])
def execute_grasp():
    """执行抓取任务"""
    global executor_node, task_queue, result_queue

    if executor_node is None:
        return jsonify({"success": False, "message": "服务未初始化"}), 503

    try:
        data = request.get_json()
        depth_path = data.get("depth_path")
        seg_json_path = data.get("seg_json_path")
        affordance_path = data.get("affordance_path")
        target_object_index = data.get("target_object_index")
        plan_only = data.get("plan_only", False)  # 默认执行，设为True则只规划
        return_trajectories = data.get("return_trajectories", False)  # 是否返回完整轨迹数据

        if not all([depth_path, seg_json_path, affordance_path]):
            return jsonify({"success": False, "message": "缺少必要参数: depth_path, seg_json_path, affordance_path"}), 400

        # 临时切换执行模式
        original_mode = executor_node.execution_mode
        if plan_only:
            executor_node.execution_mode = False

        try:
            # 生成任务ID
            import uuid
            task_id = str(uuid.uuid4())

            # 提交任务到队列
            task = {
                "task_id": task_id,
                "type": "grasp",
                "params": {
                    "depth_path": depth_path,
                    "seg_json_path": seg_json_path,
                    "affordance_path": affordance_path,
                    "target_object_index": target_object_index,
                    "return_trajectories": return_trajectories
                }
            }
            task_queue.put(task)

            # 等待结果（最多30分钟 - 考虑多个候选点的规划）
            timeout = 1800  # 30分钟
            start_time = time.time()
            print(f"[Flask] 等待任务 {task_id} 的结果...")
            while True:
                try:
                    result_item = result_queue.get(timeout=1)
                    print(f"[Flask] 收到结果，task_id={result_item.get('task_id')}")
                    if result_item["task_id"] == task_id:
                        print(f"[Flask] 匹配成功，返回结果: {result_item['result'].get('success')}")
                        return jsonify(result_item["result"])
                    else:
                        # 不是我们的结果，放回队列
                        print(f"[Flask] task_id不匹配，放回队列")
                        result_queue.put(result_item)
                except Exception as e:
                    # 队列为空或超时，继续等待
                    if "Empty" not in str(type(e)):
                        print(f"[Flask] 等待结果时出错: {e}")

                if time.time() - start_time > timeout:
                    print(f"[Flask] 任务 {task_id} 超时")
                    return jsonify({"success": False, "message": "任务超时"}), 504

        finally:
            executor_node.execution_mode = original_mode

    except Exception as e:
        return jsonify({"success": False, "message": f"执行失败: {str(e)}"}), 500


def run_grasp_pipeline(depth_path, seg_json_path, affordance_path, target_object_index=None, return_trajectories=False):
    """执行完整的抓取流程（HTTP服务模式）"""
    global executor_node

    if executor_node is None:
        return {"success": False, "message": "服务未初始化"}

    # 使用executor_node中已经加载的配置
    config = executor_node.config

    try:
        # 调用核心逻辑
        results = _execute_grasp_core(
            executor=executor_node,
            depth_path=depth_path,
            seg_json_path=seg_json_path,
            affordance_path=affordance_path,
            config=config,
            target_object_index=target_object_index,
            return_full_trajectories=return_trajectories
        )

        if results and results[0]["success"]:
            return {
                "success": True,
                "message": "抓取任务完成",
                "results": results
            }
        else:
            return {"success": False, "message": "抓取执行失败"}

    except Exception as e:
        executor_node.get_logger().error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"执行出错: {str(e)}"}


def start_service(host="0.0.0.0", port=8000, robot_name="xarm7", execution_mode=True):
    """启动HTTP服务"""
    import socket

    init_ros_service(robot_name=robot_name, execution_mode=execution_mode)

    # 检测端口是否可用，不可用则自动递增
    actual_port = port
    for offset in range(10):
        actual_port = port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, actual_port))
                break
            except OSError:
                if offset == 0:
                    print(f"  端口 {actual_port} 被占用，尝试其他端口...")
    else:
        print(f"  端口 {port}-{port+9} 全部被占用，启动失败")
        return

    if actual_port != port:
        print(f"  原端口 {port} 被占用，改用端口 {actual_port}")

    print(f"✓ HTTP服务启动在 http://{host}:{actual_port}")
    print(f"  - 健康检查: GET  http://{host}:{actual_port}/health")
    print(f"  - 执行抓取: POST http://{host}:{actual_port}/grasp")
    app.run(host=host, port=actual_port, debug=False, threaded=True)


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    robot_name = sys.argv[2] if len(sys.argv) > 2 else "xarm7"
    execution_mode = True  # 默认执行模式

    print(f"启动MoveIt服务模式...")
    print(f"  机械臂: {robot_name}")
    print(f"  端口: {port}")
    print(f"  执行模式: {execution_mode}")

    start_service(port=port, robot_name=robot_name, execution_mode=execution_mode)
