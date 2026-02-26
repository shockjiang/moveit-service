#!/usr/bin/env python3
import json
import os
import queue
import socket
import sys
import tempfile
import time
import threading
import traceback
import uuid

from flask import Flask, request, jsonify
import rclpy
from rclpy.executors import MultiThreadedExecutor
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory, MoveItErrorCodes
from builtin_interfaces.msg import Duration

from moveit_grasp_core import GraspExecutor, _execute_grasp_core

app = Flask(__name__)

# Global state
_node: GraspExecutor = None
_task_queue: queue.Queue = None
_result_queue: queue.Queue = None
_ros_exec: MultiThreadedExecutor = None
_worker_thread: threading.Thread = None
_current_robot: str = None
_execution_mode: bool = False
_init_lock = threading.Lock()


# ---------------------------------------------------------------------------
#  Worker-thread task handlers (safe to call ROS actions)
# ---------------------------------------------------------------------------

def _json_to_robot_trajectory(traj_json):
    traj = RobotTrajectory()
    jt = traj.joint_trajectory
    jt.joint_names = traj_json["joint_names"]

    for pt_data in traj_json["points"]:
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in pt_data["positions"]]
        if pt_data.get("velocities"):
            pt.velocities = [float(v) for v in pt_data["velocities"]]
        if pt_data.get("accelerations"):
            pt.accelerations = [float(v) for v in pt_data["accelerations"]]
        tfs = pt_data["time_from_start"]
        pt.time_from_start = Duration(sec=int(tfs["sec"]), nanosec=int(tfs["nanosec"]))
        jt.points.append(pt)

    return traj


def _run_update_scene(node, params):
    node.scene_manager.update_scene(params["depth_path"], params["seg_json_path"])
    return {"success": True, "message": "Scene updated"}


def _run_move_home(node, _params):
    config = node.config
    n_joints = len(node.arm_joint_names)
    home_joints = config.get("home", {}).get("joints", [0.0] * n_joints)

    original_mode = node.execution_mode
    node.execution_mode = True
    try:
        goal = node.build_joint_goal(
            joint_positions=home_joints,
            drive_joint_rad=node.gripper_joint_max,
            tolerance=0.001,
            allowed_time=30.0,
            planner_id="RRTConnect",
            start_joint_state=None,
            plan_only=False,
        )
        result_msg = node._send_action_goal(
            node.move_action_client, goal,
            send_timeout=30.0, result_timeout=60.0, label="[HOME]")

        if result_msg is None or result_msg.result.error_code.val != MoveItErrorCodes.SUCCESS:
            error_val = result_msg.result.error_code.val if result_msg else "None"
            return {"success": False, "message": f"HOME failed, error_code={error_val}"}
        return {"success": True, "message": "HOME position reached"}
    finally:
        node.execution_mode = original_mode


def _run_execute_trajectory(node, params):
    traj = _json_to_robot_trajectory(params["trajectory_json"])

    original_mode = node.execution_mode
    node.execution_mode = True
    try:
        ok = node.execute_trajectory(traj, timeout_sec=120.0)
    finally:
        node.execution_mode = original_mode

    if ok:
        return {"success": True, "message": "Trajectory executed"}
    return {"success": False, "message": "Trajectory execution failed"}


def _run_set_gripper(node, params):
    original_mode = node.execution_mode
    node.execution_mode = True
    try:
        ok = node.set_gripper(float(params["position"]))
    finally:
        node.execution_mode = original_mode

    if ok:
        return {"success": True, "message": f"Gripper set to {params['position']}"}
    return {"success": False, "message": "Gripper control failed"}


def _run_grasp(node, params):
    plan_only = params.get("plan_only", False)

    original_mode = node.execution_mode
    if plan_only:
        node.execution_mode = False
    try:
        results = _execute_grasp_core(
            executor=node,
            depth_path=params["depth_path"],
            seg_json_path=params["seg_json_path"],
            affordance_path=params["affordance_path"],
            config=node.config,
            target_object_index=params.get("target_object_index"),
            return_full_trajectories=params.get("return_trajectories", True),
        )
        if results and results[0]["success"]:
            return {"success": True, "message": "Grasp completed", "results": results}
        return {"success": False, "message": "Grasp planning failed"}
    finally:
        node.execution_mode = original_mode


_TASK_DISPATCH = {
    "update_scene": _run_update_scene,
    "move_home": _run_move_home,
    "execute_trajectory": _run_execute_trajectory,
    "set_gripper": _run_set_gripper,
    "grasp": _run_grasp,
}


# ---------------------------------------------------------------------------
#  Worker thread + submit helper
# ---------------------------------------------------------------------------

def _ros_worker():
    print("[Worker] ROS2 worker thread started")
    while True:
        try:
            task = _task_queue.get()
            if task is None:
                break

            task_id = task["task_id"]
            task_type = task["type"]
            params = task["params"]
            print(f"[Worker] Processing {task_id}: {task_type}")

            try:
                handler = _TASK_DISPATCH.get(task_type)
                if handler is None:
                    result = {"success": False, "message": f"Unknown task type: {task_type}"}
                else:
                    result = handler(_node, params)
                _result_queue.put({"task_id": task_id, "result": result})
            except Exception:
                traceback.print_exc()
                _result_queue.put({
                    "task_id": task_id,
                    "result": {"success": False, "message": traceback.format_exc()},
                })
            finally:
                _task_queue.task_done()
        except Exception:
            traceback.print_exc()


def _submit_and_wait(task_type, params, timeout=120):
    task_id = str(uuid.uuid4())
    _task_queue.put({"task_id": task_id, "type": task_type, "params": params})

    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"success": False, "message": "Task timeout"}
        try:
            item = _result_queue.get(timeout=min(remaining, 1.0))
            if item["task_id"] == task_id:
                return item["result"]
            _result_queue.put(item)
        except queue.Empty:
            pass


# ---------------------------------------------------------------------------
#  Initialization / shutdown
# ---------------------------------------------------------------------------

def _shutdown_ros():
    global _node, _task_queue, _result_queue, _ros_exec, _worker_thread, _current_robot

    if _task_queue is not None:
        _task_queue.put(None)
    if _worker_thread is not None:
        _worker_thread.join(timeout=5.0)
        _worker_thread = None

    if _ros_exec is not None:
        _ros_exec.shutdown()
        _ros_exec = None

    if _node is not None:
        _node.destroy_node()
        _node = None

    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass

    _task_queue = None
    _result_queue = None
    _current_robot = None
    time.sleep(1.0)


def init_ros_service(robot_name, execution_mode=True):
    global _node, _task_queue, _result_queue, _ros_exec, _worker_thread
    global _current_robot, _execution_mode

    with _init_lock:
        if _node is not None and _current_robot == robot_name:
            return

        if _node is not None:
            print(f"[Init] Switching robot: {_current_robot} -> {robot_name}")
            _shutdown_ros()

        config_path = f"config/{robot_name}_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        if not rclpy.ok():
            rclpy.init()

        _node = GraspExecutor(
            execution_mode=execution_mode,
            camera_params=config["camera"],
            config=config,
        )
        _execution_mode = execution_mode

        _ros_exec = MultiThreadedExecutor(num_threads=4)
        _ros_exec.add_node(_node)
        threading.Thread(target=_ros_exec.spin, daemon=True).start()

        _task_queue = queue.Queue()
        _result_queue = queue.Queue()
        _worker_thread = threading.Thread(target=_ros_worker, daemon=True)
        _worker_thread.start()

        _current_robot = robot_name
        time.sleep(2.0)
        print(f"[Init] ROS service ready (robot={robot_name}, execution_mode={execution_mode})")


def _ensure_robot(data):
    """If request includes 'robot', ensure the service is initialized for that robot."""
    robot = data.get("robot") if isinstance(data, dict) else None
    if robot and robot != _current_robot:
        init_ros_service(robot_name=robot, execution_mode=_execution_mode)


# ---------------------------------------------------------------------------
#  Flask routes
# ---------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({"success": False, "message": str(e)}), 500


def _check_node():
    if _node is None:
        return jsonify({"success": False, "message": "Service not initialized"}), 503
    return None


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "MoveIt HTTP Service",
        "version": "1.0",
        "robot": _current_robot or "unknown",
        "endpoints": {
            "health": "GET /health",
            "update_scene": "POST /update_scene",
            "grasp": "POST /grasp",
            "forward": "POST /forward (multipart: depth file + JSON fields)",
            "move_home": "POST /move_home",
            "execute_trajectory": "POST /execute_trajectory",
            "set_gripper": "POST /set_gripper",
            "cleanup": "POST /cleanup",
            "remove_object": "POST /remove_object",
            "attach_object": "POST /attach_object",
            "detach_object": "POST /detach_object",
        },
    })


@app.route("/health", methods=["GET"])
def health_check():
    err = _check_node()
    if err:
        return err
    return jsonify({"status": "healthy", "robot": _current_robot})


@app.route("/remove_object", methods=["POST"])
def remove_object():
    err = _check_node()
    if err:
        return err
    data = request.get_json()
    instance_id = data.get("instance_id")
    if not instance_id:
        return jsonify({"success": False, "message": "Missing instance_id"}), 400
    ok = _node.remove_object_from_scene(instance_id)
    return jsonify({"success": ok})


@app.route("/attach_object", methods=["POST"])
def attach_object():
    err = _check_node()
    if err:
        return err
    data = request.get_json()
    instance_id = data.get("instance_id")
    if not instance_id:
        return jsonify({"success": False, "message": "Missing instance_id"}), 400
    ok = _node.attach_object_mesh(instance_id)
    return jsonify({"success": ok})


@app.route("/detach_object", methods=["POST"])
def detach_object():
    err = _check_node()
    if err:
        return err
    ok = _node.detach_object()
    return jsonify({"success": ok})


@app.route("/cleanup", methods=["POST"])
def cleanup_scene():
    err = _check_node()
    if err:
        return err
    data = request.get_json() or {}
    basket_id = data.get("basket_id", "basket_1")

    _node.detach_object()
    time.sleep(0.3)
    _node.clear_pointcloud_obstacles()
    _node.remove_basket_from_scene(basket_id)
    _node.clear_octomap()
    time.sleep(0.5)
    return jsonify({"success": True, "message": "Scene cleanup done"})


@app.route("/update_scene", methods=["POST"])
def update_scene():
    data = request.get_json()
    _ensure_robot(data)
    err = _check_node()
    if err:
        return err
    if not data.get("depth_path") or not data.get("seg_json_path"):
        return jsonify({"success": False, "message": "Missing depth_path or seg_json_path"}), 400

    result = _submit_and_wait("update_scene", {
        "depth_path": data["depth_path"],
        "seg_json_path": data["seg_json_path"],
    }, timeout=30)
    return jsonify(result), 200 if result.get("success") else 500


@app.route("/move_home", methods=["POST"])
def move_home():
    data = request.get_json() or {}
    _ensure_robot(data)
    err = _check_node()
    if err:
        return err
    result = _submit_and_wait("move_home", {}, timeout=120)
    return jsonify(result), 200 if result.get("success") else 500


@app.route("/execute_trajectory", methods=["POST"])
def execute_trajectory():
    err = _check_node()
    if err:
        return err
    data = request.get_json()
    traj_json = data.get("trajectory")
    if not traj_json or "joint_names" not in traj_json or "points" not in traj_json:
        return jsonify({"success": False, "message": "Missing or invalid trajectory"}), 400

    result = _submit_and_wait("execute_trajectory", {"trajectory_json": traj_json}, timeout=180)
    return jsonify(result), 200 if result.get("success") else 500


@app.route("/set_gripper", methods=["POST"])
def set_gripper():
    err = _check_node()
    if err:
        return err
    data = request.get_json()
    position = data.get("position")
    if position is None:
        return jsonify({"success": False, "message": "Missing position"}), 400

    result = _submit_and_wait("set_gripper", {"position": float(position)}, timeout=30)
    return jsonify(result), 200 if result.get("success") else 500


@app.route("/grasp", methods=["POST"])
def execute_grasp():
    data = request.get_json()
    _ensure_robot(data)
    err = _check_node()
    if err:
        return err

    depth_path = data.get("depth_path")
    seg_json_path = data.get("seg_json_path")
    affordance_path = data.get("affordance_path")
    if not all([depth_path, seg_json_path, affordance_path]):
        return jsonify({"success": False, "message": "Missing depth_path, seg_json_path, or affordance_path"}), 400

    result = _submit_and_wait("grasp", {
        "depth_path": depth_path,
        "seg_json_path": seg_json_path,
        "affordance_path": affordance_path,
        "target_object_index": data.get("target_object_index"),
        "plan_only": data.get("plan_only", False),
        "return_trajectories": data.get("return_trajectories", True),
    }, timeout=1800)
    return jsonify(result), 200 if result.get("success") else 500


@app.route("/forward", methods=["POST"])
def forward():
    """Accept uploaded depth image + JSON strings, run grasp planning."""
    # --- parse multipart form data ---
    if "depth" not in request.files:
        return jsonify({"success": False, "message": "Missing 'depth' file upload"}), 400

    seg_json_str = request.form.get("seg_json")
    affordance_json_str = request.form.get("affordance_json")
    if not seg_json_str or not affordance_json_str:
        return jsonify({"success": False, "message": "Missing seg_json or affordance_json"}), 400

    robot_name = request.form.get("robot_name", "xarm7")
    target_object_index_str = request.form.get("target_object_index", "-1")
    try:
        target_object_index = int(target_object_index_str)
    except (ValueError, TypeError):
        target_object_index = -1
    if target_object_index < 0:
        target_object_index = None

    # --- ensure robot is initialized ---
    _ensure_robot({"robot": robot_name})
    err = _check_node()
    if err:
        return err

    # --- write uploads to temporary files ---
    tmp_files = []
    try:
        depth_file = request.files["depth"]
        tmp_depth = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        depth_file.save(tmp_depth)
        tmp_depth.close()
        tmp_files.append(tmp_depth.name)

        tmp_seg = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
        tmp_seg.write(seg_json_str)
        tmp_seg.close()
        tmp_files.append(tmp_seg.name)

        tmp_aff = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
        tmp_aff.write(affordance_json_str)
        tmp_aff.close()
        tmp_files.append(tmp_aff.name)

        # --- run grasp planning ---
        result = _submit_and_wait("grasp", {
            "depth_path": tmp_depth.name,
            "seg_json_path": tmp_seg.name,
            "affordance_path": tmp_aff.name,
            "target_object_index": target_object_index,
            "plan_only": True,
            "return_trajectories": True,
        }, timeout=1800)

        # --- cleanup scene ---
        try:
            _node.detach_object()
            time.sleep(0.3)
            _node.clear_pointcloud_obstacles()
            _node.remove_basket_from_scene("basket_1")
            _node.clear_octomap()
            time.sleep(0.5)
        except Exception:
            traceback.print_exc()

        return jsonify(result), 200 if result.get("success") else 500
    finally:
        for path in tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def start_service(host="0.0.0.0", port=8000, robot_name="xarm7", execution_mode=True):
    init_ros_service(robot_name=robot_name, execution_mode=execution_mode)

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
                    print(f"Port {actual_port} in use, trying next...")
    else:
        print(f"Ports {port}-{port + 9} all in use")
        return

    if actual_port != port:
        print(f"Port {port} in use, using {actual_port}")

    print(f"HTTP service at http://{host}:{actual_port}")
    app.run(host=host, port=actual_port, debug=False, threaded=True)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 14086
    robot_name = sys.argv[2] if len(sys.argv) > 2 else "xarm7"
    execution_mode = False

    print(f"Starting MoveIt service...")
    print(f"  Robot: {robot_name}")
    print(f"  Port: {port}")
    print(f"  Execution mode: {execution_mode}")

    start_service(port=port, robot_name=robot_name, execution_mode=execution_mode)
