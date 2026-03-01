#!/usr/bin/env python3
"""MoveIt Grasp Server — GET /, GET /health, POST /predict, POST /grasp_plan."""

import argparse, base64, json, os, sys, tempfile, time, threading, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from flask import Flask, request, jsonify, abort, render_template
import rclpy
from rclpy.executors import MultiThreadedExecutor
from moveit_grasp_core import GraspExecutor, _execute_grasp_core, format_grasp_result
import grasp_pipeline

app = Flask(__name__)
_node: GraspExecutor = None
_robot: str = None
_exec_mode: bool = False
_lock = threading.Lock()
_grasp_server_url: str = None
_grasp_server_type: str = "auto"  # "auto", "generate", "predict"
_port: int = 14086

def _init(robot, exec_mode=False):
    global _node, _robot, _exec_mode
    with _lock:
        if _node and _robot == robot: return
        if _node: _node.destroy_node()
        if not rclpy.ok(): rclpy.init()
        with open(f"config/{robot}_config.json") as f: cfg = json.load(f)
        _node = GraspExecutor(execution_mode=exec_mode, camera_params=cfg["camera"], config=cfg)
        _exec_mode = exec_mode
        ex = MultiThreadedExecutor(num_threads=4); ex.add_node(_node)
        threading.Thread(target=ex.spin, daemon=True).start()
        _robot = robot; time.sleep(2.0)
        print(f"[Init] Ready  robot={robot}  exec={exec_mode}")

def _cleanup_scene(remove_basket=False):
    """Detach objects, clear obstacles/octomap, optionally remove basket."""
    _node.detach_object(); time.sleep(0.3)
    _node.clear_pointcloud_obstacles()
    if remove_basket:
        try: _node.remove_basket_from_scene("basket_1")
        except Exception: pass
    _node.clear_octomap(); time.sleep(0.5)

def _safe_num(val, type_fn=float, default=None):
    """Convert val to number, returning default on failure."""
    if val is None:
        return default
    try:
        return type_fn(val)
    except (ValueError, TypeError):
        return default

def _plan_single_object(depth_path, seg_json_path, affordance_path,
                        target_object_index=None, camera=None,
                        start_pos=None, end_pos=None, target_pos=None,
                        servo_dt=None, pos_tol=None, ori_tol=None):
    """Cleanup scene, run grasp core, cleanup again, return formatted result + raw."""
    _cleanup_scene()
    results = _execute_grasp_core(
        executor=_node, depth_path=depth_path,
        seg_json_path=seg_json_path, affordance_path=affordance_path,
        config=_node.config, target_object_index=target_object_index,
        return_full_trajectories=True, camera=camera,
        start_pos=start_pos, end_pos=end_pos, target_pos=target_pos,
        pos_tol=pos_tol, ori_tol=ori_tol)
    _cleanup_scene(remove_basket=True)
    return format_grasp_result(results, dt=servo_dt), results

def _parse():
    ct = request.content_type or ""
    if "multipart" in ct:
        if "dpt" not in request.files: abort(400, "Missing 'dpt'")
        objs_s, seg_s = request.form.get("objs"), request.form.get("seg_json")
        if not objs_s or not seg_s: abort(400, "Missing 'objs'/'seg_json'")
        for k in ("robot_name", "camera"):
            if not request.form.get(k): abort(400, f"Missing '{k}'")
        tmp = []
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        request.files["dpt"].save(f); f.close(); tmp.append(f.name)
        for s in (seg_s, objs_s):
            f = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
            f.write(s); f.close(); tmp.append(f.name)
        if "rgb" in request.files:
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            request.files["rgb"].save(f); f.close(); tmp.append(f.name)
        tgt = _safe_num(request.form.get("target_object_index", "-1"), int, -1)
        sim = request.form.get("execution_simulation", "true")
        servo_dt = _safe_num(request.form.get("servo_dt", "0"), float, 0)
        pos_tol = _safe_num(request.form.get("pos_tol"), float)
        ori_tol = _safe_num(request.form.get("ori_tol"), float)
        parsed = {"depth_path": tmp[0], "seg_json_path": tmp[1], "affordance_path": tmp[2],
                "robot": request.form["robot_name"],
                "target_object_index": None if tgt < 0 else tgt,
                "plan_only": sim.lower() not in ("false","0","no"),
                "servo_dt": servo_dt or None,
                "pos_tol": pos_tol, "ori_tol": ori_tol,
                "camera": json.loads(request.form["camera"])}
        for k in ("start_pos", "target_pos", "end_pos"):
            v = request.form.get(k)
            parsed[k] = json.loads(v) if v else None
        return parsed, tmp
    d = request.get_json(silent=True)
    if not d: abort(400, "Need JSON or multipart")
    for k in ("robot_name", "dpt", "objs", "seg_json", "camera"):
        if not d.get(k): abort(400, f"Missing {k}")
    tmp = []
    def _resolve(val, suffix):
        if isinstance(val, str) and os.path.exists(val): return val
        f = tempfile.NamedTemporaryFile(suffix=suffix, mode="wb" if suffix == ".png" else "w", delete=False)
        if suffix == ".png":
            f.write(base64.b64decode(val) if isinstance(val, str) else val)
        else:
            f.write(json.dumps(val) if not isinstance(val, str) else val)
        f.close(); tmp.append(f.name); return f.name
    tgt = d.get("target_object_index")
    return {"depth_path": _resolve(d["dpt"], ".png"),
            "seg_json_path": _resolve(d["seg_json"], ".json"),
            "affordance_path": _resolve(d["objs"], ".json"),
            "robot": d["robot_name"],
            "target_object_index": None if isinstance(tgt, int) and tgt < 0 else tgt,
            "plan_only": d.get("execution_simulation", True),
            "servo_dt": d.get("servo_dt"),
            "pos_tol": d.get("pos_tol"), "ori_tol": d.get("ori_tol"),
            "camera": d["camera"],
            "start_pos": d.get("start_pos"),
            "target_pos": d.get("target_pos"),
            "end_pos": d.get("end_pos")}, tmp

# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exc(e):
    traceback.print_exc()
    return jsonify({"success": False, "message": str(e)}), getattr(e, "code", 500)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    info = {
        "service": "MoveIt Grasp Server",
        "status": "healthy" if _node else "not_initialized",
        "robot": _robot or "unknown",
        "grasp_server": _grasp_server_url or "not_configured",
        "grasp_server_type": _grasp_server_type,
    }
    return jsonify(info)

@app.route("/predict", methods=["POST"])
def predict():
    params, tmp = _parse()
    try:
        r = params.pop("robot")
        servo_dt = params.pop("servo_dt", None)
        if r != _robot: _init(r, _exec_mode)
        if not _node: abort(503, "Not initialized")
        orig = _node.execution_mode
        plan_only = params.get("plan_only", True)
        _node.execution_mode = not plan_only
        print(f"[predict] plan_only={plan_only}, execution_mode={_node.execution_mode} (was {orig})")
        try:
            result, _ = _plan_single_object(
                params["depth_path"], params["seg_json_path"], params["affordance_path"],
                target_object_index=params.get("target_object_index"),
                camera=params.get("camera"),
                start_pos=params.get("start_pos"), end_pos=params.get("end_pos"),
                target_pos=params.get("target_pos"),
                servo_dt=servo_dt,
                pos_tol=params.get("pos_tol"), ori_tol=params.get("ori_tol"))
        finally:
            _node.execution_mode = orig
        return jsonify(result), 200 if result.get("success") else 500
    finally:
        for p in tmp:
            try: os.unlink(p)
            except OSError: pass

@app.route("/grasp_plan", methods=["POST"])
def grasp_plan():
    """Full pipeline: grasp3d detection -> MoveIt planning for top N objects."""
    if not _grasp_server_url:
        abort(400, "--grasp-server not configured")
    if "rgb" not in request.files:
        abort(400, "Missing 'rgb' file upload")
    if "depth" not in request.files:
        abort(400, "Missing 'depth' file upload")
    if not _node:
        abort(503, "Not initialized")
    rgb_bytes = request.files["rgb"].read()
    request.files["depth"].seek(0)
    depth_bytes = request.files["depth"].read()
    text_prompt = request.form.get("text_prompt", "object")
    num_objects = max(1, min(10, _safe_num(request.form.get("num_objects", "3"), int, 3)))
    result = grasp_pipeline.run_grasp_pipeline(
        rgb_bytes=rgb_bytes, depth_bytes=depth_bytes,
        text_prompt=text_prompt, num_objects=num_objects,
        config=_node.config, robot_name=_robot,
        grasp_server_url=_grasp_server_url,
        predict_url=f"http://127.0.0.1:{_port}/predict",
        grasp_server_type=_grasp_server_type)
    return jsonify(result)


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--port", type=int, default=14086)
    pa.add_argument("--host", default="0.0.0.0")
    pa.add_argument("--robot", default="xarm7")
    pa.add_argument("--execution-mode", action="store_true")
    pa.add_argument("--grasp-server", default=None,
                    help="Grasp3D server URL (e.g. http://10.0.73.5:10003)")
    pa.add_argument("--grasp-server-type", default="auto",
                    choices=["auto", "generate", "predict"],
                    help="Grasp server API type: generate (/generate), predict (/api/predict), auto (detect)")
    pa.add_argument("--test", action="store_true")
    a = pa.parse_args()

    _grasp_server_url = a.grasp_server
    _grasp_server_type = a.grasp_server_type
    _port = a.port
    print(f"MoveIt Grasp Server  robot={a.robot}  port={a.port}  exec={a.execution_mode}"
          f"  grasp_server={_grasp_server_url or 'none'}  type={_grasp_server_type}")
    _init(a.robot, a.execution_mode)

    if a.test:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        P = {k: os.path.join(d, v) for k, v in [
            ("dpt","grasp-wrist-dpt_opt.png"),("objs","affordance.json"),("seg","rgb_detection_wrist.json")]}
        for p in P.values():
            if not os.path.exists(p): print(f"Missing: {p}"); sys.exit(1)
        CAM = {"intrinsics":{"fx":909.665,"fy":909.533,"cx":636.739,"cy":376.35},
               "extrinsics":{"translation":[.325,.028,.658],"quaternion":[-.703,.71,-.026,.021]}}
        PS = {"start_pos":[.27,0,.307,-3.14159,0,0],"target_pos":[.4,-.55,.1,-3.14159,0,0],
              "end_pos":[.27,0,.307,-3.14159,0,0]}
        with app.test_client() as c:
            print("\n--- GET /health ---")
            r = c.get("/health"); print(f"  {r.status_code} {r.get_json()}")
            print("\n--- POST /predict (JSON) ---")
            r = c.post("/predict", content_type="application/json", data=json.dumps({
                "dpt":P["dpt"],"objs":P["objs"],"seg_json":P["seg"],
                "robot_name":a.robot,"camera":CAM,"servo_dt":0.02,**PS}))
            print(f"  {r.status_code} {json.dumps(r.get_json(),indent=2,ensure_ascii=False)}")
        print("Done.")
    else:
        app.run(host=a.host, port=a.port, debug=False, threaded=True)
