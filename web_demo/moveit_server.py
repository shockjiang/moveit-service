#!/usr/bin/env python3
"""MoveIt Grasp Server — only GET / and POST /predict."""

import argparse, base64, io, json, os, sys, tempfile, time, threading, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from flask import Flask, request, jsonify, abort
from werkzeug.datastructures import FileStorage
import rclpy
from rclpy.executors import MultiThreadedExecutor
from moveit_grasp_core import GraspExecutor, _execute_grasp_core, format_grasp_result

app = Flask(__name__)
_node: GraspExecutor = None
_robot: str = None
_exec_mode: bool = False
_lock = threading.Lock()

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

def _parse():
    ct = request.content_type or ""
    if "multipart" in ct:
        if "dpt" not in request.files: abort(400, "Missing 'dpt'")
        objs_s, seg_s = request.form.get("objs"), request.form.get("seg_json")
        if not objs_s or not seg_s: abort(400, "Missing 'objs'/'seg_json'")
        for k in ("robot_name","start_pos","target_pos","end_pos","camera"):
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
        try:    tgt = int(request.form.get("target_object_index", "-1"))
        except: tgt = -1
        sim = request.form.get("execution_simulation", "true")
        try:    servo_dt = float(request.form.get("servo_dt", "0"))
        except: servo_dt = 0
        return {"depth_path": tmp[0], "seg_json_path": tmp[1], "affordance_path": tmp[2],
                "robot": request.form["robot_name"],
                "target_object_index": None if tgt < 0 else tgt,
                "plan_only": sim.lower() not in ("false","0","no"),
                "servo_dt": servo_dt or None,
                **{k: json.loads(request.form[k]) for k in ("camera","start_pos","target_pos","end_pos")}}, tmp
    d = request.get_json(silent=True)
    if not d: abort(400, "Need JSON or multipart")
    for k in ("robot_name","dpt","objs","seg_json","start_pos","target_pos","end_pos","camera"):
        if not d.get(k): abort(400, f"Missing {k}")
    # dpt/objs/seg_json: string→路径, 否则→写临时文件
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
            "camera": d["camera"], "start_pos": d["start_pos"],
            "target_pos": d["target_pos"], "end_pos": d["end_pos"]}, tmp

@app.errorhandler(Exception)
def handle_exc(e):
    traceback.print_exc()
    return jsonify({"success": False, "message": str(e)}), getattr(e, "code", 500)

@app.route("/")
def index():
    return jsonify({"service": "MoveIt Grasp Server",
                    "status": "healthy" if _node else "not_initialized",
                    "robot": _robot or "unknown"})

@app.route("/predict", methods=["POST"])
def predict():
    params, tmp = _parse()
    try:
        r = params.pop("robot")
        servo_dt = params.pop("servo_dt", None)
        if r != _robot: _init(r, _exec_mode)
        if not _node: abort(503, "Not initialized")
        # Pre-cleanup (sleep 等待 planning scene 同步)
        _node.detach_object(); time.sleep(0.3)
        _node.clear_pointcloud_obstacles()
        _node.clear_octomap(); time.sleep(0.5)
        # Call core directly — params passed in, core handles priority
        orig = _node.execution_mode
        if params.get("plan_only"): _node.execution_mode = False
        try:
            results = _execute_grasp_core(
                executor=_node, depth_path=params["depth_path"],
                seg_json_path=params["seg_json_path"], affordance_path=params["affordance_path"],
                config=_node.config, target_object_index=params.get("target_object_index"),
                return_full_trajectories=True,
                camera=params.get("camera"), start_pos=params.get("start_pos"),
                end_pos=params.get("end_pos"), target_pos=params.get("target_pos"))
        finally:
            _node.execution_mode = orig
        # Post-cleanup
        _node.detach_object(); time.sleep(0.3)
        _node.clear_pointcloud_obstacles()
        _node.remove_basket_from_scene("basket_1")
        _node.clear_octomap(); time.sleep(0.5)
        # Format — servo_dt 不为 None 时做插值重采样
        result = format_grasp_result(results, dt=servo_dt)
        return jsonify(result), 200 if result.get("success") else 500
    finally:
        for p in tmp:
            try: os.unlink(p)
            except OSError: pass

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--port", type=int, default=14086)
    pa.add_argument("--host", default="0.0.0.0")
    pa.add_argument("--robot", default="xarm7")
    pa.add_argument("--execution-mode", action="store_true")
    pa.add_argument("--test", action="store_true")
    a = pa.parse_args()

    print(f"MoveIt Grasp Server  robot={a.robot}  port={a.port}  exec={a.execution_mode}")
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
            print("\n--- GET / ---")
            r = c.get("/"); print(f"  {r.status_code} {r.get_json()}")
            print("\n--- POST /predict (JSON) ---")
            r = c.post("/predict", content_type="application/json", data=json.dumps({
                "dpt":P["dpt"],"objs":P["objs"],"seg_json":P["seg"],
                "robot_name":a.robot,"camera":CAM,"servo_dt":0.02,**PS}))
            print(f"  {r.status_code} {json.dumps(r.get_json(),indent=2,ensure_ascii=False)}")
        print("Done.")
    else:
        app.run(host=a.host, port=a.port, debug=False, threaded=True)
