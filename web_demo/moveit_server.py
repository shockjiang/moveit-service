#!/usr/bin/env python3
"""MoveIt Grasp Server — GET /, GET /health, POST /predict, POST /grasp_plan."""

import argparse, base64, io, json, os, sys, tempfile, time, threading, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from flask import Flask, request, jsonify, abort, render_template
from werkzeug.datastructures import FileStorage
import requests as http_requests
import numpy as np
import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from moveit_grasp_core import GraspExecutor, _execute_grasp_core, format_grasp_result

app = Flask(__name__)
_node: GraspExecutor = None
_robot: str = None
_exec_mode: bool = False
_lock = threading.Lock()
_grasp_server_url: str = None
_grasp_server_type: str = "auto"  # "auto", "generate", "predict"

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
        try:    pos_tol = float(request.form["pos_tol"]) if request.form.get("pos_tol") else None
        except: pos_tol = None
        try:    ori_tol = float(request.form["ori_tol"]) if request.form.get("ori_tol") else None
        except: ori_tol = None
        return {"depth_path": tmp[0], "seg_json_path": tmp[1], "affordance_path": tmp[2],
                "robot": request.form["robot_name"],
                "target_object_index": None if tgt < 0 else tgt,
                "plan_only": sim.lower() not in ("false","0","no"),
                "servo_dt": servo_dt or None,
                "pos_tol": pos_tol, "ori_tol": ori_tol,
                **{k: json.loads(request.form[k]) for k in ("camera","start_pos","target_pos","end_pos")}}, tmp
    d = request.get_json(silent=True)
    if not d: abort(400, "Need JSON or multipart")
    for k in ("robot_name","dpt","objs","seg_json","start_pos","target_pos","end_pos","camera"):
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
            "camera": d["camera"], "start_pos": d["start_pos"],
            "target_pos": d["target_pos"], "end_pos": d["end_pos"]}, tmp

# ---------------------------------------------------------------------------
#  Grasp3D helpers
# ---------------------------------------------------------------------------

def _detect_grasp_server_type():
    """Auto-detect grasp server type by probing /api/health first."""
    global _grasp_server_type
    if _grasp_server_type != "auto":
        return _grasp_server_type
    try:
        r = http_requests.get(f"{_grasp_server_url}/api/health", timeout=5)
        if r.status_code == 200:
            _grasp_server_type = "predict"
            print(f"[grasp] Auto-detected server type: predict (/api/predict)")
            return "predict"
    except Exception:
        pass
    _grasp_server_type = "generate"
    print(f"[grasp] Auto-detected server type: generate (/generate)")
    return "generate"


def _call_grasp3d(rgb_bytes, text_prompt, depth_bytes=None):
    """Call grasp3d server, supporting both /generate and /api/predict endpoints.

    /generate (server_grasp.py):
      - Input: img_source (file), text_prompt (form)
      - Output: [[{dt_score, dt_bbox, dt_mask, affs, scores}, ...]]
      - Masks at 384x384, no category filtering

    /api/predict (web_demo3d/server.py):
      - Input: images (file), depth_images (file), text_prompt, min_score, chosen_policy
      - Output: {success, results: [{objects: [{bbox, score, category, mask, affs, scores, aph}]}]}
      - Masks at original resolution, supports text-based category filtering
    """
    stype = _detect_grasp_server_type()

    if stype == "predict":
        return _call_grasp3d_predict(rgb_bytes, text_prompt, depth_bytes)
    else:
        return _call_grasp3d_generate(rgb_bytes, text_prompt)


def _call_grasp3d_generate(rgb_bytes, text_prompt):
    """Call /generate endpoint (server_grasp.py). Returns unified object list."""
    resp = http_requests.post(
        f"{_grasp_server_url}/generate",
        files={"img_source": ("rgb.png", rgb_bytes, "image/png")},
        data={"text_prompt": text_prompt},
        timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # unwrap batch dim: data = [[obj0, obj1, ...]]
    if isinstance(data, list) and len(data) > 0:
        objects = data[0] if isinstance(data[0], list) else data
    else:
        objects = data if isinstance(data, list) else []
    # Tag source so downstream knows mask format
    for obj in objects:
        obj["_source"] = "generate"
    return objects


def _call_grasp3d_predict(rgb_bytes, text_prompt, depth_bytes=None):
    """Call /api/predict endpoint (web_demo3d/server.py). Returns unified object list.
    This endpoint supports text-conditioned category filtering and depth input.
    """
    files = [("images", ("rgb.png", rgb_bytes, "image/png"))]
    if depth_bytes:
        files.append(("depth_images", ("depth.png", depth_bytes, "image/png")))
    data = {
        "text_prompt": text_prompt,
        "min_score": "0.25",
        "iou_threshold": "0.5",
        "chosen_policy": "det,grasp",
    }
    resp = http_requests.post(
        f"{_grasp_server_url}/api/predict",
        files=files, data=data, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"Grasp3d predict failed: {payload.get('error', 'unknown')}")
    results = payload.get("results", [])
    if not results:
        return []
    # Convert /api/predict format -> unified format matching /generate keys
    objects = []
    for obj in results[0].get("objects", []):
        objects.append({
            "dt_score": obj.get("score", 0.0),
            "dt_bbox": obj.get("bbox", [0, 0, 1, 1]),
            "dt_mask": obj.get("mask"),  # RLE at original resolution
            "affs": obj.get("affs", []),
            "scores": obj.get("scores", []),
            "touching_points": obj.get("touching_points", []),
            "aph": obj.get("aph"),
            "category": obj.get("category", "object"),
            "_source": "predict",
        })
    return objects

def _resize_rle_mask(rle, target_h, target_w):
    """Decode RLE mask, resize to target dimensions, re-encode as RLE."""
    from pycocotools import mask as coco_mask
    if not rle or not isinstance(rle, dict):
        return rle
    # Ensure counts is bytes for pycocotools
    rle_input = dict(rle)
    if isinstance(rle_input.get("counts"), str):
        rle_input["counts"] = rle_input["counts"].encode("utf-8")
    mask = coco_mask.decode(rle_input)  # (h, w) uint8
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return rle
    mask_resized = cv2.resize(mask, (target_w, target_h),
                              interpolation=cv2.INTER_NEAREST)
    rle_out = coco_mask.encode(np.asfortranarray(mask_resized.astype(np.uint8)))
    rle_out["counts"] = rle_out["counts"].decode("utf-8")
    return rle_out

def _grasp3d_to_seg_json(objects, text_prompt, img_h, img_w):
    """Convert grasp3d objects to SceneManager seg_json format.

    /generate source: dt_bbox is pixel coords, dt_mask is RLE at 384x384.
    /predict source:  dt_bbox is normalized [0-1], dt_mask is RLE at original res.
    SceneManager expects: score, bbox (pixel), mask (RLE at original resolution).
    """
    converted = []
    for obj in objects:
        source = obj.get("_source", "generate")
        bbox = list(obj.get("dt_bbox", [0, 0, 1, 1]))

        if source == "predict":
            # /api/predict returns normalized bboxes -> convert to pixel coords
            bbox = [bbox[0] * img_w, bbox[1] * img_h,
                    bbox[2] * img_w, bbox[3] * img_h]
            mask = obj.get("dt_mask")  # already at original resolution
        else:
            # /generate returns pixel bboxes + 384x384 masks
            mask = _resize_rle_mask(obj.get("dt_mask"), img_h, img_w)

        converted.append({
            "score": obj.get("dt_score", 0.0),
            "bbox": bbox,
            "mask": mask,
            "category": obj.get("category", "object"),
        })
    return {"results": [{"text_prompt": text_prompt, "objects": converted}]}

def _project_3d_to_2d(point_3d, camera):
    """Project a 3D base-frame point back to 2D pixel coords.
    Inverse of GraspExecutor.transform_to_base():
        pt_cam = R_cam2base.T @ (pt_base - t_cam2base)
        u = pt_cam[0] * fx / pt_cam[2] + cx
        v = pt_cam[1] * fy / pt_cam[2] + cy
    Returns (u, v) as ints or None if behind camera.
    """
    from scipy.spatial.transform import Rotation
    ext = camera["extrinsics"]
    intr = camera["intrinsics"]
    t = np.array(ext["translation"])
    q = ext["quaternion"]
    R = Rotation.from_quat(q).as_matrix()
    pt_base = np.array(point_3d)
    pt_cam = R.T @ (pt_base - t)
    if pt_cam[2] <= 0:
        return None
    u = int(pt_cam[0] * intr["fx"] / pt_cam[2] + intr["cx"])
    v = int(pt_cam[1] * intr["fy"] / pt_cam[2] + intr["cy"])
    return (u, v)

# Distinct colors for objects
_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (255, 0, 128),
]

def _generate_pointcloud(rgb_img, depth_path, camera, downsample=5):
    """Generate a colored 3D point cloud from RGB + depth using camera params."""
    from scipy.spatial.transform import Rotation
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return {"x": [], "y": [], "z": [], "colors": []}
    intr = camera["intrinsics"]
    ext = camera["extrinsics"]
    depth_scale = camera.get("depth_scale", 0.001)
    max_range = camera.get("max_range_m", 3.0)
    R = Rotation.from_quat(ext["quaternion"]).as_matrix()
    t = np.array(ext["translation"])

    h, w = depth_img.shape[:2]
    xs, ys, zs, colors = [], [], [], []
    for v in range(0, h, downsample):
        for u in range(0, w, downsample):
            d = depth_img[v, u] * depth_scale
            if d <= 0 or d > max_range:
                continue
            pt_cam = np.array([(u - intr["cx"]) * d / intr["fx"],
                               (v - intr["cy"]) * d / intr["fy"], d])
            pt_base = R @ pt_cam + t
            xs.append(round(float(pt_base[0]), 4))
            ys.append(round(float(pt_base[1]), 4))
            zs.append(round(float(pt_base[2]), 4))
            b, g, r = rgb_img[v, u]  # BGR
            colors.append(f"rgb({r},{g},{b})")

    return {"x": xs, "y": ys, "z": zs, "colors": colors}


def _build_key_points_3d(plan_results, config):
    """Extract 3D key points (home, basket, per-object grasp/pregrasp) for trajectory viz."""
    home = config.get("home", {}).get("position")
    basket = config.get("basket", {}).get("center")
    per_object = []
    for pr in plan_results:
        if not pr.get("success"):
            continue
        per_object.append({
            "instance_id": pr.get("instance_id"),
            "grasp": pr.get("grasp_center_3d"),
            "pregrasp": pr.get("pregrasp_center_3d"),
        })
    return {"home": home, "basket": basket, "per_object": per_object}


def _draw_visualization(rgb_img, objects, plan_results, camera, config):
    """Draw bboxes, affordances, grasp points, and trajectory waypoints on RGB."""
    vis = rgb_img.copy()
    h, w = vis.shape[:2]
    planned_ids = set()
    result_by_id = {}
    for pr in plan_results:
        iid = pr.get("instance_id", "")
        planned_ids.add(iid)
        result_by_id[iid] = pr

    # Draw bboxes for all detected objects
    for i, obj in enumerate(objects):
        color = _COLORS[i % len(_COLORS)]
        bbox = obj.get("dt_bbox", [])
        source = obj.get("_source", "generate")
        iid = f"obj_{i}"
        thickness = 3 if iid in planned_ids else 1
        if len(bbox) == 4:
            if source == "predict":
                # Normalized bbox -> pixel coords
                x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
            else:
                x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cat = obj.get("category", "")
            cat_str = f" {cat}" if cat and cat != "object" else ""
            label = f"{iid}{cat_str} ({obj.get('dt_score', 0):.2f})"
            cv2.putText(vis, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

    # Draw affordance rects + projected key points for planned objects
    basket_center = config.get("basket", {}).get("center")
    home_pos = config.get("home", {}).get("position")
    for pr in plan_results:
        iid = pr.get("instance_id", "")
        try:
            idx = int(iid.split("_")[-1])
        except (ValueError, IndexError):
            continue
        color = _COLORS[idx % len(_COLORS)]

        # Draw affordance rotated rect from grasp3d affs: [cx, cy, w, h, angle]
        if idx < len(objects):
            obj = objects[idx]
            affs = obj.get("affs", [])
            if affs and isinstance(affs[0], (list, tuple)) and len(affs[0]) >= 5:
                best_aff = affs[0]
                cx, cy = best_aff[0], best_aff[1]
                rw, rh = best_aff[2], best_aff[3]
                angle = float(best_aff[4]) * 180.0 / 3.14159  # rad -> deg
                box = cv2.boxPoints(((cx, cy), (rw, rh), angle))
                box = np.int32(box)
                cv2.drawContours(vis, [box], 0, color, 2)

        # Project key 3D points
        key_points = []
        grasp_3d = pr.get("grasp_center_3d")
        pregrasp_3d = pr.get("pregrasp_center_3d")
        if grasp_3d:
            key_points.append(("G", grasp_3d))
        if pregrasp_3d:
            key_points.append(("P", pregrasp_3d))
        if basket_center:
            key_points.append(("B", basket_center))
        if home_pos:
            key_points.append(("H", home_pos))

        projected = []
        for label, pt3d in key_points:
            px = _project_3d_to_2d(pt3d, camera)
            if px and 0 <= px[0] < w and 0 <= px[1] < h:
                projected.append((label, px))
                cv2.circle(vis, px, 6, color, -1)
                cv2.putText(vis, label, (px[0] + 8, px[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw arrows between consecutive key points
        for j in range(len(projected) - 1):
            cv2.arrowedLine(vis, projected[j][1], projected[j + 1][1],
                            color, 1, cv2.LINE_AA, tipLength=0.03)

    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


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
        _node.detach_object(); time.sleep(0.3)
        _node.clear_pointcloud_obstacles()
        _node.clear_octomap(); time.sleep(0.5)
        orig = _node.execution_mode
        if params.get("plan_only"): _node.execution_mode = False
        try:
            results = _execute_grasp_core(
                executor=_node, depth_path=params["depth_path"],
                seg_json_path=params["seg_json_path"], affordance_path=params["affordance_path"],
                config=_node.config, target_object_index=params.get("target_object_index"),
                return_full_trajectories=True,
                camera=params.get("camera"), start_pos=params.get("start_pos"),
                end_pos=params.get("end_pos"), target_pos=params.get("target_pos"),
                pos_tol=params.get("pos_tol"), ori_tol=params.get("ori_tol"))
        finally:
            _node.execution_mode = orig
        _node.detach_object(); time.sleep(0.3)
        _node.clear_pointcloud_obstacles()
        _node.remove_basket_from_scene("basket_1")
        _node.clear_octomap(); time.sleep(0.5)
        result = format_grasp_result(results, dt=servo_dt)
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

    text_prompt = request.form.get("text_prompt", "object")
    try:
        num_objects = int(request.form.get("num_objects", "3"))
    except ValueError:
        num_objects = 3
    num_objects = max(1, min(10, num_objects))

    tmp = []
    try:
        # Save depth to temp file
        depth_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        request.files["depth"].save(depth_tmp)
        depth_tmp.close()
        tmp.append(depth_tmp.name)

        # Read RGB bytes
        rgb_bytes = request.files["rgb"].read()
        request.files["rgb"].seek(0)

        # Read depth bytes for predict-mode grasp server
        request.files["depth"].seek(0)
        depth_bytes = request.files["depth"].read()
        request.files["depth"].seek(0)

        # Decode RGB image for visualization later
        rgb_arr = np.frombuffer(rgb_bytes, np.uint8)
        rgb_img = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
        if rgb_img is None:
            abort(400, "Could not decode RGB image")

        # 1. Call grasp3d detection (depth_bytes passed for /api/predict mode)
        try:
            objects = _call_grasp3d(rgb_bytes, text_prompt, depth_bytes=depth_bytes)
        except http_requests.ConnectionError:
            abort(502, f"Cannot reach grasp3d server at {_grasp_server_url}")
        except http_requests.HTTPError as e:
            abort(502, f"Grasp3d server error: {e}")

        if not objects:
            abort(400, "No objects detected by grasp3d")

        # 2. Sort by dt_score descending, take top N
        objects.sort(key=lambda o: o.get("dt_score", 0), reverse=True)
        top_objects = objects[:num_objects]

        # 3. Build seg_json + affordance temp files
        img_h, img_w = rgb_img.shape[:2]
        seg_json = _grasp3d_to_seg_json(objects, text_prompt, img_h, img_w)
        seg_tmp = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
        json.dump(seg_json, seg_tmp)
        seg_tmp.close()
        tmp.append(seg_tmp.name)

        aff_tmp = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
        # affordance file: list of objects with affs/scores from grasp3d
        affordance_data = []
        for obj in objects:
            affordance_data.append({
                "affs": obj.get("affs", []),
                "scores": obj.get("scores", []),
                "touching_points": obj.get("touching_points", []),
            })
        json.dump(affordance_data, aff_tmp)
        aff_tmp.close()
        tmp.append(aff_tmp.name)

        camera = _node.config.get("camera", {})
        start_pos = [0.27, 0, 0.307, -3.14159, 0, 0]
        target_pos = [0.4, -0.55, 0.1, -3.14159, 0, 0]
        end_pos = [0.27, 0, 0.307, -3.14159, 0, 0]

        # 4. Plan each top object sequentially
        plan_results = []
        orig = _node.execution_mode
        _node.execution_mode = False  # plan only, do not execute

        for i, obj in enumerate(top_objects):
            obj_idx = objects.index(obj)
            t0 = time.perf_counter()
            diag_steps = []  # collect diagnostic info per step
            try:
                _node.detach_object(); time.sleep(0.3)
                _node.clear_pointcloud_obstacles()
                _node.clear_octomap(); time.sleep(0.5)

                # --- Diagnostic: verify input data ---
                aff_obj = affordance_data[obj_idx]
                n_affs = len(aff_obj.get("affs", []))
                n_scores = len(aff_obj.get("scores", []))
                diag_steps.append(f"affs={n_affs}, scores={n_scores}")
                if n_affs > 0:
                    diag_steps.append(f"aff0={aff_obj['affs'][0]}")
                print(f"[grasp_plan] obj_{obj_idx}: {n_affs} affordances, calling _execute_grasp_core...")

                results = _execute_grasp_core(
                    executor=_node,
                    depth_path=depth_tmp.name,
                    seg_json_path=seg_tmp.name,
                    affordance_path=aff_tmp.name,
                    config=_node.config,
                    target_object_index=obj_idx,
                    return_full_trajectories=True,
                    camera=camera,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    target_pos=target_pos)

                _node.detach_object(); time.sleep(0.3)
                _node.clear_pointcloud_obstacles()
                try:
                    _node.remove_basket_from_scene("basket_1")
                except Exception:
                    pass
                _node.clear_octomap(); time.sleep(0.5)

                result = format_grasp_result(results)
                result["instance_id"] = result.get("instance_id", f"obj_{obj_idx}")
                result["elapsed"] = round(time.perf_counter() - t0, 2)
                # Add diagnostics on failure
                if not result.get("success"):
                    n_results = len(results) if results else 0
                    diag = f"core returned {n_results} result(s)"
                    if n_results == 0:
                        diag += " (empty - no candidates or all planning failed)"
                    elif results:
                        r0 = results[0]
                        diag += f", success={r0.get('success')}, id={r0.get('instance_id')}"
                        diag += f", rank={r0.get('rank')}"
                    diag_steps.append(diag)
                    result["message"] = f"Grasp planning failed ({'; '.join(diag_steps)})"
                    print(f"[grasp_plan] obj_{obj_idx} FAILED: {'; '.join(diag_steps)}")
                    if results:
                        print(f"[grasp_plan]   results[0] keys: {list(results[0].keys())}")
                else:
                    print(f"[grasp_plan] obj_{obj_idx} OK in {result['elapsed']}s")
                plan_results.append(result)
            except Exception as e:
                traceback.print_exc()
                diag_steps.append(f"exception: {e}")
                plan_results.append({
                    "success": False,
                    "instance_id": f"obj_{obj_idx}",
                    "message": f"Exception: {e} ({'; '.join(diag_steps)})",
                    "elapsed": round(time.perf_counter() - t0, 2),
                })

        _node.execution_mode = orig

        # 5. Draw visualization
        vis_bytes = _draw_visualization(rgb_img, objects, plan_results, camera, _node.config)
        vis_b64 = base64.b64encode(vis_bytes).decode()

        # 6. Build summary of detected objects
        obj_summary = []
        for i, obj in enumerate(objects):
            obj_summary.append({
                "index": i,
                "score": round(obj.get("dt_score", 0), 4),
                "bbox": obj.get("dt_bbox", []),
                "category": obj.get("category", "object"),
                "planned": i < len(top_objects),
            })

        # 7. Generate 3D point cloud and key points
        pc = _generate_pointcloud(rgb_img, depth_tmp.name, camera)
        kp = _build_key_points_3d(plan_results, _node.config)

        return jsonify({
            "success": True,
            "visualization": vis_b64,
            "pointcloud_3d": pc,
            "key_points_3d": kp,
            "objects": obj_summary,
            "plan_results": plan_results,
        })

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
    pa.add_argument("--grasp-server", default=None,
                    help="Grasp3D server URL (e.g. http://10.0.73.5:10003)")
    pa.add_argument("--grasp-server-type", default="auto",
                    choices=["auto", "generate", "predict"],
                    help="Grasp server API type: generate (/generate), predict (/api/predict), auto (detect)")
    pa.add_argument("--test", action="store_true")
    a = pa.parse_args()

    _grasp_server_url = a.grasp_server
    _grasp_server_type = a.grasp_server_type
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
