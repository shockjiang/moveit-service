"""Grasp pipeline: grasp3d detection -> format conversion -> MoveIt planning -> visualization."""

import base64, json, os, tempfile, time, traceback
import requests as http_requests
import numpy as np
import cv2

# Distinct colors for objects
_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (255, 0, 128),
]


def _detect_grasp_server_type(server_url, server_type="auto"):
    """Auto-detect grasp server type by probing /api/health first."""
    if server_type != "auto":
        return server_type
    try:
        r = http_requests.get(f"{server_url}/api/health", timeout=5)
        if r.status_code == 200:
            print(f"[grasp] Auto-detected server type: predict (/api/predict)")
            return "predict"
    except Exception:
        pass
    print(f"[grasp] Auto-detected server type: generate (/generate)")
    return "generate"


def _call_grasp3d(server_url, server_type, rgb_bytes, text_prompt, depth_bytes=None):
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
    stype = _detect_grasp_server_type(server_url, server_type)
    if stype == "predict":
        return _call_grasp3d_predict(server_url, rgb_bytes, text_prompt, depth_bytes)
    else:
        return _call_grasp3d_generate(server_url, rgb_bytes, text_prompt)


def _call_grasp3d_generate(server_url, rgb_bytes, text_prompt):
    """Call /generate endpoint (server_grasp.py). Returns unified object list."""
    resp = http_requests.post(
        f"{server_url}/generate",
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


def _call_grasp3d_predict(server_url, rgb_bytes, text_prompt, depth_bytes=None):
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
        f"{server_url}/api/predict",
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


def _build_diag_message(obj_idx, affordance_data, result):
    """Add diagnostic info to result dict and print status."""
    diag_steps = []
    if obj_idx < len(affordance_data):
        aff_obj = affordance_data[obj_idx]
        n_affs = len(aff_obj.get("affs", []))
        n_scores = len(aff_obj.get("scores", []))
        diag_steps.append(f"affs={n_affs}, scores={n_scores}")
        if n_affs > 0:
            diag_steps.append(f"aff0={aff_obj['affs'][0]}")
    if not result.get("success"):
        result["message"] = f"Grasp planning failed ({'; '.join(diag_steps)})"
        print(f"[grasp_plan] obj_{obj_idx} FAILED: {'; '.join(diag_steps)}")
    else:
        print(f"[grasp_plan] obj_{obj_idx} OK in {result.get('elapsed', '?')}s")


def _cleanup_tmp(paths):
    """Remove temporary files, ignoring errors."""
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def run_grasp_pipeline(rgb_bytes, depth_bytes, text_prompt, num_objects,
                       config, robot_name, grasp_server_url,
                       predict_url, grasp_server_type="auto"):
    """Full grasp pipeline: detection -> format conversion -> planning -> visualization.

    1. Call grasp3d to detect objects
    2. Convert to seg_json + affordance temp files
    3. POST /predict per object for planning
    4. Generate 2D visualization + 3D point cloud
    Returns: dict (matching /grasp_plan response format)
    """
    tmp = []
    try:
        # Save depth to temp file
        depth_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        depth_tmp.write(depth_bytes)
        depth_tmp.close()
        tmp.append(depth_tmp.name)

        # Decode RGB image for visualization
        rgb_arr = np.frombuffer(rgb_bytes, np.uint8)
        rgb_img = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
        if rgb_img is None:
            return {"success": False, "message": "Could not decode RGB image"}

        # 1. Call grasp3d detection
        try:
            objects = _call_grasp3d(grasp_server_url, grasp_server_type,
                                    rgb_bytes, text_prompt, depth_bytes=depth_bytes)
        except http_requests.ConnectionError:
            return {"success": False,
                    "message": f"Cannot reach grasp3d server at {grasp_server_url}"}
        except http_requests.HTTPError as e:
            return {"success": False, "message": f"Grasp3d server error: {e}"}

        if not objects:
            return {"success": False, "message": "No objects detected by grasp3d"}

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

        affordance_data = []
        for obj in objects:
            affordance_data.append({
                "affs": obj.get("affs", []),
                "scores": obj.get("scores", []),
                "touching_points": obj.get("touching_points", []),
            })
        aff_tmp = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
        json.dump(affordance_data, aff_tmp)
        aff_tmp.close()
        tmp.append(aff_tmp.name)

        camera = config.get("camera", {})

        # 4. Plan each top object via POST /predict
        plan_results = []
        for i, obj in enumerate(top_objects):
            obj_idx = objects.index(obj)
            t0 = time.perf_counter()
            try:
                resp = http_requests.post(predict_url, json={
                    "robot_name": robot_name,
                    "dpt": depth_tmp.name,
                    "seg_json": seg_tmp.name,
                    "objs": aff_tmp.name,
                    "camera": camera,
                    "target_object_index": obj_idx,
                    "execution_simulation": True,  # plan only
                }, timeout=120)
                result = resp.json()
                result["instance_id"] = result.get("instance_id", f"obj_{obj_idx}")
                result["elapsed"] = round(time.perf_counter() - t0, 2)
                _build_diag_message(obj_idx, affordance_data, result)
                plan_results.append(result)
            except Exception as e:
                traceback.print_exc()
                plan_results.append({
                    "success": False,
                    "instance_id": f"obj_{obj_idx}",
                    "message": f"Exception: {e}",
                    "elapsed": round(time.perf_counter() - t0, 2),
                })

        # 5. Draw visualization
        vis_bytes = _draw_visualization(rgb_img, objects, plan_results, camera, config)
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
        kp = _build_key_points_3d(plan_results, config)

        return {
            "success": True,
            "visualization": vis_b64,
            "pointcloud_3d": pc,
            "key_points_3d": kp,
            "objects": obj_summary,
            "plan_results": plan_results,
        }

    finally:
        _cleanup_tmp(tmp)
