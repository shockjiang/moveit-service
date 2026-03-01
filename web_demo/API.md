# MoveIt Grasp Server API

## GET /health

```json
{
  "service": "MoveIt Grasp Server",
  "status": "healthy | not_initialized",
  "robot": "xarm7",
  "grasp_server": "http://... | not_configured",
  "grasp_server_type": "auto | generate | predict"
}
```

---

## POST /predict

### Input (JSON)

```jsonc
{
  "robot_name":  "xarm7",                          // 必填
  "camera": {                                       // 必填
    "intrinsics": { "fx": 909.665, "fy": 909.533, "cx": 636.739, "cy": 376.35 },
    "extrinsics": {
      "translation": [0.325, 0.028, 0.658],        // cam->base (m)
      "quaternion":  [-0.703, 0.71, -0.026, 0.021]  // cam->base (xyzw)
    }
  },
  "dpt":      "/path/to/depth.png",                // 必填, 路径 或 base64
  "objs":     "/path/to/affordance.json",           // 必填, 路径 或 list
  "seg_json": "/path/to/detection.json",            // 必填, 路径 或 dict
  "start_pos":   [x, y, z, roll, pitch, yaw],       // 可选, 不传则用 config 默认值
  "target_pos":  [x, y, z, roll, pitch, yaw],       // 可选
  "end_pos":     [x, y, z, roll, pitch, yaw],       // 可选
  "target_object_index": -1,                         // 可选, -1=自动
  "execution_simulation": true,                      // 可选, true=仅规划
  "servo_dt": 0.02,                                  // 可选, 插值步长(s), 不传=原始稀疏点
  "pos_tol": 0.01,                                   // 可选, 位置容差(m)
  "ori_tol": 0.1                                     // 可选, 姿态容差(rad)
}
```

`dpt`/`objs`/`seg_json` 自动判断：string + 文件存在 -> 路径；否则 dpt 按 base64、objs/seg_json 按 JSON 解析。

### Input (Multipart Form)

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| dpt | file | Y | 深度图 |
| objs | string(JSON) | Y | affordance 数据 |
| seg_json | string(JSON) | Y | 检测分割数据 |
| robot_name | string | Y | |
| camera | string(JSON) | Y | |
| start_pos | string(JSON) | N | |
| target_pos | string(JSON) | N | |
| end_pos | string(JSON) | N | |
| target_object_index | string | N | |
| execution_simulation | string | N | "true"/"false" |
| servo_dt | string | N | |
| pos_tol | string | N | |
| ori_tol | string | N | |

### Output

```jsonc
// 成功
{
  "success": true,
  "obj_index": 2,
  "instance_id": "obj_2",
  "planning_time": 3.45,
  "trajectory": {
    "approaching":    { "positions": [[j1..j7], ...], "velocities": [...], "accelerations": [...], "dt": 0.02, "duration": 2.5 },
    "grasp_approach": { ... },
    "retreat":        { ... },
    "carrying":       { ... },
    "returning":      { ... }
  }
}

// 失败
{ "success": false, "message": "..." }
```

`dt` 和 `duration` 仅在传入 `servo_dt` 时返回。

---

## POST /grasp_plan

需要 `--grasp-server` 启动参数。始终仅规划，不执行。

### Input (Multipart Form)

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| rgb | file | Y | RGB 图片 |
| depth | file | Y | 深度图 (uint16 PNG) |
| text_prompt | string | N | 检测提示词，默认 `"object"` |
| num_objects | string(int) | N | 规划物体数，默认 3，范围 [1,10] |

### Output

```jsonc
{
  "success": true,
  "visualization": "<base64 JPEG>",
  "pointcloud_3d": {
    "x": [...], "y": [...], "z": [...],
    "colors": ["rgb(128,64,32)", ...]
  },
  "key_points_3d": {
    "home": [x, y, z],
    "basket": [x, y, z],
    "per_object": [
      { "instance_id": "obj_0", "grasp": [x,y,z], "pregrasp": [x,y,z] }
    ]
  },
  "objects": [
    { "index": 0, "score": 0.95, "bbox": [x1,y1,x2,y2], "category": "bottle", "planned": true }
  ],
  "plan_results": [
    {
      "success": true,
      "instance_id": "obj_0",
      "planning_time": 3.45,
      "elapsed": 5.12,
      "trajectory": { "approaching": {...}, "grasp_approach": {...}, "retreat": {...}, "carrying": {...}, "returning": {...} }
    }
  ]
}
```
