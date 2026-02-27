# MoveIt Grasp Server API

## GET /

健康检查。

**Response:**
```json
{
  "service": "MoveIt Grasp Server",
  "status": "healthy | not_initialized",
  "robot": "xarm7"
}
```

---

## POST /predict

抓取规划。输入深度图 + 物体检测 + 位姿参数，返回 5 阶段关节轨迹。

### Input (JSON)

每个数据字段支持**两种传法**：文件路径(string) 或 直接传数据。

```jsonc
{
  "robot_name":  "xarm7",                            // 必填, 机器人名称
  "camera": {                                         // 必填, 相机内外参
    "intrinsics": {
      "fx": 909.665, "fy": 909.533,
      "cx": 636.739, "cy": 376.35
    },
    "extrinsics": {
      "translation": [0.325, 0.028, 0.658],          // cam→base 平移 (m)
      "quaternion":  [-0.703, 0.71, -0.026, 0.021]   // cam→base 旋转 (xyzw)
    }
  },
  "dpt":         "...",                               // 必填, 深度图 — 见下方类型说明
  "objs":        "..." | [...],                       // 必填, 物体 affordance — 见下方类型说明
  "seg_json":    "..." | {...},                       // 必填, 分割/检测数据 — 见下方类型说明
  "start_pos":   [0.27, 0, 0.307, -3.14, 0, 0],      // 必填, 起始位姿 [x,y,z,roll,pitch,yaw] (m, rad)
  "target_pos":  [0.4, -0.55, 0.1, -3.14, 0, 0],     // 必填, 放置框中心 [x,y,z,roll,pitch,yaw] (m, rad)
  "end_pos":     [0.27, 0, 0.307, -3.14, 0, 0],       // 必填, 结束位姿 [x,y,z,roll,pitch,yaw] (m, rad)
  "target_object_index": -1,                           // 可选, 指定目标物体索引, -1=自动选择最优
  "execution_simulation": true,                        // 可选, true=仅规划不执行, false=规划并执行
  "servo_dt": 0.02                                     // 可选, 伺服插值步长(秒), 不传=返回原始稀疏点
}
```

#### dpt / objs / seg_json 类型说明

| 字段 | 传路径 (string) | 传数据 |
|------|----------------|--------|
| dpt | `"/path/to/depth.png"` 服务器本地路径 | `"iVBORw0KGgo..."` base64 编码的 PNG 二进制 |
| objs | `"/path/to/affordance.json"` 服务器本地路径 | `[{"score":0.9, "affs":[...], "dt_bbox":[...], "dt_mask":{...}}, ...]` 直接传 list |
| seg_json | `"/path/to/detection.json"` 服务器本地路径 | `{"results":[...], "chosen_policy":"...", ...}` 直接传 dict |

自动判断规则：
- **string + 文件存在** → 当作路径读取
- **string + 文件不存在** → dpt 按 base64 解码，objs/seg_json 按 JSON 字符串解析
- **list / dict** → 直接作为数据使用

#### 全部字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| robot_name | string | Y | 机器人名称，对应 `config/{name}_config.json` |
| camera | object | Y | 相机内参(intrinsics) + 外参(extrinsics, cam2base) |
| dpt | string 或 string(base64) | Y | 深度图，uint16 PNG，单位 mm |
| objs | string 或 list | Y | 物体 affordance 数据，含 mask/bbox/score/affs |
| seg_json | string 或 dict | Y | 检测分割数据 |
| start_pos | float[6] | Y | 起始位姿 [x,y,z,roll,pitch,yaw]，其中 [3:] 作为抓取姿态基准 |
| target_pos | float[6] | Y | 放置框中心 [x,y,z,roll,pitch,yaw]，[:3] 作为 carrying 目标 |
| end_pos | float[6] | Y | 结束位姿 [x,y,z,roll,pitch,yaw]，returning 阶段目标（不一定等于 start_pos） |
| target_object_index | int | N | 指定抓哪个物体，-1 或不传则自动选最优 |
| execution_simulation | bool | N | true(默认)=仅规划，false=规划+真机执行 |
| servo_dt | float | N | 伺服模式插值步长（秒）。传入后做三次 Hermite 样条插值重采样。推荐 0.02 (50Hz) |

### Input (Multipart Form)

适用于文件上传场景：

| Form 字段 | 类型 | 说明 |
|-----------|------|------|
| dpt | file | 深度图文件上传 |
| objs | string (JSON) | affordance 数据的 JSON 字符串 |
| seg_json | string (JSON) | 检测分割数据的 JSON 字符串 |
| rgb | file | 可选，RGB 图 |
| robot_name | string | 机器人名称 |
| camera | string (JSON) | 相机内外参 |
| start_pos | string (JSON) | 起始位姿 |
| target_pos | string (JSON) | 放置框中心 |
| end_pos | string (JSON) | 结束位姿 |
| target_object_index | string | 可选，目标物体索引 |
| execution_simulation | string | 可选，"true"/"false" |
| servo_dt | string | 可选，插值步长 |

---

### Output (成功)

```json
{
  "success": true,
  "obj_index": 2,                     // 被选中物体在 affordance 列表中的索引
  "instance_id": "obj_2",             // 物体实例 ID
  "planning_time": 3.45,              // 规划耗时 (秒)
  "trajectory": {
    "approaching": {                   // 阶段1: 接近预抓取点
      "positions":     [[j1,...,j7], ...],   // 各 waypoint 的 7 关节角度 (rad)
      "velocities":    [[v1,...,v7], ...],   // 各 waypoint 的 7 关节角速度 (rad/s)
      "accelerations": [[a1,...,a7], ...],   // 各 waypoint 的 7 关节角加速度 (rad/s²)
      "dt": 0.02,                      // [仅 servo_dt 模式] 点间时间间隔 (秒)
      "duration": 2.5                  // [仅 servo_dt 模式] 本段总时长 (秒)
    },
    "grasp_approach": { ... },         // 阶段2: 从预抓取点到抓取点
    "retreat":        { ... },         // 阶段3: 抓取后回退到预抓取点
    "carrying":       { ... },         // 阶段4: 搬运到放置框上方
    "returning":      { ... }          // 阶段5: 从放置框回到 end_pos
  }
}
```

#### 轨迹数据说明

**无 servo_dt（原始模式）：**
- positions/velocities/accelerations 为 MoveIt 规划的原始稀疏 waypoints
- 点数较少（通常 10~50 个），时间间隔不均匀
- 适合支持轨迹跟踪的控制器（如 ROS FollowJointTrajectory）

**有 servo_dt（插值模式）：**
- 通过三次 Hermite 样条对原始轨迹插值，按固定 dt 等间距重采样
- 点数密集（如 dt=0.02 时约 100~500 个/段），时间间隔均匀
- 适合伺服模式逐点发送（如 xArm `set_servo_angle_j`，每 dt 秒发一个点）
- 额外返回 `dt`（点间间隔）和 `duration`（本段总时长）

### Output (失败)

```json
{
  "success": false,
  "message": "Grasp planning failed"
}
```

---

## 参数优先级

server 传入的参数优先于 config 文件中的默认值：

```
传入参数 > config/{robot}_config.json > 硬编码默认值
```

| 传入参数 | 覆盖 config 路径 | 用途 |
|---------|-----------------|------|
| camera | config.camera | 相机标定参数 |
| start_pos[3:] | config.home.orientation | 抓取姿态基准 (roll,pitch,yaw) |
| target_pos[:3] | config.basket.center | 放置框中心坐标 |
| end_pos | config.home.joints (fallback) | returning 阶段终点 |

---

## 示例

### curl — 传路径

```bash
curl -X POST http://localhost:14086/predict \
  -H "Content-Type: application/json" \
  -d '{
    "robot_name": "xarm7",
    "dpt": "/path/to/depth.png",
    "objs": "/path/to/affordance.json",
    "seg_json": "/path/to/detection.json",
    "camera": {
      "intrinsics": {"fx":909.665,"fy":909.533,"cx":636.739,"cy":376.35},
      "extrinsics": {"translation":[0.325,0.028,0.658],"quaternion":[-0.703,0.71,-0.026,0.021]}
    },
    "start_pos":  [0.27, 0, 0.307, -3.14159, 0, 0],
    "target_pos": [0.4, -0.55, 0.1, -3.14159, 0, 0],
    "end_pos":    [0.27, 0, 0.307, -3.14159, 0, 0],
    "servo_dt": 0.02
  }'
```

### Python — 传数据

```python
import requests, base64

# 深度图 base64
with open("depth.png", "rb") as f:
    dpt_b64 = base64.b64encode(f.read()).decode()

# affordance 直接传 list
objs = [
    {"score": 0.95, "affs": [...], "dt_bbox": [x1,y1,x2,y2], "dt_mask": {...}},
    {"score": 0.82, "affs": [...], "dt_bbox": [x1,y1,x2,y2], "dt_mask": {...}},
]

# seg_json 直接传 dict
seg = {"results": [...], "chosen_policy": "dinox"}

resp = requests.post("http://localhost:14086/predict", json={
    "robot_name": "xarm7",
    "dpt": dpt_b64,
    "objs": objs,
    "seg_json": seg,
    "camera": {
        "intrinsics": {"fx": 909.665, "fy": 909.533, "cx": 636.739, "cy": 376.35},
        "extrinsics": {"translation": [0.325, 0.028, 0.658], "quaternion": [-0.703, 0.71, -0.026, 0.021]}
    },
    "start_pos":  [0.27, 0, 0.307, -3.14159, 0, 0],
    "target_pos": [0.4, -0.55, 0.1, -3.14159, 0, 0],
    "end_pos":    [0.27, 0, 0.307, -3.14159, 0, 0],
    "servo_dt": 0.02
})
result = resp.json()
```

### client.py

```bash
python client.py \
  --dpt test_data/grasp-wrist-dpt_opt.png \
  --objs test_data/affordance.json \
  --seg test_data/rgb_detection_wrist.json \
  --server http://localhost:14086
```
