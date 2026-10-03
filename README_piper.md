对：server 只出物理 **EEF10**，真机用 SDK **`EndPoseCtrl`（固件 IK）** 执行，通常比主机再写一套 IK 更省事。

## Server 启动（已实现）

```bash
# OpenPI msgpack 协议（真机 WebsocketClientPolicy）
CUDA_VISIBLE_DEVICES=6 bash scripts/deploy_piper_openpi.sh \
  /data/zixian_guo/projects/haoming/project/PI/OpenWAM/outputs/banana_piper_ft/2026-10-02_15-49-35
# 等价：python scripts/deploy.py --ckpt-dir /path/to/banana_ckpt --protocol openpi --port 8000

# 可选 RTC metadata
CUDA_VISIBLE_DEVICES=6 bash scripts/deploy_piper_openpi.sh /path/to/banana_ckpt --training-rtc --max-delay 8

# 联调 smoke（需 server 已起）
python scripts/infer_banana_piper_openpi_smoke.py --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/healthz
```

实现：`openwam/deploy/openpi_piper_server.py`（`--protocol openpi`）。

---

# OpenWAM Server ↔ Piper 真机：EEF10 线缆协议说明

## 0. 目标与分工

| 角色 | 职责 |
|---|---|
| **OpenWAM Server** | WebSocket 推理服务；**线缆上只收发物理 raw EEF10（10-D）**；内部负责 normalize / 80-D unify / gather |
| **真机客户端** | 采图、组观测、调 `infer`；把 EEF10 转成 Piper `EndPoseCtrl` + 夹爪；可选 RTC 队列 |
| **不在本协议内** | 主机侧解析 IK、80-D 向量上总线、LIBERO 7D OSC、关节 7D |

**原则**：Deploy 总线 = **物理 raw EEF10**，不是统一 80-D，也不是归一化值。

---

## 1. 传输层（必须兼容 openpi-client）

真机使用 `packages/openpi-client` 的 `WebsocketClientPolicy`，server 必须兼容：

- **URL**：`ws://<host>:<port>`（默认端口由部署决定，常为 `8000`）
- **压缩**：关闭（`compression=None`）
- **帧大小**：不限制（`max_size=None`）
- **序列化**：msgpack + numpy 扩展（与 `openpi_client.msgpack_numpy` 一致）
  - `ndarray` → `{b"__ndarray__": True, b"data", b"dtype", b"shape"}`
- **鉴权（可选）**：请求头 `Authorization: Api-Key <key>`
- **健康检查（建议）**：`GET /healthz` → `200 OK`

### 连接时序

1. Client 连上后，**Server 先发一帧 metadata**（msgpack dict）
2. 之后循环：Client 发 obs → Server `infer` → 回 result（**binary** msgpack）
3. 异常：可回 **UTF-8 文本帧**（traceback）；Client 收到 `str` 即视为错误

Server 对单连接是 **同步一问一答**；RTC 并发由 Client 线程管理，Server **无会话状态**。

---

## 2. EEF10 物理布局（线缆唯一动作/本体表示）

固定 **10 维 `float32`**，名：`EEF10` / `state_eef10` / `action_eef10`：

```text
index:  0    1    2    3    4    5    6    7    8    9
        x    y    z   r00  r10  r20  r01  r11  r21  grip
        --------  --------------  --------------  ----
        xyz (m)   R 第0列 (3)     R 第1列 (3)    open_scale
```

即：`[xyz_m(3) | rot6d(6) | gripper(1)]`

### 字段语义

1. **`xyz`（0:3）**  
   - 末端位置，单位 **米**  
   - 与 Banana 训练一致：Piper FK，内部 mm→m  

2. **`rot6d`（3:9）**  
   - 旋转矩阵 \(R\) 的前两列：`[R[:,0]; R[:,1]]`  
   - 约定名：`piper_fk_rot6d_R_cols01`  
   - **不是** 轴角 / 四元数 / 欧拉角  
   - 训练侧 rot6d 常 pin 成 identity 做统计；**线缆上仍为物理 rot6d**

3. **`gripper`（第 9 维）**  
   - 连续开合：`minus1_closed_plus1_open`  
   - **-1 = 完全闭合，+1 = 完全打开**  
   - 与开口米数：`open_scale = 2 * clip(g / gripper_max_m, 0, 1) - 1`  
   - 反变换：`g_m = (open_scale + 1) / 2 * gripper_max_m`

### proprio / action

| 字段 | 含义 |
|---|---|
| `observation/state` | 当前物理 EEF10，shape `(10,)` |
| `actions` | 目标物理 EEF10 chunk，shape `(H, 10)` |

Server 内部可对 raw EEF10 做 min-max，再 scatter 进 80-D（`unify_action_map: ["0-9"]` 左臂槽）；**进出 WebSocket 前必须 gather + unnormalize 回 10-D raw**。

---

## 3. 连接 metadata（首帧，Server → Client）

### 同步控制（最低要求）

```python
{
    "action_horizon": 30,              # 与返回 actions.shape[0] 一致
    "raw_action_dim": 10,
    "wire_action_space": "absolute",
    "action_representation": "eef10_rot6d",  # 建议显式声明
    "gripper_convention": "minus1_closed_plus1_open",
    "rot6d_convention": "piper_fk_rot6d_R_cols01",
}
```

### 若对接异步 RTC 客户端（额外）

```python
{
    "training_rtc": True,
    "max_delay": <int>,                # Client 可用 delay 上界（含）
    "action_horizon": <int>,           # 必须 > max_delay
    "raw_action_dim": 10,
    "wire_action_space": "absolute",
    "action_representation": "eef10_rot6d",
    "gripper_convention": "minus1_closed_plus1_open",
    "rot6d_convention": "piper_fk_rot6d_R_cols01",
}
```

> 注意：现成 OpenPI `rtc_main.py` 写死 `raw_action_dim=7` 与关节 canonicalize；**EEF10 需要配套的 EEF 版客户端**。Server 仍应按上表诚实声明。

---

## 4. 请求载荷（Client → Server）

```python
{
    "observation/top_image": uint8_rgb,            # (H, W, 3)，原始相机分辨率即可
    "observation/right_wrist_image": uint8_rgb,  # (H, W, 3)
    "observation/state": float32[10],              # 物理 EEF10
    "prompt": str,

    # 仅 RTC 请求携带：
    "rtc": {
        "prefix": float32[delay, 10],  # 已承诺的绝对 EEF10 命令
        "delay": int,                  # 0..max_delay；异步通常 ≥ 1
        "start_index": int,            # 观测控制 tick
        "request_id": int,
    },
}
```

### Server 必须遵守

1. **图像**：接受任意 `HxW` 的 `uint8` RGB；resize / 归一化在 Server 内完成。  
2. **`state`**：按物理 EEF10 理解；**不要**期望关节 7D。  
3. 若 Client 只提供关节：Server 可自行 FK→EEF10，但须与 Banana **同一 FK**；协议上仍建议 Client 直接发 EEF10。  
4. 无 `rtc`：按 `delay=0` 处理（无硬前缀）。  
5. 有 `rtc`：前缀是观测时刻起将执行的绝对 EEF10；`prefix[0]` 对应当前 tick 命令，**不是**响应到达时刻。

---

## 5. 响应载荷（Server → Client）

### 同步

```python
{
    "actions": float32[H, 10],   # 物理绝对 EEF10
    "policy_timing": {"infer_ms": float},   # 建议
}
```

可选附加 `server_timing`（由 WebSocket 包装层写入亦可）。

### RTC

```python
{
    "actions": float32[H, 10],
    "rtc": {
        "request_id": <回显>,
        "start_index": <回显>,
        "delay": <回显>,
        "action_space": "absolute",
    },
    "policy_timing": {"infer_ms": float},
}
```

### RTC 硬约束（Client 会校验）

1. `rtc` 四字段与请求一致。  
2. `np.array_equal(actions[:delay], prefix)` —— 前缀 **逐元素不变**（禁止 unnormalize 漂移；应用原 `prefix` 覆写）。  
3. `actions` 全有限；`shape == (action_horizon, 10)`。  
4. Server **无状态**；队列所有权在 Client。

---

## 6. 真机如何消费 EEF10（Server agent 需知情，但不实现）

Client 侧推荐路径（**不在主机算 IK**）：

1. `xyz_m` → mm → `×1000` 得到 SDK `0.001mm` 整数  
2. `rot6d` → 恢复/正交化 \(R\) → **欧拉角 RPY（度）** → `×1000` 得到 SDK `0.001°`  
3. `MotionCtrl_2(CAN, MOVE_P 或约定模式, speed, ...)` + `EndPoseCtrl(X,Y,Z,RX,RY,RZ)`  
4. gripper：`[-1,1] → 米 → SDK 0.001mm`，再 `GripperCtrl`

因此 Server **只需保证 EEF10 物理正确**；不必返回关节角。  
但必须保证：xyz / rot6d 坐标系与 Banana FK、与 Piper `EndPoseCtrl` 所用 base 系一致，否则“直接给机械臂”会系统性偏。

---

## 7. 与 80-D 统一空间的边界

```text
统一向量 [0:10)  ←→  单臂 banana 左槽 EEF10
其余槽位 mask，不得出现在 WebSocket 的 state/actions 中
```

| 阶段 | 表示 |
|---|---|
| 训练内部 | raw EEF10 → min-max → scatter → 80-D |
| **WebSocket** | **仅 raw EEF10 (10,)** / **(H,10)** |
| 双臂 EEF20 | 本 Piper 单臂部署 **不使用** |

---

## 8. Server 实现检查清单

- [ ] 首帧 metadata 含 `action_horizon`、`raw_action_dim=10`、`wire_action_space=absolute`  
- [ ] `infer` 入：读 `observation/state` 为 `(10,)` EEF10；图像 key 兼容 Piper  
- [ ] `infer` 出：`actions` 为 `(H, 10)` **物理** EEF10（已 unnormalize + gather）  
- [ ] 夹爪维语义为 `[-1,1]`，不是米、不是毫米  
- [ ] rot6d 为 \(R\) 前两列，不是 rpy  
- [ ] 成功响应为 msgpack **bytes**；错误为 **string**  
- [ ] RTC：回显字段 + `actions[:d] = prefix`  
- [ ] 延迟：控制周期（如 30 Hz）下，`max_delay` 能盖住稳态 RTT；冷启动超时由 Client 配置  
- [ ] **不要**把 80-D 或归一化向量发给 Client  

---

## 9. 最小联调步骤（Server）

1. 启动 server，用 `WebsocketClientPolicy` 连接，打印 metadata。  
2. 构造假观测：`state =` 合法 EEF10（可用当前臂 FK），两张 `uint8` 图，`prompt`。  
3. 调用 `infer`，检查 `actions.shape == (H, 10)`，`xyz` 量级为米，`grip ∈ [-1,1]`，rot6d 两列近似单位正交。  
4. （RTC）带 `prefix` 再请求，断言前缀 `array_equal`。  
5. 真机侧用同一帧 `actions[0]` 走 `EndPoseCtrl` 看是否与预期位姿一致（小步验证）。

---

## 10. 非目标 / 常见误区

| 误区 | 正确做法 |
|---|---|
| 返回 80-D | 只返回 gather 后的 10-D raw |
| 返回归一化 EEF10 | 线缆必须物理单位 |
| 当关节 7D 对接现成 `rtc_main.py` | 需 EEF 版客户端或 Server 转关节（本说明是 EEF-only） |
| 夹爪当米 | 第 9 维是 `[-1,1]` |
| rot6d 当欧拉 | Client 负责 rot6d→RPY；Server 保持 rot6d |
| Server 缓存上一 chunk | 无状态；prefix 以请求为准 |

---

## 11. 给 Server Agent 的一句话

实现一个 **openpi 兼容的 WebSocket Policy Server**：首帧报 EEF10 metadata；`infer` 吃 Piper 图像 + 物理 EEF10 state（及可选 RTC prefix），吐出物理 EEF10 `actions[H,10]`；归一化与 80-D 统一空间留在进程内，**永不出现在线缆上**。真机将用 `EndPoseCtrl` 直接执行，无需 Server 提供关节逆解。

---

如果你希望我把这篇落到仓库里（例如 `docs/openwam_eef10_server_protocol.md`），切换到 Agent 模式再说一声即可。对：server 只出物理 **EEF10**，真机用 SDK **`EndPoseCtrl`（固件 IK）** 执行，通常比主机再写一套 IK 更省事。

## Server 启动（已实现）

```bash
# OpenPI msgpack 协议（真机 WebsocketClientPolicy）
CUDA_VISIBLE_DEVICES=6 bash scripts/deploy_piper_openpi.sh /path/to/banana_ckpt
# 等价：python scripts/deploy.py --ckpt-dir /path/to/banana_ckpt --protocol openpi --port 8000

# 可选 RTC metadata
CUDA_VISIBLE_DEVICES=6 bash scripts/deploy_piper_openpi.sh /path/to/banana_ckpt --training-rtc --max-delay 8

# 联调 smoke（需 server 已起）
python scripts/infer_banana_piper_openpi_smoke.py --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/healthz
```

实现：`openwam/deploy/openpi_piper_server.py`（`--protocol openpi`）。

---

# OpenWAM Server ↔ Piper 真机：EEF10 线缆协议说明

## 0. 目标与分工

| 角色 | 职责 |
|---|---|
| **OpenWAM Server** | WebSocket 推理服务；**线缆上只收发物理 raw EEF10（10-D）**；内部负责 normalize / 80-D unify / gather |
| **真机客户端** | 采图、组观测、调 `infer`；把 EEF10 转成 Piper `EndPoseCtrl` + 夹爪；可选 RTC 队列 |
| **不在本协议内** | 主机侧解析 IK、80-D 向量上总线、LIBERO 7D OSC、关节 7D |

**原则**：Deploy 总线 = **物理 raw EEF10**，不是统一 80-D，也不是归一化值。

---

## 1. 传输层（必须兼容 openpi-client）

真机使用 `packages/openpi-client` 的 `WebsocketClientPolicy`，server 必须兼容：

- **URL**：`ws://<host>:<port>`（默认端口由部署决定，常为 `8000`）
- **压缩**：关闭（`compression=None`）
- **帧大小**：不限制（`max_size=None`）
- **序列化**：msgpack + numpy 扩展（与 `openpi_client.msgpack_numpy` 一致）
  - `ndarray` → `{b"__ndarray__": True, b"data", b"dtype", b"shape"}`
- **鉴权（可选）**：请求头 `Authorization: Api-Key <key>`
- **健康检查（建议）**：`GET /healthz` → `200 OK`

### 连接时序

1. Client 连上后，**Server 先发一帧 metadata**（msgpack dict）
2. 之后循环：Client 发 obs → Server `infer` → 回 result（**binary** msgpack）
3. 异常：可回 **UTF-8 文本帧**（traceback）；Client 收到 `str` 即视为错误

Server 对单连接是 **同步一问一答**；RTC 并发由 Client 线程管理，Server **无会话状态**。

---

## 2. EEF10 物理布局（线缆唯一动作/本体表示）

固定 **10 维 `float32`**，名：`EEF10` / `state_eef10` / `action_eef10`：

```text
index:  0    1    2    3    4    5    6    7    8    9
        x    y    z   r00  r10  r20  r01  r11  r21  grip
        --------  --------------  --------------  ----
        xyz (m)   R 第0列 (3)     R 第1列 (3)    open_scale
```

即：`[xyz_m(3) | rot6d(6) | gripper(1)]`

### 字段语义

1. **`xyz`（0:3）**  
   - 末端位置，单位 **米**  
   - 与 Banana 训练一致：Piper FK，内部 mm→m  

2. **`rot6d`（3:9）**  
   - 旋转矩阵 \(R\) 的前两列：`[R[:,0]; R[:,1]]`  
   - 约定名：`piper_fk_rot6d_R_cols01`  
   - **不是** 轴角 / 四元数 / 欧拉角  
   - 训练侧 rot6d 常 pin 成 identity 做统计；**线缆上仍为物理 rot6d**

3. **`gripper`（第 9 维）**  
   - 连续开合：`minus1_closed_plus1_open`  
   - **-1 = 完全闭合，+1 = 完全打开**  
   - 与开口米数：`open_scale = 2 * clip(g / gripper_max_m, 0, 1) - 1`  
   - 反变换：`g_m = (open_scale + 1) / 2 * gripper_max_m`

### proprio / action

| 字段 | 含义 |
|---|---|
| `observation/state` | 当前物理 EEF10，shape `(10,)` |
| `actions` | 目标物理 EEF10 chunk，shape `(H, 10)` |

Server 内部可对 raw EEF10 做 min-max，再 scatter 进 80-D（`unify_action_map: ["0-9"]` 左臂槽）；**进出 WebSocket 前必须 gather + unnormalize 回 10-D raw**。

---

## 3. 连接 metadata（首帧，Server → Client）

### 同步控制（最低要求）

```python
{
    "action_horizon": 30,              # 与返回 actions.shape[0] 一致
    "raw_action_dim": 10,
    "wire_action_space": "absolute",
    "action_representation": "eef10_rot6d",  # 建议显式声明
    "gripper_convention": "minus1_closed_plus1_open",
    "rot6d_convention": "piper_fk_rot6d_R_cols01",
}
```

### 若对接异步 RTC 客户端（额外）

```python
{
    "training_rtc": True,
    "max_delay": <int>,                # Client 可用 delay 上界（含）
    "action_horizon": <int>,           # 必须 > max_delay
    "raw_action_dim": 10,
    "wire_action_space": "absolute",
    "action_representation": "eef10_rot6d",
    "gripper_convention": "minus1_closed_plus1_open",
    "rot6d_convention": "piper_fk_rot6d_R_cols01",
}
```

> 注意：现成 OpenPI `rtc_main.py` 写死 `raw_action_dim=7` 与关节 canonicalize；**EEF10 需要配套的 EEF 版客户端**。Server 仍应按上表诚实声明。

---

## 4. 请求载荷（Client → Server）

```python
{
    "observation/top_image": uint8_rgb,            # (H, W, 3)，原始相机分辨率即可
    "observation/right_wrist_image": uint8_rgb,  # (H, W, 3)
    "observation/state": float32[10],              # 物理 EEF10
    "prompt": str,

    # 仅 RTC 请求携带：
    "rtc": {
        "prefix": float32[delay, 10],  # 已承诺的绝对 EEF10 命令
        "delay": int,                  # 0..max_delay；异步通常 ≥ 1
        "start_index": int,            # 观测控制 tick
        "request_id": int,
    },
}
```

### Server 必须遵守

1. **图像**：接受任意 `HxW` 的 `uint8` RGB；resize / 归一化在 Server 内完成。  
2. **`state`**：按物理 EEF10 理解；**不要**期望关节 7D。  
3. 若 Client 只提供关节：Server 可自行 FK→EEF10，但须与 Banana **同一 FK**；协议上仍建议 Client 直接发 EEF10。  
4. 无 `rtc`：按 `delay=0` 处理（无硬前缀）。  
5. 有 `rtc`：前缀是观测时刻起将执行的绝对 EEF10；`prefix[0]` 对应当前 tick 命令，**不是**响应到达时刻。

---

## 5. 响应载荷（Server → Client）

### 同步

```python
{
    "actions": float32[H, 10],   # 物理绝对 EEF10
    "policy_timing": {"infer_ms": float},   # 建议
}
```

可选附加 `server_timing`（由 WebSocket 包装层写入亦可）。

### RTC

```python
{
    "actions": float32[H, 10],
    "rtc": {
        "request_id": <回显>,
        "start_index": <回显>,
        "delay": <回显>,
        "action_space": "absolute",
    },
    "policy_timing": {"infer_ms": float},
}
```

### RTC 硬约束（Client 会校验）

1. `rtc` 四字段与请求一致。  
2. `np.array_equal(actions[:delay], prefix)` —— 前缀 **逐元素不变**（禁止 unnormalize 漂移；应用原 `prefix` 覆写）。  
3. `actions` 全有限；`shape == (action_horizon, 10)`。  
4. Server **无状态**；队列所有权在 Client。

---

## 6. 真机如何消费 EEF10（Server agent 需知情，但不实现）

Client 侧推荐路径（**不在主机算 IK**）：

1. `xyz_m` → mm → `×1000` 得到 SDK `0.001mm` 整数  
2. `rot6d` → 恢复/正交化 \(R\) → **欧拉角 RPY（度）** → `×1000` 得到 SDK `0.001°`  
3. `MotionCtrl_2(CAN, MOVE_P 或约定模式, speed, ...)` + `EndPoseCtrl(X,Y,Z,RX,RY,RZ)`  
4. gripper：`[-1,1] → 米 → SDK 0.001mm`，再 `GripperCtrl`

因此 Server **只需保证 EEF10 物理正确**；不必返回关节角。  
但必须保证：xyz / rot6d 坐标系与 Banana FK、与 Piper `EndPoseCtrl` 所用 base 系一致，否则“直接给机械臂”会系统性偏。

---

## 7. 与 80-D 统一空间的边界

```text
统一向量 [0:10)  ←→  单臂 banana 左槽 EEF10
其余槽位 mask，不得出现在 WebSocket 的 state/actions 中
```

| 阶段 | 表示 |
|---|---|
| 训练内部 | raw EEF10 → min-max → scatter → 80-D |
| **WebSocket** | **仅 raw EEF10 (10,)** / **(H,10)** |
| 双臂 EEF20 | 本 Piper 单臂部署 **不使用** |

---

## 8. Server 实现检查清单

- [ ] 首帧 metadata 含 `action_horizon`、`raw_action_dim=10`、`wire_action_space=absolute`  
- [ ] `infer` 入：读 `observation/state` 为 `(10,)` EEF10；图像 key 兼容 Piper  
- [ ] `infer` 出：`actions` 为 `(H, 10)` **物理** EEF10（已 unnormalize + gather）  
- [ ] 夹爪维语义为 `[-1,1]`，不是米、不是毫米  
- [ ] rot6d 为 \(R\) 前两列，不是 rpy  
- [ ] 成功响应为 msgpack **bytes**；错误为 **string**  
- [ ] RTC：回显字段 + `actions[:d] = prefix`  
- [ ] 延迟：控制周期（如 30 Hz）下，`max_delay` 能盖住稳态 RTT；冷启动超时由 Client 配置  
- [ ] **不要**把 80-D 或归一化向量发给 Client  

---

## 9. 最小联调步骤（Server）

1. 启动 server，用 `WebsocketClientPolicy` 连接，打印 metadata。  
2. 构造假观测：`state =` 合法 EEF10（可用当前臂 FK），两张 `uint8` 图，`prompt`。  
3. 调用 `infer`，检查 `actions.shape == (H, 10)`，`xyz` 量级为米，`grip ∈ [-1,1]`，rot6d 两列近似单位正交。  
4. （RTC）带 `prefix` 再请求，断言前缀 `array_equal`。  
5. 真机侧用同一帧 `actions[0]` 走 `EndPoseCtrl` 看是否与预期位姿一致（小步验证）。

---

## 10. 非目标 / 常见误区

| 误区 | 正确做法 |
|---|---|
| 返回 80-D | 只返回 gather 后的 10-D raw |
| 返回归一化 EEF10 | 线缆必须物理单位 |
| 当关节 7D 对接现成 `rtc_main.py` | 需 EEF 版客户端或 Server 转关节（本说明是 EEF-only） |
| 夹爪当米 | 第 9 维是 `[-1,1]` |
| rot6d 当欧拉 | Client 负责 rot6d→RPY；Server 保持 rot6d |
| Server 缓存上一 chunk | 无状态；prefix 以请求为准 |

---

## 11. 给 Server Agent 的一句话

实现一个 **openpi 兼容的 WebSocket Policy Server**：首帧报 EEF10 metadata；`infer` 吃 Piper 图像 + 物理 EEF10 state（及可选 RTC prefix），吐出物理 EEF10 `actions[H,10]`；归一化与 80-D 统一空间留在进程内，**永不出现在线缆上**。真机将用 `EndPoseCtrl` 直接执行，无需 Server 提供关节逆解。