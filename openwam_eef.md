# OpenWAM EEF10 真机推理

同步客户端：`examples/piper/eef_main.py`。  
Server 需已启动，并按物理 raw EEF10（10 维）收发 `observation/state` / `actions`。

## 1. 启动 CAN

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

## 2. 启动真机客户端

按实际相机序列号、任务提示和 OpenWAM server 地址修改参数：

```bash
uv run --no-sync examples/piper/eef_main.py \
    --host=192.168.3.37 \
    --port=8000 \
    --prompt="Put this carrot and this banana on the plate" \
    --head-camera-serial=339322074804 \
    --wrist-camera-serial=346522074547 \
    --can-name=can0 \
    --control-hz=30 \
    --open-loop-horizon=30 \
    --gripper-max-m=0.07 \
    --show-cameras
```

夹爪连续控制（不做开合二值化）：

```bash
uv run --no-sync examples/piper/eef_main.py \
    --host=127.0.0.1 \
    --port=8848 \
    --prompt="Put this carrot and this banana on the plate" \
    --head-camera-serial=339322074804 \
    --wrist-camera-serial=346522074547 \
    --can-name=can0 \
    --control-hz=30 \
    --open-loop-horizon=30 \
    --gripper-max-m=0.07 \
    --no-binarize-gripper
```

## 3. 异步 RTC 真机客户端

Server 需已支持 RTC（`training_rtc=true`、`max_delay`、硬前缀回显），动作仍为物理 EEF10。  
不接受 `--open-loop-horizon`；用 `--rtc-delay`（须 `≤ max_delay`）。

```bash
uv run --no-sync examples/piper/eef_rtc_main.py \
    --host=127.0.0.1 \
    --port=8848 \
    --prompt="Put this carrot and this banana on the plate" \
    --head-camera-serial=339322074804 \
    --wrist-camera-serial=346522074547 \
    --can-name=can0 \
    --control-hz=30 \
    --rtc-delay=8 \
    --gripper-max-m=0.07 \
    --no-binarize-gripper
```
