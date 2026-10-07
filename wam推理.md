## 真机启动

模型推理

```bash
conda activate openwam

python -c 'import runpy, torch; torch.cuda.set_per_process_memory_fraction(0.9, 0); runpy.run_path("scripts/deploy.py", run_name="__main__")' \
  --ckpt-dir /media/ubun/16T/checkpoints/openwam \
  --device cuda:0 \
  --protocol openpi \
  --port 8000
```

启动can口

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

启动控制

```bash
uv run --no-sync example/eef_main.py \
    --host=127.0.0.1 \
    --port=8857 \
    --prompt="pick up the block." \
    --head-camera-serial=339322074804 \
    --wrist-camera-serial=346522074547 \
    --can-name=can0 \
    --control-hz=30 \
    --open-loop-horizon=30 \
    --gripper-max-m=0.07 \
    --no-binarize-gripper
```

    --prompt="pick up the block." \

    --prompt="Retrieve the book titled 《吕思勉文选》 and place it in the black grid on the right." \

```bash
uv run --no-sync example/eef_rtc_main.py \
    --host=127.0.0.1 \
    --port=8848 \
    --prompt="pick up the block." \
    --head-camera-serial=339322074804 \
    --wrist-camera-serial=346522074547 \
    --can-name=can0 \
    --control-hz=50 \
    --rtc-delay=4 \
    --gripper-max-m=0.07 \
    --no-binarize-gripper
```