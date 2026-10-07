"""Native vertical-stack compose for recognize-book Piper."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from openwam.dataloader.book_piper_vstack import BookPiperVstackDataset
from openwam.dataloader.transforms.multiview import assemble_vstack_native


def test_assemble_vstack_native_pixel_identity():
    top = Image.new("RGB", (640, 480), (12, 34, 56))
    bottom = Image.new("RGB", (640, 480), (200, 10, 10))
    canvas = assemble_vstack_native(top, bottom)
    assert canvas.size == (640, 960)
    arr = np.asarray(canvas)
    np.testing.assert_array_equal(arr[:480], np.asarray(top))
    np.testing.assert_array_equal(arr[480:], np.asarray(bottom))


def test_assemble_vstack_native_pads_narrower_without_resize():
    top = Image.new("RGB", (100, 20), (255, 0, 0))
    bottom = Image.new("RGB", (80, 10), (0, 255, 0))
    canvas, mask = assemble_vstack_native(top, bottom, return_missing_mask=True)
    assert canvas.size == (100, 30)
    arr = np.asarray(canvas)
    np.testing.assert_array_equal(arr[:20, :100], np.asarray(top))
    np.testing.assert_array_equal(arr[20:, :80], np.asarray(bottom))
    np.testing.assert_array_equal(arr[20:, 80:], 0)
    assert np.asarray(mask)[20:, 80:].min() == 255
    assert np.asarray(mask)[:20].max() == 0


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "derived/eef").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    (tmp_path / "splits").mkdir()
    cameras = ["observation.images.top_head", "observation.images.hand_right"]
    info = dict(
        codebase_version="v2.1",
        fps=30,
        chunks_size=1000,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        features={
            **{c: {"dtype": "video", "shape": [480, 640, 3]} for c in cameras},
            "action": {"shape": [7]},
            "observation.state": {"shape": [7]},
        },
    )
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    episodes = [dict(episode_index=i, length=6, tasks=["first task"]) for i in range(10)]
    (tmp_path / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, episodes)))
    tasks = [dict(task_index=0, task="first task")]
    (tmp_path / "meta/tasks.jsonl").write_text("\n".join(map(json.dumps, tasks)))
    (tmp_path / "derived/eef/meta.json").write_text(json.dumps({"format": "banana_piper_eef10_v1"}))
    (tmp_path / "splits/manifest.json").write_text(
        json.dumps({"splits": {"train": list(range(9)), "validation": [9]}})
    )
    for i in range(10):
        frame = np.arange(6)
        pd.DataFrame(
            {"episode_index": [i] * 6, "frame_index": frame, "task_index": [0] * 6}
        ).to_parquet(tmp_path / f"data/chunk-000/episode_{i:06d}.parquet")
        action = np.repeat((frame + i * 100).astype(np.float32)[:, None], 10, axis=1)
        state = action + 10
        np.savez(tmp_path / f"derived/eef/episode_{i:06d}.npz", state_eef10=state, action_eef10=action)
    return tmp_path


def _fake_camera(self, camera, row, ep, offset, indices, *, kind):
    n = max(int(len(indices)), 1)
    color = (12, 34, 56) if kind == "head" else (200, 10, 10)
    return [Image.new("RGB", (640, 480), color) for _ in range(n)]


def test_getitem_vstack_canvas_is_native_960x640(corpus, monkeypatch):
    monkeypatch.setattr(BookPiperVstackDataset, "_decode_one_camera", _fake_camera)
    ds = BookPiperVstackDataset(
        str(corpus),
        num_frames=5,
        video_stride=2,
        multiview=True,
        height=960,
        width=640,
        normalize_mode=None,
        unify_action=True,
        unify_action_map=["0-9"],
        color_jitter=False,
        use_t5_cache=False,
    )
    sample = ds._getitem_impl(0)
    assert len(sample["video"]) == 3
    frame = sample["video"][0]
    assert frame.size == (640, 960)
    arr = np.asarray(frame)
    np.testing.assert_array_equal(arr[:480], np.asarray(Image.new("RGB", (640, 480), (12, 34, 56))))
    np.testing.assert_array_equal(arr[480:], np.asarray(Image.new("RGB", (640, 480), (200, 10, 10))))
