"""Recognize-book Piper reader: native dual-camera vertical stack (no resize).

Same EEF10 / unify / T5-cache contract as ``BananaPiperDataset``. Vision differs:
``observation.images.top_head`` is pasted above ``observation.images.hand_right``
at native resolution (480x640 + 480x640 → 960x640). Decode never goes through
the L-shape slot resize used by the banana / pick_blocks / book_piper L-layout.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from openwam.dataloader.banana_piper import BananaPiperDataset
from openwam.dataloader.transforms.multiview import assemble_vstack_native
from openwam.dataloader.utils.video_io import decode_video_frames as _decode_video_frames

logger = logging.getLogger(__name__)

VSTACK_CANVAS_H = 960
VSTACK_CANVAS_W = 640
NATIVE_CAM_H = 480
NATIVE_CAM_W = 640


class BookPiperVstackDataset(BananaPiperDataset):
    """Banana Piper EEF pipeline with native head/wrist vertical concat."""

    DATASET_NAME = "BookPiperVstack"

    def _expected_canvas_hw(self) -> Tuple[int, int]:
        return (VSTACK_CANVAS_H, VSTACK_CANVAS_W)

    def _post_init(self, info: dict) -> None:
        super()._post_init(info)
        features = info.get("features", {}) or {}
        for cam in (self._head_camera, self._right_wrist_camera):
            if cam is None:
                continue
            shape = (features.get(cam) or {}).get("shape")
            if not shape or len(shape) < 2:
                continue
            h, w = int(shape[0]), int(shape[1])
            if (h, w) != (NATIVE_CAM_H, NATIVE_CAM_W):
                logger.warning(
                    "BookPiperVstack(%s): %s native shape is %sx%s, expected %sx%s",
                    self._dataset_id,
                    cam,
                    h,
                    w,
                    NATIVE_CAM_H,
                    NATIVE_CAM_W,
                )
        self._native_pane_size = (NATIVE_CAM_W, NATIVE_CAM_H)

    def _decode_one_camera(self, camera, row, ep_local: int, offset: int, real_local_indices, *, kind: str) -> List:
        if camera is None:
            return []
        chunk_col = f"videos/{camera}/chunk_index"
        file_col = f"videos/{camera}/file_index"
        if chunk_col not in row.index:
            return []
        if camera not in self._ep_video_frame_offsets:
            return []

        def _load() -> List:
            path = self._dataset_dir / self._video_path_template.format(
                video_key=camera, chunk_index=int(row[chunk_col]), file_index=int(row[file_col])
            )
            v_base = int(self._ep_video_frame_offsets[camera][ep_local]) + offset
            frame_indices = (real_local_indices + v_base).tolist()
            return _decode_video_frames(
                str(path),
                frame_indices,
                NATIVE_CAM_H,
                NATIVE_CAM_W,
                keep_native=True,
            )

        if kind == "head":
            return _load()
        try:
            return _load()
        except self.WRIST_DECODE_TOLERATED as e:
            self._wrist_fail_count += 1
            if self._wrist_fail_count == 1 or self._wrist_fail_count % self._fail_log_every == 0:
                logger.warning(
                    "%s(%s): %d cumulative wrist decode failures (latest: %s, camera=%s)",
                    self.DATASET_NAME,
                    self._dataset_id,
                    self._wrist_fail_count,
                    type(e).__name__,
                    camera,
                )
            return []

    def _decode_window_video(
        self,
        row,
        ep_local: int,
        offset: int,
        real_local_indices,
        idx: int,
        *,
        return_missing_masks: bool = False,
    ):
        head_frames = self._decode_one_camera(
            self._head_camera, row, ep_local, offset, real_local_indices, kind="head"
        )
        if not head_frames:
            raise RuntimeError(
                f"empty head-video decode for {self.DATASET_NAME}({self._dataset_id}) at idx={idx}"
            )
        right_frames = (
            self._decode_one_camera(
                self._right_wrist_camera, row, ep_local, offset, real_local_indices, kind="wrist"
            )
            if self._right_wrist_camera
            else []
        )

        n_real = len(head_frames)
        if n_real < self._num_video_frames:
            pad = self._num_video_frames - n_real
            head_frames = head_frames + [head_frames[-1]] * pad
            if right_frames:
                right_frames = right_frames + [right_frames[-1]] * pad

        fallback = self._native_pane_size
        video = []
        missing_masks = [] if return_missing_masks else None
        for fi in range(self._num_video_frames):
            top = head_frames[fi]
            bottom = right_frames[fi] if right_frames else None
            assembled = assemble_vstack_native(
                top,
                bottom,
                fallback_size=fallback,
                return_missing_mask=return_missing_masks,
            )
            if return_missing_masks:
                frame, missing_mask = assembled
                video.append(frame)
                missing_masks.append(missing_mask)
            else:
                video.append(assembled)
            w, h = video[-1].size
            if (h, w) != (self._height, self._width):
                raise RuntimeError(
                    f"{self.DATASET_NAME}: vstack canvas {h}x{w} != configured "
                    f"{self._height}x{self._width} at idx={idx} frame={fi}"
                )
        if return_missing_masks:
            return video, missing_masks
        return video


__all__ = [
    "BookPiperVstackDataset",
    "NATIVE_CAM_H",
    "NATIVE_CAM_W",
    "VSTACK_CANVAS_H",
    "VSTACK_CANVAS_W",
]
