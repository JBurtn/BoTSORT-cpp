from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from botsort import _botsort

if TYPE_CHECKING:
    from .config import (
        BotSortConfig,
        MethodParams
    )

class BotSort:

    def __init__(
        self,
        config: "BotSortConfig",
        gmc_config: "MethodParams" | None = None,
        enable_gmc: bool = True,
        enable_reid: bool = False,
    ):

        if gmc_config is not None:
            config.gmc_enabled = enable_gmc
            config.setGMCMethod(gmc_config)
        
        config.reid_enabled = enable_reid
        self.config = config
        self.gmc_config = gmc_config
        self.tracker = None

    def init_tracker(self, frame_rate=24):
        self.config.frame_rate = int(round(frame_rate))
        self.config.track_buffer = int(round(frame_rate))

        self.tracker = _botsort.BotSort(
            self.config.c_init(),
            self.gmc_config.c_init
        )

    def track(
        self,
        bounding_boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        frame: np.ndarray,
    ) -> np.ndarray:

        if (bounding_boxes.ndim == 2 and bounding_boxes.shape[1] != 4)\
            or (bounding_boxes.ndim == 1 and bounding_boxes.shape[0] % 4 != 0):
            raise ValueError("bounding boxes do not have shape: (n, 4) found: "\
                            f"({bounding_boxes.shape[0]}, {bounding_boxes.shape[1]})")
        elif bounding_boxes.ndim > 2:
            raise ValueError("bounding boxes do not have correct ndim of 2 or less")

        if scores.ndim > 2:
            raise ValueError("scores do not have correct ndim of 2 or less")
        
        if class_ids.ndim > 2:
            raise ValueError("class ids do not have correct ndim of 2 or less")
        if frame.ndim != 3:
            raise ValueError("frame does not have correct ndim of 3")
        
        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)

        if class_ids.size != scores.size:    
            raise ValueError("Invalid size for class ids or scores: "
                             + str(class_ids.size) + ", "
                             + str(scores.size))
        
        frame = np.ascontiguousarray(frame)
        bounding_boxes[..., ::2] = \
            bounding_boxes[..., ::2].clip(min=0, max=frame.shape[1])
        
        bounding_boxes[..., 1::2] = \
            bounding_boxes[..., 1::2].clip(min=0, max=frame.shape[0])

        bounding_boxes = bounding_boxes.astype(np.float32).flatten()
        scores = scores.astype(np.float32).flatten()
        class_ids = class_ids.astype(np.int64).flatten()
        
        result = self.tracker.track(
            bounding_boxes,
            scores,
            class_ids,
            frame
        )
        indices = result[:, 7]
        return result[:, :7], indices.astype(np.int32)

        