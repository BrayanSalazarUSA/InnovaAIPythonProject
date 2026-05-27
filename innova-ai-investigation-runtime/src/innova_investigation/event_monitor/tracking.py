from __future__ import annotations

import warnings

import numpy as np
import supervision as sv

from .models import Detection


class ObjectTracker:
    """Assigns stable track IDs to detections across frames using ByteTrack.

    Tracking is what lets the monitor say "this is still the same vehicle" and
    avoid writing a new evidence image every time YOLO sees it again.
    """

    def __init__(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.tracker = sv.ByteTrack()

    def update(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        xyxy = np.array([detection.bbox for detection in detections], dtype=float)
        confidence = np.array([detection.confidence for detection in detections], dtype=float)
        class_id = np.array([detection.class_id if detection.class_id is not None else -1 for detection in detections], dtype=int)
        source_index = np.arange(len(detections), dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            data={"source_index": source_index},
        )
        tracked = self.tracker.update_with_detections(sv_detections)

        output = list(detections)
        tracker_ids = tracked.tracker_id
        if tracker_ids is None:
            return output

        tracked_source_indexes = tracked.data.get("source_index", [])
        for tracked_index, tracker_id in zip(tracked_source_indexes, tracker_ids, strict=False):
            source = int(tracked_index)
            if 0 <= source < len(output):
                output[source].track_id = int(tracker_id)
        return output
