from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .models import Detection, EventRule


class YoloDetector:
    """Small wrapper around Ultralytics YOLO.

    It normalizes model output into our own Detection model so the rest of
    the feature does not depend directly on Ultralytics internals.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)

    @property
    def model_name(self) -> str:
        return Path(self.model_path).name

    def detect(self, frame: np.ndarray, *, min_confidence: float) -> list[Detection]:
        results = self.model(frame, conf=min_confidence, verbose=False)
        return self._results_to_detections(results)

    def track(self, frame: np.ndarray, *, min_confidence: float, tracker: str = "botsort.yaml") -> list[Detection]:
        results = self.model.track(frame, conf=min_confidence, persist=True, tracker=tracker, verbose=False)
        return self._results_to_detections(results)

    def _results_to_detections(self, results) -> list[Detection]:
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = str(self.model.names[class_id])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                track_id = int(box.id[0]) if getattr(box, "id", None) is not None else None
                detections.append(
                    Detection(
                        class_name=class_name,
                        class_id=class_id,
                        confidence=round(confidence, 4),
                        bbox=(x1, y1, x2, y2),
                        model_name=self.model_name,
                        track_id=track_id,
                    )
                )
        return detections


class DetectorRegistry:
    """Loads the default model plus optional custom rule models.

    A Roboflow dataset is normally exported/trained into a local `.pt` file.
    Put that file in `resources/event_monitor/models/` and reference it from
    a `custom_model` rule.
    """

    def __init__(self, *, default_model: str | Path) -> None:
        self.default_detector = YoloDetector(default_model)
        self.custom_detectors: dict[str, YoloDetector] = {}

    def detector_for_rule(self, rule: EventRule) -> YoloDetector:
        if rule.type != "custom_model" or not rule.model_path:
            return self.default_detector
        key = str(rule.model_path)
        if key not in self.custom_detectors:
            self.custom_detectors[key] = YoloDetector(key)
        return self.custom_detectors[key]


def crop_detection(frame: np.ndarray, detection: Detection) -> np.ndarray:
    """Crop a detection bbox and clamp it safely inside frame boundaries."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return frame[y1:y2, x1:x2]


def draw_detection(frame: np.ndarray, detection: Detection, label: str) -> np.ndarray:
    annotated = frame.copy()
    x1, y1, x2, y2 = detection.bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(
        annotated,
        label,
        (x1, max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 220, 255),
        2,
    )
    return annotated
