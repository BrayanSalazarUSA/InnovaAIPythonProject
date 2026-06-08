from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .. import config as runtime_config
from .detectors import DetectorRegistry, crop_detection
from .models import MonitorConfig, SavedEvent, resolve_path
from .rules import rule_matches
from .storage import EventStorage


ProgressCallback = Callable[[dict[str, object]], None]


class _FfmpegFrameReader:
    """Decode live streams through ffmpeg instead of OpenCV.

    OpenCV's FFmpeg wrapper is noisy and brittle with some HEVC/H.265 NVR
    streams. The live-view path already relies on ffmpeg/HLS, so this reader
    keeps the monitor closer to the path that operators know works.
    """

    width = 960
    height = 540

    def __init__(self, source: str, *, sample_every_seconds: float) -> None:
        self.source = source
        self.sample_every_seconds = max(0.1, float(sample_every_seconds or 0.25))
        self.proc: subprocess.Popen | None = None
        self.frame_size = self.width * self.height * 3

    def open(self) -> bool:
        if not shutil.which("ffmpeg"):
            return False
        fps = min(8.0, max(0.2, 1.0 / self.sample_every_seconds))
        vf = (
            f"fps={fps},"
            f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
            f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:black"
        )
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if self.source.lower().startswith("rtsp://"):
            command.extend(["-rtsp_transport", "tcp", "-timeout", "15000000"])
        command.extend([
            "-i",
            self.source,
            "-an",
            "-vf",
            vf,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        self.proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 2,
        )
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.proc or not self.proc.stdout:
            return False, None
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame.copy()

    def release(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def load_monitor_config(path: str | Path) -> MonitorConfig:
    """Load a JSON config file for one camera/source."""

    config_path = Path(path).expanduser()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return MonitorConfig.model_validate(payload)


class EventMonitor:
    """Reads a camera/video source and saves evidence when rules match.

    This first MVP is intentionally simple:
    - It samples a frame every N seconds.
    - It runs YOLO detections.
    - It evaluates rules.
    - It saves crop + full frame + JSON metadata.
    """

    def __init__(
        self,
        config: MonitorConfig,
        *,
        project_root: Path | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root or runtime_config.PROJECT_ROOT
        self.on_progress = on_progress
        self.output_dir = resolve_path(config.output_dir, base_dir=self.project_root)
        self.storage = EventStorage(config, output_dir=self.output_dir)
        self._stop_requested = False

        default_model_path = config.yolo_model
        # Keep simple model names like "yolo11n.pt" untouched so Ultralytics can
        # download/use its cache. Resolve explicit local paths from project root.
        if "/" in default_model_path or "\\" in default_model_path:
            default_model_path = str(resolve_path(default_model_path, base_dir=self.project_root))

        for rule in self.config.rules:
            if rule.model_path:
                rule.model_path = str(resolve_path(rule.model_path, base_dir=self.project_root))

        self.detectors = DetectorRegistry(default_model=default_model_path)

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> dict[str, object]:
        source_lower = self.config.source.lower()
        source_is_rtsp = source_lower.startswith("rtsp://")
        source_is_live_stream = source_is_rtsp or source_lower.startswith("http://") or source_lower.startswith("https://") or ".m3u8" in source_lower
        capture = None
        ffmpeg_reader = None
        if source_is_live_stream:
            ffmpeg_reader = _FfmpegFrameReader(
                self.config.source,
                sample_every_seconds=float(self.config.sample_every_seconds),
            )
            if not ffmpeg_reader.open():
                ffmpeg_reader = None

        if ffmpeg_reader is None:
            previous_capture_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            if source_is_rtsp:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp|stimeout;15000000|max_delay;500000|fflags;nobuffer"
                )
            capture = cv2.VideoCapture(self.config.source, cv2.CAP_FFMPEG)
            if not capture.isOpened():
                # If the source is a relative local video, resolve it from project root.
                local_source = resolve_path(self.config.source, base_dir=self.project_root)
                capture = cv2.VideoCapture(str(local_source), cv2.CAP_FFMPEG)
            if previous_capture_options is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous_capture_options
            if not capture.isOpened():
                raise RuntimeError(f"No pude abrir el stream/video: {self.config.source}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if capture is not None else 0.0
        if fps <= 0:
            # RTSP streams often do not report FPS reliably. We sample by wall clock below.
            fps = 15.0

        sample_every = max(0.1, float(self.config.sample_every_seconds))
        next_sample_at = 0.0
        started_monotonic = time.monotonic()
        frame_index = -1
        sampled_frames = 0
        saved_events: list[SavedEvent] = []
        last_saved_by_rule: dict[str, float] = {}
        object_states: dict[str, dict[str, object]] = {}
        object_registry: dict[int, dict[str, object]] = {}
        raw_track_to_object: dict[str, int] = {}
        next_object_id = 1
        recent_untracked_objects: list[dict[str, object]] = []

        self._emit({"stage": "start", "detail": "Monitor iniciado.", "events": 0})

        try:
            while not self._stop_requested:
                if ffmpeg_reader is not None:
                    ok, frame = ffmpeg_reader.read()
                else:
                    ok, frame = capture.read()
                if not ok or frame is None:
                    break

                frame_index += 1
                video_seconds = time.monotonic() - started_monotonic if ffmpeg_reader is not None else frame_index / fps
                runtime_seconds = time.monotonic() - started_monotonic
                if self.config.max_runtime_seconds and runtime_seconds >= self.config.max_runtime_seconds:
                    break

                # For file videos we sample using video timestamp. For RTSP, this is still good enough
                # for the MVP because frame_index/fps advances roughly with time.
                if video_seconds < next_sample_at:
                    continue
                next_sample_at = video_seconds + sample_every
                sampled_frames += 1

                default_rules = [rule for rule in self.config.rules if rule.type != "custom_model"]
                custom_rules = [rule for rule in self.config.rules if rule.type == "custom_model"]
                detections_by_rule_name = self._detect_for_rules(frame, default_rules, custom_rules)

                # Evaluate more specific attribute rules first. If a black car
                # matches both "vehicle" and "black_vehicle", we keep one
                # object event and prefer the richer black_vehicle metadata.
                ordered_rules = sorted(self.config.rules, key=self._rule_priority)
                for rule in ordered_rules:
                    for detection in detections_by_rule_name.get(rule.name, []):
                        matched, extra = rule_matches(rule, detection, frame)
                        if not matched:
                            continue

                        object_key = ""
                        if detection.track_id is not None:
                            object_id, next_object_id = self._resolve_object_id(
                                detection,
                                frame,
                                video_seconds,
                                object_registry,
                                raw_track_to_object,
                                next_object_id,
                            )
                            detection.object_id = object_id
                            object_key = f"{detection.class_name}:object:{object_id}"
                            extra = {
                                **extra,
                                "track_id": detection.track_id,
                                "object_id": object_id,
                                "object_key": object_key,
                            }
                        elif self.config.save_once_per_track and self._recent_untracked_match(
                            recent_untracked_objects,
                            detection,
                            video_seconds,
                        ):
                            continue

                        quality = self._event_quality(detection, frame)
                        state = object_states.get(object_key) if object_key else None
                        if state and self.config.save_once_per_track:
                            current_specificity = int(state.get("specificity") or 0)
                            next_specificity = self._rule_specificity(rule)
                            current_quality = float(state.get("quality") or 0.0)
                            if next_specificity < current_specificity:
                                self.storage.touch_object_summary(object_key=object_key, detection=detection, video_seconds=video_seconds)
                                continue
                            if next_specificity == current_specificity and quality <= current_quality + 0.04:
                                self.storage.touch_object_summary(object_key=object_key, detection=detection, video_seconds=video_seconds)
                                continue
                            event_id = str(state["event_id"])
                        else:
                            event_id = None
                            cooldown_key = object_key or f"{rule.name}:{detection.class_name}"
                            last_saved = last_saved_by_rule.get(cooldown_key, -999999.0)
                            if video_seconds - last_saved < max(0.0, rule.cooldown_seconds):
                                continue
                            last_saved_by_rule[cooldown_key] = video_seconds

                        event = self.storage.save(
                            frame=frame,
                            detection=detection,
                            rule=rule,
                            frame_index=frame_index,
                            video_seconds=video_seconds,
                            extra=extra,
                            event_id=event_id,
                        )
                        if state is None:
                            saved_events.append(event)
                        if object_key:
                            object_states[object_key] = {
                                "event_id": event.event_id,
                                "quality": quality,
                                "specificity": self._rule_specificity(rule),
                                "rule_name": rule.name,
                            }
                        if detection.track_id is None:
                            self._remember_untracked_object(recent_untracked_objects, detection, video_seconds)
                        self._emit(
                            {
                                "stage": "event",
                                "detail": self._event_detail(rule.name, detection),
                                "event_id": event.event_id,
                                "track_id": detection.track_id,
                                "object_id": detection.object_id,
                                "events": len(saved_events),
                            }
                        )

                if sampled_frames % 10 == 0:
                    self._emit(
                        {
                            "stage": "running",
                            "detail": f"Frames muestreados: {sampled_frames}; eventos: {len(saved_events)}",
                            "sampled_frames": sampled_frames,
                            "events": len(saved_events),
                            "video_seconds": round(video_seconds, 2),
                        }
                    )
        finally:
            if capture is not None:
                capture.release()
            if ffmpeg_reader is not None:
                ffmpeg_reader.release()

        if sampled_frames <= 0:
            raise RuntimeError(
                "El stream abrió, pero no entregó frames decodificables. "
                "Probable causa: RTSP/HEVC inestable, codec H.265 corrupto o timeout de red."
            )

        summary = {
            "ok": True,
            "camera_id": self.config.camera_id,
            "camera_name": self.config.camera_name,
            "sampled_frames": sampled_frames,
            "events_saved": len(saved_events),
            "objects_saved": len(object_states) + len(recent_untracked_objects),
            "output_dir": str(self.output_dir),
        }
        self._emit({"stage": "done", "detail": "Monitor finalizado.", **summary})
        return summary

    def _detect_for_rules(self, frame, default_rules, custom_rules):
        detections_by_rule_name: dict[str, list] = {}

        if default_rules:
            if self.config.enable_tracking:
                detections = self.detectors.default_detector.track(
                    frame,
                    min_confidence=float(self.config.tracker_confidence),
                    tracker=self.config.tracker_type,
                )
            else:
                min_confidence = min(float(rule.min_confidence) for rule in default_rules)
                detections = self.detectors.default_detector.detect(frame, min_confidence=min_confidence)
            for rule in default_rules:
                detections_by_rule_name[rule.name] = detections

        for rule in custom_rules:
            detector = self.detectors.detector_for_rule(rule)
            detections_by_rule_name[rule.name] = detector.detect(frame, min_confidence=rule.min_confidence)

        return detections_by_rule_name

    @staticmethod
    def _event_detail(rule_name: str, detection) -> str:
        track = f" track #{detection.track_id}" if detection.track_id is not None else ""
        obj = f" object #{detection.object_id}" if detection.object_id is not None else ""
        return f"Evento guardado: {rule_name} ({detection.class_name}{track}{obj})"

    @staticmethod
    def _rule_priority(rule) -> int:
        if rule.type in {"vehicle_color", "person_upper_color", "custom_model"}:
            return 0
        return 10

    @staticmethod
    def _rule_specificity(rule) -> int:
        if rule.type in {"vehicle_color", "person_upper_color", "custom_model"}:
            return 2
        return 1

    @staticmethod
    def _event_quality(detection, frame) -> float:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        area_ratio = bbox_area / float(max(1, width * height))
        # Confidence is primary; larger/closer crops usually make better
        # evidence. Cap the area contribution so a giant partial object does
        # not always win.
        return float(detection.confidence) + min(0.25, area_ratio * 1.5)

    def _resolve_object_id(
        self,
        detection,
        frame,
        video_seconds: float,
        object_registry: dict[int, dict[str, object]],
        raw_track_to_object: dict[str, int],
        next_object_id: int,
    ) -> tuple[int, int]:
        raw_key = f"{detection.class_name}:track:{detection.track_id}"
        if raw_key in raw_track_to_object:
            object_id = raw_track_to_object[raw_key]
            self._update_object_registry(object_registry, object_id, detection, frame, video_seconds)
            return object_id, next_object_id

        object_id = self._find_matching_object(detection, frame, video_seconds, object_registry)
        if object_id is None:
            object_id = next_object_id
            next_object_id += 1

        raw_track_to_object[raw_key] = object_id
        self._update_object_registry(object_registry, object_id, detection, frame, video_seconds)
        return object_id, next_object_id

    def _find_matching_object(self, detection, frame, video_seconds: float, object_registry) -> int | None:
        if not self.config.enable_reid_merge:
            return None

        signature = self._appearance_signature(frame, detection)
        center = self._bbox_center(detection.bbox)
        frame_diag = float((frame.shape[0] ** 2 + frame.shape[1] ** 2) ** 0.5)
        best_id = None
        best_score = -1.0
        for object_id, state in object_registry.items():
            if state.get("class_name") != detection.class_name:
                continue
            age = video_seconds - float(state.get("last_seen_seconds") or 0.0)
            if age < 0 or age > self.config.reid_merge_seconds:
                continue

            similarity = self._cosine_similarity(signature, state.get("signature"))
            last_center = state.get("center") or center
            distance_ratio = self._point_distance(center, last_center) / max(1.0, frame_diag)
            max_distance = 0.85 if detection.class_name == "person" else 0.55
            if distance_ratio > max_distance:
                continue

            score = similarity - (distance_ratio * 0.25)
            if similarity >= self.config.reid_similarity_threshold and score > best_score:
                best_id = int(object_id)
                best_score = score
        return best_id

    def _update_object_registry(self, object_registry, object_id: int, detection, frame, video_seconds: float) -> None:
        signature = self._appearance_signature(frame, detection)
        existing = object_registry.get(object_id)
        if existing and existing.get("signature") is not None:
            signature = (np.asarray(existing["signature"], dtype=float) * 0.7) + (signature * 0.3)
            norm = np.linalg.norm(signature)
            if norm > 0:
                signature = signature / norm

        object_registry[object_id] = {
            "class_name": detection.class_name,
            "last_seen_seconds": video_seconds,
            "center": self._bbox_center(detection.bbox),
            "bbox": detection.bbox,
            "signature": signature,
        }

    @staticmethod
    def _appearance_signature(frame, detection) -> np.ndarray:
        crop = crop_detection(frame, detection)
        if crop.size == 0:
            return np.zeros(48, dtype=float)
        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256]).flatten()
        # Compress to keep this lightweight and avoid overfitting to tiny noise.
        hist = hist.reshape(24, 24).mean(axis=1)
        hist = np.concatenate([hist, np.array([float(crop.mean()), float(crop.std())])])
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _point_distance(a, b) -> float:
        return float(((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5)

    @staticmethod
    def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return intersection / float(area_a + area_b - intersection)

    def _recent_untracked_match(self, recent_objects, detection, video_seconds: float) -> bool:
        cutoff_seconds = 20.0
        recent_objects[:] = [item for item in recent_objects if video_seconds - float(item["video_seconds"]) <= cutoff_seconds]
        for item in recent_objects:
            if item["class_name"] != detection.class_name:
                continue
            if self._bbox_iou(item["bbox"], detection.bbox) >= 0.45:
                return True
        return False

    @staticmethod
    def _remember_untracked_object(recent_objects, detection, video_seconds: float) -> None:
        recent_objects.append(
            {
                "class_name": detection.class_name,
                "bbox": detection.bbox,
                "video_seconds": video_seconds,
            }
        )

    def _emit(self, update: dict[str, object]) -> None:
        if self.on_progress is not None:
            self.on_progress(update)


def run_monitor_from_config(config_path: str | Path) -> dict[str, object]:
    config = load_monitor_config(config_path)

    def print_progress(update: dict[str, object]) -> None:
        stage = update.get("stage", "-")
        detail = update.get("detail", "")
        print(f"[{stage}] {detail}", flush=True)

    return EventMonitor(config, on_progress=print_progress).run()
