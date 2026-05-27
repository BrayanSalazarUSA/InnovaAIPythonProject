from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .detectors import crop_detection, draw_detection
from .models import Detection, EventRule, MonitorConfig, SavedEvent


class EventStorage:
    """Persists frame, crop and metadata for every event."""

    def __init__(self, config: MonitorConfig, *, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.events_dir = self.output_dir / "events"
        self.frames_dir = self.output_dir / "frames"
        self.crops_dir = self.output_dir / "crops"
        self.metadata_dir = self.output_dir / "metadata"
        self.tracks_dir = self.output_dir / "tracks"
        for directory in (self.events_dir, self.frames_dir, self.crops_dir, self.metadata_dir, self.tracks_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "events.jsonl"
        self.objects_path = self.output_dir / "objects.jsonl"

    def save(
        self,
        *,
        frame: np.ndarray,
        detection: Detection,
        rule: EventRule,
        frame_index: int,
        video_seconds: float,
        extra: dict[str, object],
        event_id: str | None = None,
    ) -> SavedEvent:
        now = datetime.now(timezone.utc)
        event_id = event_id or uuid.uuid4().hex
        safe_rule = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in rule.name.lower())
        stem = f"{event_id[:12]}_{safe_rule}"

        crop = crop_detection(frame, detection)
        crop_path = self.crops_dir / f"{stem}_crop.jpg"
        frame_path = self.frames_dir / f"{stem}_frame.jpg"
        annotated_path = self.frames_dir / f"{stem}_annotated.jpg"

        cv2.imwrite(str(crop_path), crop)
        cv2.imwrite(str(frame_path), frame)
        if self.config.save_annotated_frames:
            track_label = f" #{detection.track_id}" if detection.track_id is not None else ""
            label = f"{rule.name}{track_label} | {detection.class_name} {detection.confidence:.2f}"
            annotated = draw_detection(frame, detection, label)
            cv2.imwrite(str(annotated_path), annotated)
        else:
            annotated_path = Path("")

        event = SavedEvent(
            event_id=event_id,
            rule_name=rule.name,
            rule_type=rule.type,
            camera_id=self.config.camera_id,
            camera_name=self.config.camera_name,
            source=self.config.source,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
            track_id=detection.track_id,
            object_id=detection.object_id,
            timestamp_utc=now,
            video_seconds=round(video_seconds, 3),
            frame_index=frame_index,
            crop_path=str(crop_path),
            frame_path=str(frame_path),
            annotated_frame_path=str(annotated_path) if annotated_path else "",
            extra=extra,
        )

        metadata_path = self.metadata_dir / f"{stem}.json"
        payload = event.model_dump(mode="json")
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._upsert_event(payload)
        self._upsert_object_summary(payload)
        if detection.object_id is not None:
            self._save_track_index(event, payload)
        return event

    def _upsert_event(self, payload: dict[str, object]) -> None:
        event_id = payload.get("event_id")
        lines: list[str] = []
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                try:
                    existing = json.loads(line)
                except Exception:
                    lines.append(line)
                    continue
                if existing.get("event_id") != event_id:
                    lines.append(line)
        lines.append(json.dumps(payload, ensure_ascii=False))
        self.index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _upsert_object_summary(self, event: dict[str, object]) -> None:
        object_key = self._object_key(event)
        existing_objects = self._load_jsonl(self.objects_path)
        current = next((item for item in existing_objects if item.get("object_key") == object_key), None)
        summary = self._merge_object_summary(current, event, object_key)

        updated = [item for item in existing_objects if item.get("object_key") != object_key]
        updated.append(summary)
        updated.sort(key=lambda item: str(item.get("last_seen_utc") or ""), reverse=True)
        self.objects_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in updated) + "\n", encoding="utf-8")

    def touch_object_summary(
        self,
        *,
        object_key: str,
        detection: Detection,
        video_seconds: float,
    ) -> None:
        existing_objects = self._load_jsonl(self.objects_path)
        updated = False
        now = datetime.now(timezone.utc).isoformat()
        for item in existing_objects:
            if item.get("object_key") != object_key:
                continue
            first_video = float(item.get("first_video_seconds") or video_seconds)
            last_video = max(float(item.get("last_video_seconds") or 0.0), float(video_seconds))
            item["last_seen_utc"] = now
            item["last_video_seconds"] = round(last_video, 3)
            item["duration_seconds"] = round(max(0.0, last_video - first_video), 3)
            item["track_ids"] = self._merge_unique(list(item.get("track_ids") or []), detection.track_id)
            updated = True
            break
        if updated:
            existing_objects.sort(key=lambda item: str(item.get("last_seen_utc") or item.get("timestamp_utc") or ""), reverse=True)
            self.objects_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in existing_objects) + "\n", encoding="utf-8")

    @staticmethod
    def _object_key(event: dict[str, object]) -> str:
        class_name = str(event.get("class_name") or "object")
        object_id = event.get("object_id")
        track_id = event.get("track_id")
        if object_id is not None:
            return f"{class_name}:object:{object_id}"
        if track_id is not None:
            return f"{class_name}:track:{track_id}"
        return f"{class_name}:event:{event.get('event_id')}"

    @staticmethod
    def _merge_unique(values: list[object], value: object) -> list[object]:
        merged = list(values or [])
        if value is not None and value not in merged:
            merged.append(value)
        return merged

    def _merge_object_summary(self, current: dict[str, object] | None, event: dict[str, object], object_key: str) -> dict[str, object]:
        current = current or {}
        first_video = float(current.get("first_video_seconds") if current.get("first_video_seconds") is not None else event.get("video_seconds") or 0.0)
        last_video = max(float(current.get("last_video_seconds") or 0.0), float(event.get("video_seconds") or 0.0))
        best_confidence = max(float(current.get("best_confidence") or 0.0), float(event.get("confidence") or 0.0))
        extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
        attributes = dict(current.get("attributes") or {})
        if extra.get("target_color") or extra.get("dominant_color"):
            attributes["color"] = extra.get("target_color") or extra.get("dominant_color")
            attributes["color_score"] = extra.get("color_score")
            attributes["dominant_color"] = extra.get("dominant_color")

        summary = {
            **event,
            "object_key": object_key,
            "first_seen_utc": current.get("first_seen_utc") or event.get("timestamp_utc"),
            "last_seen_utc": event.get("timestamp_utc"),
            "first_video_seconds": round(first_video, 3),
            "last_video_seconds": round(last_video, 3),
            "duration_seconds": round(max(0.0, last_video - first_video), 3),
            "best_confidence": round(best_confidence, 4),
            "best_event_id": event.get("event_id"),
            "rule_names": self._merge_unique(list(current.get("rule_names") or []), event.get("rule_name")),
            "track_ids": self._merge_unique(list(current.get("track_ids") or []), event.get("track_id")),
            "attributes": attributes,
            "event": event,
        }
        return summary

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        items: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        return items

    def _save_track_index(self, event: SavedEvent, payload: dict[str, object]) -> None:
        object_label = event.object_id if event.object_id is not None else event.track_id
        track_dir = self.tracks_dir / f"{event.class_name}_{object_label}"
        track_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = track_dir / "metadata.json"
        existing: dict[str, object] = {}
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        first_seen = existing.get("first_seen_utc") or payload.get("timestamp_utc")
        best_confidence = max(float(existing.get("best_confidence") or 0.0), float(event.confidence))
        track_payload = {
            "track_id": event.track_id,
            "object_id": event.object_id,
            "rule_name": event.rule_name,
            "class_name": event.class_name,
            "camera_id": event.camera_id,
            "camera_name": event.camera_name,
            "first_seen_utc": first_seen,
            "last_seen_utc": payload.get("timestamp_utc"),
            "best_confidence": round(best_confidence, 4),
            "event_id": event.event_id,
            "event": payload,
        }
        metadata_path.write_text(json.dumps(track_payload, indent=2), encoding="utf-8")


def _load_events_from_index(index_path: Path, *, limit: int) -> list[dict[str, object]]:
    lines = index_path.read_text(encoding="utf-8").splitlines()
    events: list[dict[str, object]] = []
    for line in lines[-limit:]:
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def _index_paths_for(output_dir: Path, index_name: str) -> list[Path]:
    """Find JSONL indexes for a requested output directory.

    A camera normally writes to its own folder, for example
    output/event_monitor/demo-camera-1/events.jsonl. When the UI asks for the
    parent output/event_monitor folder, we merge the child indexes so a manual
    camera selection does not hide events that were already captured.

    Some UI paths include a transient camera/channel id like output/event_monitor/364.
    If that folder does not exist, fall back to the existing parent monitor folder
    and merge its camera children.
    """
    direct_index = output_dir / index_name
    if direct_index.exists():
        return [direct_index]

    search_root = output_dir
    if not search_root.exists() and search_root.parent.exists():
        search_root = search_root.parent

    if not search_root.exists():
        return []

    return sorted(search_root.glob(f"*/{index_name}"))


def load_recent_events(output_dir: Path, *, limit: int = 100) -> list[dict[str, object]]:
    """Load recent monitor events."""
    index_paths = _index_paths_for(output_dir, "events.jsonl")
    if not index_paths:
        return []

    events: list[dict[str, object]] = []
    for index_path in index_paths:
        if index_path.exists():
            events.extend(_load_events_from_index(index_path, limit=limit))

    events.sort(key=lambda item: str(item.get("timestamp_utc") or ""), reverse=True)
    return events[:limit]


def load_recent_objects(output_dir: Path, *, limit: int = 100) -> list[dict[str, object]]:
    index_paths = _index_paths_for(output_dir, "objects.jsonl")
    objects: list[dict[str, object]] = []
    for index_path in index_paths:
        if index_path.exists():
            objects.extend(EventStorage._load_jsonl(index_path))
    objects.sort(key=lambda item: str(item.get("last_seen_utc") or item.get("timestamp_utc") or ""), reverse=True)
    return objects[:limit]
