from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


RuleType = Literal["class", "vehicle_color", "person_upper_color", "custom_model"]


class EventRule(BaseModel):
    """One rule that decides whether a detection should be saved as evidence.

    The MVP supports simple rules first:
    - class: save detections by YOLO class name, e.g. person/car/truck.
    - vehicle_color: save vehicle detections whose crop looks like a color.
    - person_upper_color: save people whose upper-body crop looks like a color.
    - custom_model: same idea as class, but intended for a Roboflow/Ultralytics
      model placed locally as a .pt file.
    """

    name: str
    type: RuleType = "class"
    class_names: list[str] = Field(default_factory=list)
    min_confidence: float = 0.35
    color: str | None = None
    color_threshold: float | None = None
    model_path: str | None = None
    cooldown_seconds: float = 8.0


class MonitorConfig(BaseModel):
    """Configuration for one camera or video source."""

    camera_id: str = "demo-camera"
    camera_name: str = "Demo Camera"
    source: str
    nvr_id: str = ""
    nvr_name: str = ""
    vendor: str = ""
    host: str = ""
    http_port: int = 0
    sdk_port: int = 0
    rtsp_port: int = 554
    logical_channel: int = 1
    stream_variant: str = "main"
    username: str = ""
    password: str = ""
    output_dir: str = "output/event_monitor"
    yolo_model: str = "yolo11n.pt"
    sample_every_seconds: float = 0.25
    max_runtime_seconds: float | None = None
    enable_tracking: bool = True
    save_once_per_track: bool = True
    tracker_type: str = "botsort.yaml"
    tracker_confidence: float = 0.1
    enable_reid_merge: bool = True
    reid_merge_seconds: float = 180.0
    reid_similarity_threshold: float = 0.86
    save_annotated_frames: bool = True
    rules: list[EventRule]


class Detection(BaseModel):
    """Normalized detection returned by a detector."""

    class_name: str
    class_id: int | None = None
    confidence: float
    bbox: tuple[int, int, int, int]
    model_name: str
    track_id: int | None = None
    object_id: int | None = None


class SavedEvent(BaseModel):
    """Metadata persisted next to every saved event."""

    event_id: str
    rule_name: str
    rule_type: RuleType
    camera_id: str
    camera_name: str
    source: str
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: int | None = None
    object_id: int | None = None
    timestamp_utc: datetime
    video_seconds: float
    frame_index: int
    crop_path: str
    frame_path: str
    annotated_frame_path: str = ""
    extra: dict[str, object] = Field(default_factory=dict)


def resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    """Resolve config paths while keeping absolute paths untouched."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path
