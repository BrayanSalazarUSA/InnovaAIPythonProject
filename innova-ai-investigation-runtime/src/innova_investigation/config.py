from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(PROJECT_ROOT / ".env")

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
DOCS_DIR = PROJECT_ROOT / "docs"
VENDOR_DIR = PROJECT_ROOT / "vendor"
SDK_ARCHIVES_DIR = VENDOR_DIR / "sdk_archives"

DEFAULT_QUERY_IMAGE_PATH = RESOURCES_DIR / "examples" / "sanctuary-debris.png"
DEFAULT_VIDEO_PATH = RESOURCES_DIR / "videos" / "sanctuary-debris.mp4"
QUERY_IMAGE_PATH = DEFAULT_QUERY_IMAGE_PATH
VIDEO_PATH = DEFAULT_VIDEO_PATH

DEFAULT_FRAME_STEP = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.58
DEFAULT_MAX_RESULTS = 20
DEFAULT_ENABLE_PERSON_DETECTION = True
DEFAULT_PERSON_DETECTION_FRAME_STEP = 2
DEFAULT_PREVIEW_CALLBACK_SAMPLE_INTERVAL = 1
FRAME_STEP = DEFAULT_FRAME_STEP
SIMILARITY_THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD
MAX_RESULTS = DEFAULT_MAX_RESULTS
ENABLE_PERSON_DETECTION = DEFAULT_ENABLE_PERSON_DETECTION

PROCESSING_MAX_WIDTH = 960
PREVIEW_MAX_WIDTH = 1280
TOP_CANDIDATES_PER_FRAME = 4
TOP_TEMPLATE_MATCHES_PER_SCALE = 5
MAX_SCORING_CANDIDATES = 16
MIN_FEATURE_MATCHES = 10

DEFAULT_SHOW_PREVIEW = True
DEFAULT_SAVE_ANNOTATED_VIDEO = True
SHOW_PREVIEW = DEFAULT_SHOW_PREVIEW
PREVIEW_WINDOW_NAME = "AI Visual Search - CCTV"
PREVIEW_WAIT_MS = 1
DISPLAY_QUERY_THUMBNAIL = True
SAVE_ANNOTATED_VIDEO = DEFAULT_SAVE_ANNOTATED_VIDEO
DRAW_ZONE_GRID = True
ZONE_GRID_ROWS = 3
ZONE_GRID_COLS = 3
TRACE_MAX_POINTS = 140
EVIDENCE_CLIP_SECONDS_BEFORE = 4
EVIDENCE_CLIP_SECONDS_AFTER = 4
PREVIEW_TITLE = "Innova AI Investigation Runtime"
PREVIEW_SUBTITLE = "Visual investigation and CCTV evidence workflow."

BRAND_GOLD = (84, 168, 219)
BRAND_GOLD_SOFT = (125, 199, 232)
BRAND_BLUE = (143, 89, 25)
BRAND_BLUE_SOFT = (186, 139, 74)
BRAND_PANEL = (34, 26, 16)
BRAND_PANEL_ALT = (48, 36, 22)
BRAND_TEXT_PRIMARY = (245, 244, 240)
BRAND_TEXT_MUTED = (225, 219, 205)
BRAND_TEXT_ACCENT = (247, 220, 132)
BRAND_GRID = (83, 68, 41)
PERSON_BOX_COLOR = (214, 156, 96)
PERSON_BOX_ASSOCIATED_COLOR = (247, 220, 132)

TEMPLATE_SCALES = (0.35, 0.45, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45)
MIN_TEMPLATE_SCORE = 0.30
MIN_WINDOW_SIZE = 48
CONTOUR_PROPOSAL_LIMIT = 10
CONTOUR_EXPAND_RATIO = 0.16
CONTOUR_MIN_AREA_RATIO = 0.0025
CONTOUR_MAX_AREA_RATIO = 0.60
ASPECT_RATIO_TOLERANCE = 0.70
SIMILARITY_COLOR_WEIGHT_STRONG = 0.16
SIMILARITY_COLOR_WEIGHT_MEDIUM = 0.18
SIMILARITY_COLOR_WEIGHT_LOW_TEXTURE = 0.12

TEMPORAL_DUPLICATE_WINDOW_SECONDS = 1.5
IOU_DUPLICATE_THRESHOLD = 0.35
PERSON_DETECTION_TRIGGER_MODE = "always"
PERSON_DETECTION_FRAME_STEP = DEFAULT_PERSON_DETECTION_FRAME_STEP
PREVIEW_CALLBACK_SAMPLE_INTERVAL = DEFAULT_PREVIEW_CALLBACK_SAMPLE_INTERVAL
PERSON_CONFIDENCE_THRESHOLD = 0.18
PERSON_DETECTION_DEVICE = os.getenv("INNOVA_PERSON_DETECTION_DEVICE", "cpu")
PERSON_TRACK_MAX_AGE = 8
PERSON_TRACK_MAX_DISTANCE = 180
PERSON_TRACK_MIN_IOU = 0.08
PERSON_ASSOCIATION_DISTANCE = 220
MAX_ASSOCIATED_PEOPLE_PER_MATCH = 4
PERSON_NEAR_MATCH_BONUS = 0.14
PERSON_VISIBLE_MATCH_BONUS = 0.05
EARLY_MATCH_TIME_PENALTY_PER_SECOND = 0.002

BACKEND_API_BASE_URL = os.getenv("INNOVA_BACKEND_API_URL", "http://127.0.0.1:8080/api").rstrip("/")
PUBLIC_API_BASE_URL = os.getenv("INNOVA_PUBLIC_API_BASE_URL", "").strip().rstrip("/")
NVR_PROFILES_PATH = Path(os.getenv("INNOVA_NVR_PROFILES_PATH", str(RESOURCES_DIR / "nvr_profiles.local.json")))
SSH_KEY_PATH = Path(os.getenv("INNOVA_SSH_KEY_PATH", str(RESOURCES_DIR / "keys" / "elastic-beanstalk.pem")))
REMOTE_BRIDGE_HOST = os.getenv("INNOVA_REMOTE_BRIDGE_HOST", "127.0.0.1")
REMOTE_BRIDGE_USER = os.getenv("INNOVA_REMOTE_BRIDGE_USER", "ubuntu")
REMOTE_BRIDGE_PYTHON = os.getenv("INNOVA_REMOTE_BRIDGE_PYTHON", "python3")
HIKVISION_REMOTE_SDK_DIR = os.getenv(
    "INNOVA_HIKVISION_REMOTE_SDK_DIR",
    "/opt/innova/hikvision/current/lib",
)
DAHUA_REMOTE_SDK_DIR = os.getenv("INNOVA_DAHUA_REMOTE_SDK_DIR", "/opt/innova/dahua")
Dahua_LINUX_ARCHIVE = SDK_ARCHIVES_DIR / "General_NetSDK_Eng_Linux64_IS_V3.060.0000003.0.R.251127.tar.gz"
HIKVISION_LINUX_ARCHIVE = SDK_ARCHIVES_DIR / "EN-HCNetSDKV6.1.9.4_build20220412_linux64.zip"


def _resolve_yolo_model_path() -> Path:
    configured = Path(os.getenv("INNOVA_YOLO_MODEL_PATH", str(PROJECT_ROOT / "yolo11n.pt")))
    if configured.exists():
        return configured

    return configured


YOLO_MODEL_PATH = _resolve_yolo_model_path()


def _refresh_output_paths() -> None:
    global CROPS_DIR, ANNOTATIONS_DIR, CLIPS_DIR, PERSONS_DIR
    global REPORT_JSON_PATH, REPORT_CSV_PATH, SUMMARY_PATH, ANNOTATED_VIDEO_PATH

    CROPS_DIR = OUTPUT_DIR / "crops"
    ANNOTATIONS_DIR = OUTPUT_DIR / "annotated_frames"
    CLIPS_DIR = OUTPUT_DIR / "clips"
    PERSONS_DIR = OUTPUT_DIR / "persons"
    REPORT_JSON_PATH = OUTPUT_DIR / "report.json"
    REPORT_CSV_PATH = OUTPUT_DIR / "report.csv"
    SUMMARY_PATH = OUTPUT_DIR / "summary.txt"
    ANNOTATED_VIDEO_PATH = OUTPUT_DIR / "analysis_preview.mp4"


def apply_runtime_overrides(
    *,
    query_image_path: Path | str | None = None,
    video_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    show_preview: bool | None = None,
    save_annotated_video: bool | None = None,
    frame_step: int | None = None,
    similarity_threshold: float | None = None,
    max_results: int | None = None,
    enable_person_detection: bool | None = None,
    person_detection_trigger_mode: str | None = None,
    person_detection_frame_step: int | None = None,
    preview_callback_sample_interval: int | None = None,
) -> None:
    global QUERY_IMAGE_PATH, VIDEO_PATH, OUTPUT_DIR
    global SHOW_PREVIEW, SAVE_ANNOTATED_VIDEO
    global FRAME_STEP, SIMILARITY_THRESHOLD, MAX_RESULTS, ENABLE_PERSON_DETECTION
    global PERSON_DETECTION_TRIGGER_MODE, PERSON_DETECTION_FRAME_STEP, PREVIEW_CALLBACK_SAMPLE_INTERVAL

    if query_image_path is not None:
        QUERY_IMAGE_PATH = Path(query_image_path)
    if video_path is not None:
        VIDEO_PATH = Path(video_path)
    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir)
    if show_preview is not None:
        SHOW_PREVIEW = show_preview
    if save_annotated_video is not None:
        SAVE_ANNOTATED_VIDEO = save_annotated_video
    if frame_step is not None:
        FRAME_STEP = max(1, frame_step)
    if similarity_threshold is not None:
        SIMILARITY_THRESHOLD = float(similarity_threshold)
    if max_results is not None:
        MAX_RESULTS = max(1, max_results)
    if enable_person_detection is not None:
        ENABLE_PERSON_DETECTION = enable_person_detection
    if person_detection_trigger_mode is not None:
        PERSON_DETECTION_TRIGGER_MODE = person_detection_trigger_mode
    if person_detection_frame_step is not None:
        PERSON_DETECTION_FRAME_STEP = max(1, person_detection_frame_step)
    if preview_callback_sample_interval is not None:
        PREVIEW_CALLBACK_SAMPLE_INTERVAL = max(1, preview_callback_sample_interval)

    _refresh_output_paths()


def reset_runtime_overrides() -> None:
    apply_runtime_overrides(
        query_image_path=DEFAULT_QUERY_IMAGE_PATH,
        video_path=DEFAULT_VIDEO_PATH,
        output_dir=DEFAULT_OUTPUT_DIR,
        show_preview=DEFAULT_SHOW_PREVIEW,
        save_annotated_video=DEFAULT_SAVE_ANNOTATED_VIDEO,
        frame_step=DEFAULT_FRAME_STEP,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        max_results=DEFAULT_MAX_RESULTS,
        enable_person_detection=DEFAULT_ENABLE_PERSON_DETECTION,
        person_detection_trigger_mode="always",
        person_detection_frame_step=DEFAULT_PERSON_DETECTION_FRAME_STEP,
        preview_callback_sample_interval=DEFAULT_PREVIEW_CALLBACK_SAMPLE_INTERVAL,
    )


_refresh_output_paths()
