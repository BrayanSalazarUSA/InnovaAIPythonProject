from __future__ import annotations

import base64
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import config as runtime_config
from .similarity_search import SimilaritySearcher
from .video_processor import VideoProcessor


ProgressCallback = Callable[[str, str, float], None]


@dataclass(slots=True)
class QuickScanResult:
    report: dict[str, Any]
    top_hits: list[dict[str, Any]]


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rem = seconds - (minutes * 60)
    hours = minutes // 60
    minutes = minutes % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{rem:05.2f}".replace(".00", "")
    return f"{minutes:d}:{rem:05.2f}".replace(".00", "")


def resize_for_processing(frame: Any) -> tuple[Any, float]:
    height, width = frame.shape[:2]
    if width <= runtime_config.PROCESSING_MAX_WIDTH:
        return frame, 1.0
    ratio = runtime_config.PROCESSING_MAX_WIDTH / width
    new_size = (int(width * ratio), int(height * ratio))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), ratio


def resolve_zone(frame_shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    frame_height, frame_width = frame_shape
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    col = min(
        runtime_config.ZONE_GRID_COLS - 1,
        max(0, int(center_x / max(frame_width, 1) * runtime_config.ZONE_GRID_COLS)),
    )
    row = min(
        runtime_config.ZONE_GRID_ROWS - 1,
        max(0, int(center_y / max(frame_height, 1) * runtime_config.ZONE_GRID_ROWS)),
    )
    return f"{chr(65 + row)}{col + 1}"


def _roi_to_pixels(frame_shape: tuple[int, int], roi: dict[str, Any] | None) -> tuple[int, int, int, int] | None:
    if not isinstance(roi, dict):
        return None
    frame_height, frame_width = frame_shape
    try:
        x = float(roi.get("x", 0.0))
        y = float(roi.get("y", 0.0))
        width = float(roi.get("width", roi.get("w", 0.0)))
        height = float(roi.get("height", roi.get("h", 0.0)))
    except Exception:
        return None
    x1 = int(round(x * frame_width))
    y1 = int(round(y * frame_height))
    x2 = int(round((x + width) * frame_width))
    y2 = int(round((y + height) * frame_height))
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y2 = max(y1 + 1, min(frame_height, y2))
    return x1, y1, x2, y2


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0:
        return 0.0
    a_area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    b_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter_area / max(1.0, a_area + b_area - inter_area)


def _bbox_center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def _change_score(reference_roi: Any, candidate_roi: Any) -> float:
    if reference_roi is None or candidate_roi is None or reference_roi.size == 0 or candidate_roi.size == 0:
        return 0.0
    size = (160, 160)
    ref = cv2.resize(reference_roi, size, interpolation=cv2.INTER_AREA)
    cand = cv2.resize(candidate_roi, size, interpolation=cv2.INTER_AREA)
    ref_gray = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    cand_gray = cv2.GaussianBlur(cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    mean_delta = float(np.mean(cv2.absdiff(ref_gray, cand_gray))) / 255.0
    edge_delta = float(
        np.mean(
            cv2.absdiff(
                cv2.Canny(ref_gray, 40, 120),
                cv2.Canny(cand_gray, 40, 120),
            )
        )
    ) / 255.0
    return round(max(0.0, min(1.0, (mean_delta * 1.55) + (edge_delta * 0.45))), 4)


def _dark_object_score(roi_image: Any) -> float:
    if roi_image is None or roi_image.size == 0:
        return 0.0
    small = cv2.resize(roi_image, (180, 180), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    dark_fraction = float(np.mean(gray < 90))
    saturated_dark_fraction = float(np.mean((gray < 115) & (hsv[:, :, 1] > 24)))
    low_quartile_darkness = max(0.0, (120.0 - float(np.percentile(gray, 25))) / 120.0)
    score = (0.48 * dark_fraction) + (0.37 * saturated_dark_fraction) + (0.15 * low_quartile_darkness)
    return round(max(0.0, min(1.0, score)), 4)


def _dark_roi_metrics(roi_image: Any, baseline_roi: Any | None = None) -> dict[str, float]:
    if roi_image is None or roi_image.size == 0:
        return {
            "darkObjectScore": 0.0,
            "darkAreaFraction": 0.0,
            "largestDarkComponent": 0.0,
            "largestDarkComponentWidth": 0.0,
            "largestDarkComponentHeight": 0.0,
            "largestDarkComponentAspect": 0.0,
            "darkComponentEdgeContact": 0.0,
            "darkComponentTouchesBorder": 0.0,
            "darkStructurePenalty": 0.0,
            "darkObjectDelta": 0.0,
            "darkAreaDelta": 0.0,
            "darkComponentDelta": 0.0,
        }

    height, width = roi_image.shape[:2]
    max_width = 260
    if width > max_width:
        ratio = max_width / max(width, 1)
        size = (max_width, max(1, int(round(height * ratio))))
        small = cv2.resize(roi_image, size, interpolation=cv2.INTER_AREA)
    else:
        small = roi_image.copy()

    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    dark_mask = ((gray < 82) | ((gray < 118) & (hsv[:, :, 1] > 22))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    roi_area = float(max(1, dark_mask.shape[0] * dark_mask.shape[1]))
    dark_area_fraction = float(np.count_nonzero(dark_mask)) / roi_area
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    largest_component = 0.0
    largest_width_fraction = 0.0
    largest_height_fraction = 0.0
    largest_aspect = 0.0
    edge_contact = 0.0
    touches_border = 0.0
    if component_count > 1:
        largest_index = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_component = float(stats[largest_index, cv2.CC_STAT_AREA]) / roi_area
        comp_x = int(stats[largest_index, cv2.CC_STAT_LEFT])
        comp_y = int(stats[largest_index, cv2.CC_STAT_TOP])
        comp_w = int(stats[largest_index, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[largest_index, cv2.CC_STAT_HEIGHT])
        mask_height, mask_width = dark_mask.shape[:2]
        largest_width_fraction = float(comp_w) / float(max(1, mask_width))
        largest_height_fraction = float(comp_h) / float(max(1, mask_height))
        largest_aspect = float(comp_h) / float(max(1, comp_w))
        border_hits = int(comp_x <= 1)
        border_hits += int(comp_y <= 1)
        border_hits += int(comp_x + comp_w >= mask_width - 1)
        border_hits += int(comp_y + comp_h >= mask_height - 1)
        touches_border = 1.0 if border_hits > 0 else 0.0
        edge_contact = float(border_hits) / 4.0

    huge_component_score = min(1.0, max(0.0, (largest_component - 0.18) / 0.28))
    edge_score = max(edge_contact, touches_border * min(1.0, max(0.0, (largest_component - 0.08) / 0.18)))
    vertical_score = min(1.0, max(0.0, (largest_aspect - 1.45) / 1.2)) * min(
        1.0,
        max(0.0, (largest_height_fraction - 0.55) / 0.35),
    )
    dark_structure_penalty = max(
        0.0,
        min(1.0, (0.42 * huge_component_score) + (0.36 * edge_score) + (0.22 * vertical_score)),
    )

    baseline_metrics: dict[str, float] | None = None
    if baseline_roi is not None and baseline_roi.size:
        baseline_metrics = _dark_roi_metrics(baseline_roi, None)

    dark_score = _dark_object_score(roi_image)
    baseline_score = float(baseline_metrics.get("darkObjectScore", 0.0)) if baseline_metrics else 0.0
    baseline_area = float(baseline_metrics.get("darkAreaFraction", 0.0)) if baseline_metrics else 0.0
    baseline_component = float(baseline_metrics.get("largestDarkComponent", 0.0)) if baseline_metrics else 0.0
    return {
        "darkObjectScore": round(float(dark_score), 4),
        "darkAreaFraction": round(float(dark_area_fraction), 4),
        "largestDarkComponent": round(float(largest_component), 4),
        "largestDarkComponentWidth": round(float(largest_width_fraction), 4),
        "largestDarkComponentHeight": round(float(largest_height_fraction), 4),
        "largestDarkComponentAspect": round(float(largest_aspect), 4),
        "darkComponentEdgeContact": round(float(edge_contact), 4),
        "darkComponentTouchesBorder": round(float(touches_border), 4),
        "darkStructurePenalty": round(float(dark_structure_penalty), 4),
        "darkObjectDelta": round(max(0.0, float(dark_score) - baseline_score), 4),
        "darkAreaDelta": round(max(0.0, float(dark_area_fraction) - baseline_area), 4),
        "darkComponentDelta": round(max(0.0, float(largest_component) - baseline_component), 4),
    }


def ensure_openable_clip(video_path: Path, *, output_dir: Path) -> Path:
    def ffmpeg_path() -> Path | None:
        candidate = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        path = Path(candidate)
        return path if path.exists() else None

    def ffprobe_path() -> Path | None:
        candidate = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
        path = Path(candidate)
        return path if path.exists() else None

    def is_browser_mp4(path: Path) -> bool:
        if path.suffix.lower() != ".mp4":
            return False
        probe_bin = ffprobe_path()
        if probe_bin is None:
            return False
        command = [
            str(probe_bin),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            return False
        output = completed.stdout.lower()
        return "codec_name=h264" in output and "pix_fmt=yuv420p" in output

    def convert_to_browser_mp4(source: Path) -> Path:
        ffmpeg_bin = ffmpeg_path()
        if ffmpeg_bin is None:
            raise FileNotFoundError("ffmpeg no está disponible para convertir el clip.")

        output_dir.mkdir(parents=True, exist_ok=True)
        converted = output_dir / f"{source.stem}_browser.mp4"
        if converted.resolve() == source.resolve():
            converted = output_dir / f"{source.stem}_browser_compatible.mp4"
        command = [
            str(ffmpeg_bin),
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(converted),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0 or not converted.exists():
            raise RuntimeError(completed.stderr.strip() or "No se pudo convertir el clip a MP4.")
        return converted

    capture = cv2.VideoCapture(str(video_path))
    is_openable = capture.isOpened()
    capture.release()

    if is_openable and is_browser_mp4(video_path):
        return video_path

    if is_openable or video_path.suffix.lower() != ".mp4":
        converted = convert_to_browser_mp4(video_path)
        check = cv2.VideoCapture(str(converted))
        ok = check.isOpened()
        check.release()
        if not ok:
            raise RuntimeError("El MP4 convertido existe, pero OpenCV no lo puede abrir.")
        return converted

    ffmpeg_bin = ffmpeg_path()
    if ffmpeg_bin is None:
        if is_openable:
            return video_path
        raise FileNotFoundError("ffmpeg no está disponible para convertir el clip.")

    converted = convert_to_browser_mp4(video_path)
    check = cv2.VideoCapture(str(converted))
    ok = check.isOpened()
    check.release()
    if not ok:
        raise RuntimeError("El MP4 convertido existe, pero OpenCV no lo puede abrir.")
    return converted


def extract_video_segment(
    *,
    source_video: Path,
    output_video: Path,
    start_seconds: float,
    end_seconds: float,
) -> Path:
    def _is_valid_segment(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            if path.stat().st_size < 1024:
                return False
        except Exception:
            return False
        capture = cv2.VideoCapture(str(path))
        is_openable = capture.isOpened()
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_openable else 0
        capture.release()
        return is_openable and total_frames > 0

    output_video.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if Path(ffmpeg_bin).exists():
        command = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-to",
            f"{end_seconds:.3f}",
            "-i",
            str(source_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_video),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0 and _is_valid_segment(output_video):
            return ensure_openable_clip(output_video, output_dir=output_video.parent)
        if output_video.exists():
            try:
                output_video.unlink()
            except Exception:
                pass

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el clip base: {source_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start_frame = max(0, int(round(start_seconds * fps)))
    end_frame = max(start_frame + 1, int(round(end_seconds * fps)))
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    while frame_index < end_frame:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        frame_index += 1
    writer.release()
    capture.release()
    if not _is_valid_segment(output_video):
        raise RuntimeError(f"No se pudo extraer un segmento de video válido desde {source_video}.")
    return ensure_openable_clip(output_video, output_dir=output_video.parent)


def quick_scan_clip(
    *,
    query_path: Path,
    video_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    similarity_threshold: float,
    stage_label: str,
    keep_top: int = 12,
    time_offset_seconds: float = 0.0,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    query_image = cv2.imread(str(query_path))
    if query_image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen de referencia: {query_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el clip: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps else 0.0
    sample_interval_frames = max(1, int(round(sample_every_seconds * fps)))

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = output_dir / "quick_crops"
    frame_dir = output_dir / "quick_frames"
    crop_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    searcher = SimilaritySearcher()
    query_signature = searcher.build_query_signature(query_image)
    previous_threshold = runtime_config.SIMILARITY_THRESHOLD
    runtime_config.SIMILARITY_THRESHOLD = similarity_threshold

    top_hits: list[dict[str, Any]] = []
    earliest_hit: dict[str, Any] | None = None
    sampled_frames = 0
    frame_index = 0

    try:
        while frame_index < max(total_frames, 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                frame_index += sample_interval_frames
                continue

            sampled_frames += 1
            processed_frame, resize_ratio = resize_for_processing(frame)
            candidates = searcher.search(processed_frame, query_signature)
            timestamp_seconds = frame_index / fps if fps else 0.0
            absolute_seconds = time_offset_seconds + timestamp_seconds

            status_label = "Sin hallazgo"
            if candidates:
                best_candidate = candidates[0]
                x1, y1, x2, y2 = best_candidate.bbox
                if resize_ratio != 1.0:
                    x1 = int(round(x1 / resize_ratio))
                    y1 = int(round(y1 / resize_ratio))
                    x2 = int(round(x2 / resize_ratio))
                    y2 = int(round(y2 / resize_ratio))
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(x1 + 1, min(x2, frame.shape[1]))
                y2 = max(y1 + 1, min(y2, frame.shape[0]))
                bbox = (x1, y1, x2, y2)

                hit = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(timestamp_seconds, 2),
                    "absolute_seconds": round(absolute_seconds, 2),
                    "timestamp_label": format_seconds(absolute_seconds),
                    "score": round(float(best_candidate.final_score), 4),
                    "template_score": round(float(best_candidate.template_score), 4),
                    "color_score": round(float(best_candidate.color_score), 4),
                    "feature_score": round(float(best_candidate.feature_score), 4),
                    "structure_score": round(float(best_candidate.structure_score), 4),
                    "bbox": bbox,
                    "zone_id": resolve_zone(frame.shape[:2], bbox),
                    "_frame": frame.copy(),
                }
                if earliest_hit is None or hit["absolute_seconds"] < earliest_hit["absolute_seconds"]:
                    earliest_hit = {key: value for key, value in hit.items() if key != "_frame"}

                top_hits.append(hit)
                top_hits.sort(key=lambda item: float(item["score"]), reverse=True)
                del top_hits[keep_top:]
                status_label = f"Hallazgo {hit['score']:.3f} en {hit['timestamp_label']}"

            progress_ratio = min((frame_index + 1) / max(total_frames, 1), 1.0)
            if on_progress is not None:
                on_progress(
                    stage_label,
                    f"Frame muestreado {sampled_frames} | tiempo {format_seconds(absolute_seconds)} | {status_label}",
                    float(progress_ratio),
                )
            frame_index += sample_interval_frames
    finally:
        capture.release()
        runtime_config.SIMILARITY_THRESHOLD = previous_threshold

    saved_top_hits: list[dict[str, Any]] = []
    for rank, hit in enumerate(top_hits, start=1):
        frame = hit.pop("_frame")
        x1, y1, x2, y2 = hit["bbox"]
        crop = frame[y1:y2, x1:x2]
        timestamp = float(hit["absolute_seconds"])
        base_name = f"rank_{rank:02d}_abs_{timestamp:08.2f}s_frame_{int(hit['frame_index']):06d}_score_{hit['score']:.3f}"

        crop_path = crop_dir / f"{base_name}.jpg"
        frame_path = frame_dir / f"{base_name}.jpg"

        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), runtime_config.BRAND_GOLD, 3)
        cv2.putText(
            annotated,
            f"{hit['score']:.3f} | {hit['timestamp_label']}",
            (x1, max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            runtime_config.BRAND_TEXT_PRIMARY,
            2,
        )
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
        cv2.imwrite(str(frame_path), annotated)

        saved = dict(hit)
        saved["rank"] = rank
        saved["crop_path"] = str(crop_path) if crop_path.exists() else ""
        saved["annotated_frame_path"] = str(frame_path) if frame_path.exists() else ""
        saved_top_hits.append(saved)

    saved_top_hits.sort(key=lambda item: float(item["score"]), reverse=True)

    earliest_enriched: dict[str, Any] | None = None
    if saved_top_hits:
        earliest_enriched = min(saved_top_hits, key=lambda item: float(item.get("absolute_seconds", 0.0)))

    return {
        "ok": True,
        "stage": stage_label,
        "duration_seconds": round(float(duration_seconds), 2),
        "sample_every_seconds": float(sample_every_seconds),
        # Use the enriched hit when possible so consumers (React) can render images.
        "earliest_hit": earliest_enriched or earliest_hit,
        "top_hits": saved_top_hits,
    }


def probe_static_object_clip(
    *,
    query_path: Path,
    video_path: Path,
    output_dir: Path,
    similarity_threshold: float,
    stage_label: str = "static_probe",
    sample_count: int = 5,
    roi: dict[str, Any] | None = None,
    baseline_roi_path: Path | None = None,
    baseline_similarity_score: float | None = None,
    change_threshold: float = 0.14,
    combined_threshold: float = 0.52,
    require_change: bool = True,
) -> dict[str, Any]:
    query_image = cv2.imread(str(query_path))
    if query_image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen de referencia: {query_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el microclip: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = total_frames / fps if fps else 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = output_dir / "probe_crops"
    frame_dir = output_dir / "probe_frames"
    roi_dir = output_dir / "probe_roi"
    crop_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    searcher = SimilaritySearcher()
    query_signature = searcher.build_query_signature(query_image)
    previous_threshold = runtime_config.SIMILARITY_THRESHOLD
    effective_similarity_threshold = max(0.18, float(similarity_threshold) - (0.30 if roi else 0.0))
    runtime_config.SIMILARITY_THRESHOLD = float(effective_similarity_threshold)
    baseline_roi_image = cv2.imread(str(baseline_roi_path)) if baseline_roi_path else None
    baseline_dark_object_score = _dark_object_score(baseline_roi_image) if baseline_roi_image is not None else None

    frame_indices: list[int] = []
    if total_frames > 0:
        count = max(1, int(sample_count))
        if count == 1:
            frame_indices = [max(0, total_frames // 2)]
        else:
            frame_indices = sorted(
                {
                    max(0, min(total_frames - 1, int(round(i * (total_frames - 1) / max(count - 1, 1)))))
                    for i in range(count)
                }
            )

    best_hit: dict[str, Any] | None = None
    best_roi_frame: Any | None = None
    best_roi_box: tuple[int, int, int, int] | None = None
    first_roi_image: Any | None = None
    best_change_score = 0.0
    best_combined_score = 0.0
    best_similarity_score = 0.0
    best_dark_object_score = 0.0
    best_dark_area_fraction = 0.0
    best_largest_dark_component = 0.0
    best_dark_structure_penalty = 0.0
    best_dark_area_delta = 0.0
    best_dark_component_delta = 0.0
    best_roi_object_score = 0.0
    best_roi_metrics: dict[str, float] = {}
    roi_metric_samples: list[dict[str, Any]] = []
    supportive_hits: list[dict[str, Any]] = []
    frames_reviewed = 0
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames_reviewed += 1
            roi_box = _roi_to_pixels(frame.shape[:2], roi)
            search_frame = frame
            origin_x = 0
            origin_y = 0
            roi_image = None
            if roi_box is not None:
                rx1, ry1, rx2, ry2 = roi_box
                roi_image = frame[ry1:ry2, rx1:rx2].copy()
                search_frame = roi_image
                origin_x = rx1
                origin_y = ry1
                if first_roi_image is None:
                    first_roi_image = roi_image.copy()
            processed_frame, resize_ratio = resize_for_processing(search_frame)
            candidates = searcher.search(processed_frame, query_signature)
            candidate = candidates[0] if candidates else None
            similarity_score = float(candidate.final_score) if candidate is not None else 0.0
            if baseline_roi_image is not None and roi_image is not None:
                change_score = _change_score(baseline_roi_image, roi_image)
            elif first_roi_image is not None and roi_image is not None:
                change_score = _change_score(first_roi_image, roi_image)
            else:
                change_score = 0.0
            dark_metrics = _dark_roi_metrics(roi_image, baseline_roi_image)
            dark_object_score = float(dark_metrics.get("darkObjectScore", 0.0))
            dark_area_fraction = float(dark_metrics.get("darkAreaFraction", 0.0))
            largest_dark_component = float(dark_metrics.get("largestDarkComponent", 0.0))
            dark_structure_penalty = float(dark_metrics.get("darkStructurePenalty", 0.0))
            dark_area_delta_sample = float(dark_metrics.get("darkAreaDelta", 0.0))
            dark_component_delta_sample = float(dark_metrics.get("darkComponentDelta", 0.0))
            raw_roi_object_score = (
                (0.34 * dark_object_score)
                + (0.28 * min(1.0, dark_area_fraction / 0.22))
                + (0.30 * min(1.0, largest_dark_component / 0.10))
                + (0.08 * min(1.0, change_score / max(float(change_threshold), 0.001)))
            )
            roi_object_score = raw_roi_object_score * (1.0 - (0.62 * dark_structure_penalty))
            if roi_box is not None:
                roi_metric_samples.append(
                    {
                        "frame_index": int(frame_index),
                        "timestamp_seconds": round(frame_index / fps if fps else 0.0, 2),
                        "changeScore": round(float(change_score), 4),
                        "darkObjectScore": round(float(dark_object_score), 4),
                        "darkAreaFraction": round(float(dark_area_fraction), 4),
                        "largestDarkComponent": round(float(largest_dark_component), 4),
                        "largestDarkComponentWidth": dark_metrics.get("largestDarkComponentWidth", 0.0),
                        "largestDarkComponentHeight": dark_metrics.get("largestDarkComponentHeight", 0.0),
                        "largestDarkComponentAspect": dark_metrics.get("largestDarkComponentAspect", 0.0),
                        "darkComponentEdgeContact": dark_metrics.get("darkComponentEdgeContact", 0.0),
                        "darkComponentTouchesBorder": dark_metrics.get("darkComponentTouchesBorder", 0.0),
                        "darkStructurePenalty": round(float(dark_structure_penalty), 4),
                        "darkAreaDelta": round(float(dark_area_delta_sample), 4),
                        "darkComponentDelta": round(float(dark_component_delta_sample), 4),
                        "rawRoiObjectScore": round(float(raw_roi_object_score), 4),
                        "roiObjectScore": round(float(roi_object_score), 4),
                    }
                )
            if roi_box is not None and baseline_roi_image is not None:
                combined_score = (0.72 * change_score) + (0.28 * similarity_score)
            elif roi_box is not None and not require_change:
                combined_score = similarity_score
            else:
                combined_score = similarity_score
            if roi_box is not None and (best_roi_frame is None or roi_object_score > best_roi_object_score):
                best_change_score = float(change_score)
                best_similarity_score = float(similarity_score)
                best_combined_score = float(combined_score)
                best_dark_object_score = float(dark_object_score)
                best_dark_area_fraction = float(dark_area_fraction)
                best_largest_dark_component = float(largest_dark_component)
                best_dark_structure_penalty = float(dark_structure_penalty)
                best_dark_area_delta = float(dark_area_delta_sample)
                best_dark_component_delta = float(dark_component_delta_sample)
                best_roi_object_score = float(roi_object_score)
                best_roi_metrics = dict(dark_metrics)
                best_roi_frame = frame.copy()
                best_roi_box = roi_box
            elif roi_box is not None and combined_score > best_combined_score:
                best_combined_score = float(combined_score)
            if candidate is None:
                continue
            x1, y1, x2, y2 = candidate.bbox
            if resize_ratio != 1.0:
                x1 = int(round(x1 / resize_ratio))
                y1 = int(round(y1 / resize_ratio))
                x2 = int(round(x2 / resize_ratio))
                y2 = int(round(y2 / resize_ratio))
            x1 += origin_x
            x2 += origin_x
            y1 += origin_y
            y2 += origin_y
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, frame.shape[1]))
            y2 = max(y1 + 1, min(y2, frame.shape[0]))
            timestamp = frame_index / fps if fps else 0.0
            hit = {
                "frame_index": int(frame_index),
                "timestamp_seconds": round(timestamp, 2),
                "timestamp_label": format_seconds(timestamp),
                "score": round(float(candidate.final_score), 4),
                "similarityScore": round(float(similarity_score), 4),
                "changeScore": round(float(change_score), 4),
                "darkObjectScore": round(float(dark_object_score), 4),
                "darkAreaFraction": round(float(dark_area_fraction), 4),
                "largestDarkComponent": round(float(largest_dark_component), 4),
                "darkStructurePenalty": round(float(dark_structure_penalty), 4),
                "darkAreaDelta": round(float(dark_area_delta_sample), 4),
                "darkComponentDelta": round(float(dark_component_delta_sample), 4),
                "rawRoiObjectScore": round(float(raw_roi_object_score), 4),
                "roiObjectScore": round(float(roi_object_score), 4),
                "combinedScore": round(float(combined_score), 4),
                "template_score": round(float(candidate.template_score), 4),
                "color_score": round(float(candidate.color_score), 4),
                "feature_score": round(float(candidate.feature_score), 4),
                "structure_score": round(float(candidate.structure_score), 4),
                "bbox": (x1, y1, x2, y2),
                "zone_id": resolve_zone(frame.shape[:2], (x1, y1, x2, y2)),
                "_frame": frame.copy(),
            }
            if similarity_score >= max(0.16, float(effective_similarity_threshold) - 0.10):
                supportive_hits.append(
                    {
                        "frame_index": int(frame_index),
                        "bbox": (x1, y1, x2, y2),
                        "score": float(candidate.final_score),
                        "similarityScore": float(similarity_score),
                        "combinedScore": float(combined_score),
                    }
                )
            if best_hit is None or float(hit["combinedScore"]) > float(best_hit.get("combinedScore", best_hit["score"])):
                best_hit = hit
    finally:
        capture.release()
        runtime_config.SIMILARITY_THRESHOLD = previous_threshold

    baseline_saved_path = ""
    if first_roi_image is not None:
        baseline_candidate = roi_dir / "roi_reference.jpg"
        cv2.imwrite(str(baseline_candidate), first_roi_image)
        baseline_saved_path = str(baseline_candidate) if baseline_candidate.exists() else ""

    roi_sample_path = ""
    roi_annotated_frame_path = ""
    if best_roi_frame is not None and best_roi_box is not None:
        rx1, ry1, rx2, ry2 = best_roi_box
        roi_crop = best_roi_frame[ry1:ry2, rx1:rx2]
        roi_sample = roi_dir / "roi_best.jpg"
        if roi_crop.size > 0:
            cv2.imwrite(str(roi_sample), roi_crop)
        roi_sample_path = str(roi_sample) if roi_sample.exists() else ""
        roi_annotated = best_roi_frame.copy()
        cv2.rectangle(roi_annotated, (rx1, ry1), (rx2, ry2), runtime_config.BRAND_GOLD, 3)
        annotation = (
            f"ROI dark={best_dark_object_score:.3f} area={best_dark_area_fraction:.3f} "
            f"comp={best_largest_dark_component:.3f} delta={best_dark_area_delta:.3f}"
        )
        text_y = max(28, ry1 - 10)
        cv2.putText(
            roi_annotated,
            annotation,
            (rx1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            runtime_config.BRAND_TEXT_PRIMARY,
            2,
        )
        roi_frame = roi_dir / "roi_best_annotated_frame.jpg"
        cv2.imwrite(str(roi_frame), roi_annotated)
        roi_annotated_frame_path = str(roi_frame) if roi_frame.exists() else ""

    if best_hit is not None:
        frame = best_hit.pop("_frame")
        x1, y1, x2, y2 = best_hit["bbox"]
        crop = frame[y1:y2, x1:x2]
        base_name = f"best_t_{float(best_hit['timestamp_seconds']):08.2f}s_frame_{int(best_hit['frame_index']):06d}_score_{float(best_hit['score']):.3f}"
        crop_path = crop_dir / f"{base_name}.jpg"
        frame_path = frame_dir / f"{base_name}.jpg"
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), runtime_config.BRAND_GOLD, 3)
        cv2.putText(
            annotated,
            f"{best_hit['score']:.3f} | {best_hit['timestamp_label']}",
            (x1, max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            runtime_config.BRAND_TEXT_PRIMARY,
            2,
        )
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
        cv2.imwrite(str(frame_path), annotated)
        best_hit["crop_path"] = str(crop_path) if crop_path.exists() else ""
        best_hit["annotated_frame_path"] = str(frame_path) if frame_path.exists() else ""
        best_similarity_score = max(best_similarity_score, float(best_hit.get("similarityScore", best_hit.get("score", 0.0)) or 0.0))
        best_change_score = max(best_change_score, float(best_hit.get("changeScore", 0.0) or 0.0))
        best_combined_score = max(best_combined_score, float(best_hit.get("combinedScore", best_hit.get("score", 0.0)) or 0.0))

    baseline_similarity = float(baseline_similarity_score or 0.0)
    similarity_delta = max(0.0, best_similarity_score - baseline_similarity) if baseline_similarity_score is not None else 0.0
    baseline_dark_score = float(baseline_dark_object_score or 0.0)
    dark_object_delta = (
        max(0.0, best_dark_object_score - baseline_dark_score) if baseline_dark_object_score is not None else 0.0
    )
    if best_roi_metrics:
        dark_object_delta = max(dark_object_delta, float(best_roi_metrics.get("darkObjectDelta", 0.0) or 0.0))

    persistent_dark_samples = [
        sample
        for sample in roi_metric_samples
        if float(sample.get("darkAreaFraction", 0.0) or 0.0) >= 0.055
        and float(sample.get("largestDarkComponent", 0.0) or 0.0) >= 0.018
    ]
    dark_persistence = (
        float(len(persistent_dark_samples)) / float(len(roi_metric_samples)) if roi_metric_samples else 0.0
    )
    persistent_visual_frames = 0
    persistent_visual_ratio = 0.0
    if best_hit is not None and supportive_hits:
        best_bbox = tuple(best_hit.get("bbox", (0, 0, 0, 0)))
        best_width = max(1, best_bbox[2] - best_bbox[0])
        best_height = max(1, best_bbox[3] - best_bbox[1])
        best_diag = float((best_width**2 + best_height**2) ** 0.5)
        persistent_visual_frames = sum(
            1
            for candidate_hit in supportive_hits
            if _bbox_iou(best_bbox, tuple(candidate_hit.get("bbox", (0, 0, 0, 0)))) >= 0.16
            or _bbox_center_distance(best_bbox, tuple(candidate_hit.get("bbox", (0, 0, 0, 0)))) <= max(28.0, best_diag * 0.18)
        )
        persistent_visual_ratio = (
            float(persistent_visual_frames) / float(max(1, frames_reviewed))
            if frames_reviewed > 0
            else 0.0
        )

    if not roi:
        present = best_hit is not None
        decision = "present_by_similarity" if present else "absent_by_similarity"
        reason = "Sin ROI: fallback compatible al comportamiento anterior basado en similitud visual."
        candidate_confidence = "confirmed" if present else "rejected"
        candidate_reason = reason
    else:
        enough_change = best_change_score >= float(change_threshold)
        enough_combined = best_combined_score >= float(combined_threshold)
        enough_similarity = best_similarity_score >= float(similarity_threshold)
        enough_similarity_gain = similarity_delta >= 0.10
        enough_dark_object = (
            best_dark_object_score >= 0.20
            and best_dark_area_fraction >= 0.055
            and best_largest_dark_component >= 0.018
        )
        large_dark_mass = best_dark_area_fraction >= 0.09 and best_largest_dark_component >= 0.035
        enough_dark_gain = dark_object_delta >= 0.12
        enough_dark_area_gain = best_dark_area_delta >= 0.035 or best_dark_component_delta >= 0.018
        enough_persistence = dark_persistence >= 0.50 or len(persistent_dark_samples) >= 2
        enough_visual_persistence = persistent_visual_frames >= 2 and persistent_visual_ratio >= 0.34
        visual_support_threshold = max(0.18, min(0.35, float(similarity_threshold) - 0.30))
        visual_support = best_similarity_score >= visual_support_threshold
        moderate_visual_match = best_similarity_score >= max(0.24, float(similarity_threshold) - 0.18)
        structural_dark_mass = bool(
            large_dark_mass
            and (
                best_dark_structure_penalty >= 0.45
                or best_largest_dark_component >= 0.18
                or float(best_roi_metrics.get("darkComponentTouchesBorder", 0.0) or 0.0) >= 1.0
                or float(best_roi_metrics.get("largestDarkComponentHeight", 0.0) or 0.0) >= 0.68
            )
        )
        specific_change_support = bool(
            require_change
            and visual_support
            and (
                (
                    structural_dark_mass
                    and enough_similarity
                    and enough_similarity_gain
                    and (enough_change or enough_dark_area_gain)
                )
                or (
                    not structural_dark_mass
                    and (
                        (enough_combined and enough_change)
                        or (enough_similarity and enough_similarity_gain)
                        or (enough_change and enough_dark_gain and enough_dark_area_gain)
                    )
                )
            )
        )
        visual_recovery_support = bool(
            require_change
            and moderate_visual_match
            and enough_visual_persistence
            and (
                enough_similarity_gain
                or best_change_score >= (float(change_threshold) * 0.35)
                or best_combined_score >= (float(combined_threshold) * 0.82)
                or enough_dark_area_gain
            )
        )
        specific_probe_support = bool(
            not require_change
            and visual_support
            and enough_similarity
            and not structural_dark_mass
            and best_dark_structure_penalty < 0.45
        )
        dark_manual_candidate = bool(
            enough_dark_object
            and enough_persistence
            and (
                structural_dark_mass
                or large_dark_mass
                or best_dark_structure_penalty >= 0.35
                or (not visual_support and best_dark_area_fraction >= 0.09)
            )
        )
        visual_manual_candidate = bool(
            moderate_visual_match
            and enough_visual_persistence
            and (
                best_change_score >= (float(change_threshold) * 0.2)
                or best_combined_score >= (float(combined_threshold) * 0.7)
                or similarity_delta >= 0.04
            )
        )
        present = bool(specific_change_support or specific_probe_support or visual_recovery_support)
        if present and structural_dark_mass:
            decision = "present_by_specific_similarity_gain_despite_structural_dark"
            reason = (
                "La masa oscura tiene rasgos estructurales, pero la similitud contra la referencia subió "
                "lo suficiente frente al baseline para confirmarla."
            )
        elif present and visual_recovery_support and not enough_dark_gain:
            decision = "present_by_persistent_visual_support"
            reason = (
                "La ROI mostró coincidencia visual persistente del objeto a lo largo del microclip "
                "y suficiente cambio frente al baseline para confirmarlo aunque la masa oscura no fuera dominante."
            )
        elif present and require_change and enough_change and enough_dark_gain and enough_dark_area_gain:
            decision = "present_by_roi_dark_specific_change"
            reason = (
                "La ROI tuvo cambio temporal específico, aumento de masa oscura frente al baseline "
                "y soporte mínimo del matcher visual."
            )
        elif present and require_change and enough_similarity and enough_similarity_gain and not enough_change:
            decision = "present_by_similarity_gain"
            reason = (
                "La ROI no tuvo un cambio global alto, pero la similitud contra la referencia subió claramente "
                "frente al baseline inicial."
            )
        elif present:
            decision = "present_by_temporal_change"
            reason = "La ROI cambió respecto a la referencia temporal y la similitud visual apoyó la decisión."
        elif visual_manual_candidate:
            decision = "candidate_persistent_visual_manual_review"
            reason = (
                "La ROI mostró coincidencia visual persistente del objeto, pero todavía no alcanzó señal temporal "
                "suficiente para confirmarlo automáticamente."
            )
        elif dark_manual_candidate:
            decision = "candidate_dark_structural_mass_manual_review"
            reason = (
                "La ROI contiene una masa oscura persistente, pero es grande/estructural o pegada a bordes "
                "y no tiene suficiente cambio/similitud específica para confirmarla como objeto abandonado."
            )
        elif not visual_support:
            decision = "rejected_without_visual_support"
            reason = "No hubo suficiente masa oscura persistente en la ROI ni soporte visual mínimo del matcher."
        elif enough_similarity and not enough_similarity_gain and require_change:
            decision = "rejected_similarity_without_baseline_gain"
            reason = "La similitud fue alta, pero no mejoró lo suficiente contra el baseline inicial."
        elif best_similarity_score >= float(similarity_threshold) and not enough_change:
            decision = "rejected_similarity_without_change"
            reason = "La similitud visual no fue suficiente porque la ROI no cambió temporalmente; evita falsos positivos de fondo."
        elif not enough_combined:
            decision = "rejected_low_combined_score"
            reason = "El puntaje combinado de cambio temporal y similitud quedó bajo el umbral."
        else:
            decision = "rejected_low_change_score"
            reason = "El cambio temporal dentro de la ROI quedó bajo el umbral."
        if present:
            candidate_confidence = "confirmed"
        elif visual_manual_candidate or dark_manual_candidate:
            candidate_confidence = "candidate"
        else:
            candidate_confidence = "rejected"
        candidate_reason = reason

    return {
        "ok": True,
        "stage": stage_label,
        "present": present,
        "candidate": candidate_confidence == "candidate",
        "candidateConfidence": candidate_confidence,
        "candidateReason": candidate_reason,
        "decision": decision,
        "reason": reason,
        "roi": roi,
        "roi_reference_path": baseline_saved_path,
        "roi_sample_path": roi_sample_path,
        "roi_annotated_frame_path": roi_annotated_frame_path,
        "changeScore": round(float(best_change_score), 4),
        "similarityScore": round(float(best_similarity_score), 4),
        "baselineSimilarityScore": round(float(baseline_similarity), 4) if baseline_similarity_score is not None else None,
        "similarityDelta": round(float(similarity_delta), 4),
        "darkObjectScore": round(float(best_dark_object_score), 4),
        "baselineDarkObjectScore": round(float(baseline_dark_score), 4) if baseline_dark_object_score is not None else None,
        "darkObjectDelta": round(float(dark_object_delta), 4),
        "darkAreaFraction": round(float(best_dark_area_fraction), 4),
        "largestDarkComponent": round(float(best_largest_dark_component), 4),
        "darkStructurePenalty": round(float(best_dark_structure_penalty), 4),
        "largestDarkComponentWidth": round(float(best_roi_metrics.get("largestDarkComponentWidth", 0.0) or 0.0), 4),
        "largestDarkComponentHeight": round(float(best_roi_metrics.get("largestDarkComponentHeight", 0.0) or 0.0), 4),
        "largestDarkComponentAspect": round(float(best_roi_metrics.get("largestDarkComponentAspect", 0.0) or 0.0), 4),
        "darkComponentEdgeContact": round(float(best_roi_metrics.get("darkComponentEdgeContact", 0.0) or 0.0), 4),
        "darkComponentTouchesBorder": round(float(best_roi_metrics.get("darkComponentTouchesBorder", 0.0) or 0.0), 4),
        "darkAreaDelta": round(float(best_dark_area_delta), 4),
        "darkComponentDelta": round(float(best_dark_component_delta), 4),
        "darkPersistence": round(float(dark_persistence), 4),
        "persistentDarkFrames": len(persistent_dark_samples),
        "persistentVisualFrames": int(persistent_visual_frames),
        "persistentVisualRatio": round(float(persistent_visual_ratio), 4),
        "roiObjectScore": round(float(best_roi_object_score), 4),
        "roiMetricSamples": roi_metric_samples,
        "combinedScore": round(float(best_combined_score), 4),
        "changeThreshold": float(change_threshold),
        "combinedThreshold": float(combined_threshold),
        "visualSupportThreshold": max(0.18, min(0.35, float(similarity_threshold) - 0.30)) if roi else None,
        "duration_seconds": round(float(duration_seconds), 2),
        "frames_reviewed": frames_reviewed,
        "similarity_threshold": float(similarity_threshold),
        "effective_similarity_threshold": float(effective_similarity_threshold),
        "best_hit": best_hit,
    }


def run_deep_analysis(
    *,
    query_path: Path,
    video_path: Path,
    output_dir: Path,
    similarity_threshold: float,
    frame_step: int,
    person_trigger_mode: str,
    person_detection_frame_step: int,
    preview_callback_sample_interval: int,
    max_results: int,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    def on_status(update: dict[str, Any]) -> None:
        if on_progress is None:
            return
        ratio = float(update.get("progress_ratio", 0.0))
        frame_idx = int(update.get("frame_index", 0)) + 1
        total = int(update.get("total_frames", 0)) or 1
        on_progress(
            "deep",
            f"Deep frame {frame_idx}/{total} | {update.get('timestamp_label','-')} | hallazgos {update.get('matches_found',0)}",
            ratio,
        )

    runtime_config.reset_runtime_overrides()
    runtime_config.apply_runtime_overrides(
        query_image_path=query_path,
        video_path=video_path,
        output_dir=output_dir,
        show_preview=False,
        save_annotated_video=True,
        frame_step=frame_step,
        similarity_threshold=similarity_threshold,
        max_results=max_results,
        enable_person_detection=True,
        person_detection_trigger_mode=person_trigger_mode,
        person_detection_frame_step=person_detection_frame_step,
        preview_callback_sample_interval=preview_callback_sample_interval,
    )
    return VideoProcessor(status_callback=on_status, preview_callback=None).run()


def file_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")
