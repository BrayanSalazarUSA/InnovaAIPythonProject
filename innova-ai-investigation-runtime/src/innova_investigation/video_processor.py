from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

import cv2
import numpy as np

from . import config
from .similarity_search import MatchCandidate, SimilaritySearcher

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency at runtime
    YOLO = None


PERSON_DETECTOR_CACHE: object | None = None
PERSON_DETECTOR_CACHE_PATH = ""


@dataclass(slots=True)
class InvestigationMatch:
    rank: int
    frame_index: int
    timestamp_seconds: float
    timestamp_label: str
    score: float
    base_score: float
    template_score: float
    color_score: float
    feature_score: float
    structure_score: float
    scale: float
    bbox: tuple[int, int, int, int]
    zone_id: str = ""
    person_count: int = 0
    nearby_person_count: int = 0
    associated_people: list["PersonSnapshot"] = field(default_factory=list, repr=False)
    crop_path: str = ""
    clip_path: str = ""
    annotated_frame_path: str = ""
    _frame: np.ndarray = field(
        repr=False,
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8),
    )


@dataclass(slots=True)
class PersonSnapshot:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    zone_id: str
    is_near_object: bool
    crop_path: str = ""


@dataclass(slots=True)
class PersonTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    zone_id: str
    last_seen_frame: int


class VideoProcessor:
    def __init__(
        self,
        status_callback: Callable[[dict[str, object]], None] | None = None,
        preview_callback: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.searcher = SimilaritySearcher()
        self.preview_enabled = config.SHOW_PREVIEW
        self.preview_writer: cv2.VideoWriter | None = None
        self.query_thumbnail: np.ndarray | None = None
        self.status_callback = status_callback
        self.preview_callback = preview_callback
        self.person_tracks: dict[int, PersonTrack] = {}
        self.next_person_track_id = 1
        self.total_person_tracks_observed = 0
        self.max_people_visible_seen = 0
        self.object_triggered_person_detection = False
        self.person_detector_error = ""
        self.person_detector = self._load_person_detector()

    def _load_person_detector(self):
        global PERSON_DETECTOR_CACHE, PERSON_DETECTOR_CACHE_PATH

        if not config.ENABLE_PERSON_DETECTION:
            self.person_detector_error = "Deteccion de personas deshabilitada."
            return None

        if YOLO is None:
            self.person_detector_error = "Ultralytics no esta instalado en este entorno."
            print(self.person_detector_error, flush=True)
            return None

        if not config.YOLO_MODEL_PATH.exists():
            self.person_detector_error = f"No se encontro el modelo YOLO: {config.YOLO_MODEL_PATH}"
            print(self.person_detector_error, flush=True)
            return None

        model_path = str(config.YOLO_MODEL_PATH)
        if PERSON_DETECTOR_CACHE is not None and PERSON_DETECTOR_CACHE_PATH == model_path:
            self.person_detector_error = ""
            print(
                f"YOLO reutilizado desde cache para personas ({config.PERSON_DETECTION_TRIGGER_MODE}).",
                flush=True,
            )
            return PERSON_DETECTOR_CACHE

        try:
            detector = YOLO(model_path)
        except Exception as error:  # pragma: no cover - depends on local runtime
            self.person_detector_error = f"No se pudo cargar YOLO: {error}"
            print(self.person_detector_error, flush=True)
            return None

        PERSON_DETECTOR_CACHE = detector
        PERSON_DETECTOR_CACHE_PATH = model_path
        self.person_detector_error = ""
        print(
            f"YOLO cargado para personas desde {config.YOLO_MODEL_PATH.name} "
            f"({config.PERSON_DETECTION_TRIGGER_MODE}).",
            flush=True,
        )
        return detector

    def run(self) -> dict[str, object]:
        self._ensure_environment()

        query_image = cv2.imread(str(config.QUERY_IMAGE_PATH))
        if query_image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen de consulta: {config.QUERY_IMAGE_PATH}")

        query_signature = self.searcher.build_query_signature(query_image)
        self.query_thumbnail = self._build_query_thumbnail(query_signature.image)

        video_capture = cv2.VideoCapture(str(config.VIDEO_PATH))
        if not video_capture.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {config.VIDEO_PATH}")

        fps = video_capture.get(cv2.CAP_PROP_FPS) or 1.0
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / fps if fps else 0.0
        self.person_tracks = {}
        self.next_person_track_id = 1
        self.total_person_tracks_observed = 0
        self.max_people_visible_seen = 0
        self.object_triggered_person_detection = False

        print(f"Video: {config.VIDEO_PATH}", flush=True)
        print(f"Query: {config.QUERY_IMAGE_PATH}", flush=True)
        print(
            f"Frames totales: {total_frames} | FPS: {fps:.2f} | Duracion: {total_seconds:.1f}s",
            flush=True,
        )

        best_matches: list[InvestigationMatch] = []
        current_visual_matches: list[InvestigationMatch] = []
        current_tracked_people: list[PersonTrack] = []
        score_trace: list[tuple[float, float]] = []
        zone_counter: Counter[str] = Counter()
        processed_frames = 0
        frame_index = -1
        interrupted_by_user = False
        stopped_after_confident_person_match = False
        early_stop_after_frame: int | None = None
        last_detection_summary = "Buscando coincidencias visuales..."
        last_detection_timestamp = "-"
        last_detection_score = 0.0
        person_detection_summary = self._build_person_detection_summary(0, False)

        if self.preview_enabled:
            self._initialize_preview_window()

        while True:
            has_frame, frame = video_capture.read()
            if not has_frame:
                break

            frame_index += 1
            timestamp_seconds = frame_index / fps
            is_sampled_frame = frame_index % config.FRAME_STEP == 0
            new_visual_matches: list[InvestigationMatch] = []

            if is_sampled_frame:
                processed_frames += 1
                processed_frame, resize_ratio = self._resize_for_processing(frame)
                candidates = self.searcher.search(processed_frame, query_signature)

                for candidate in candidates:
                    match = self._to_investigation_match(
                        candidate=candidate,
                        original_frame=frame,
                        resize_ratio=resize_ratio,
                        frame_index=frame_index,
                        timestamp_seconds=timestamp_seconds,
                    )
                    new_visual_matches.append(match)
                    zone_counter[match.zone_id] += 1

            if new_visual_matches:
                self.object_triggered_person_detection = True

            person_detection_active = self._is_person_detection_active()
            if person_detection_active:
                current_tracked_people = self._update_person_tracking(
                    frame=frame,
                    frame_index=frame_index,
                )
            else:
                self.person_tracks.clear()
                current_tracked_people = []

            self.max_people_visible_seen = max(self.max_people_visible_seen, len(current_tracked_people))
            person_detection_summary = self._build_person_detection_summary(
                visible_people=len(current_tracked_people),
                detection_active=person_detection_active,
            )

            if new_visual_matches and current_tracked_people:
                for match in new_visual_matches:
                    associated_people, person_count, nearby_person_count = self._associate_people_to_match(
                        tracked_people=current_tracked_people,
                        frame_shape=frame.shape[:2],
                        object_bbox=match.bbox,
                    )
                    match.associated_people = associated_people
                    match.person_count = person_count
                    match.nearby_person_count = nearby_person_count
                    match.score = self._compute_contextual_match_score(match)

            elif new_visual_matches:
                for match in new_visual_matches:
                    match.score = self._compute_contextual_match_score(match)

            if is_sampled_frame:
                for match in new_visual_matches:
                    self._merge_match(best_matches, match)

                current_visual_matches = new_visual_matches

                if config.EARLY_STOP_ON_PERSON_MATCH:
                    confident_threshold = max(
                        float(config.EARLY_STOP_MIN_MATCH_SCORE),
                        float(config.SIMILARITY_THRESHOLD),
                    )
                    has_confident_person_match = any(
                        match.nearby_person_count > 0 and match.score >= confident_threshold
                        for match in new_visual_matches
                    )
                    if has_confident_person_match and timestamp_seconds >= config.EARLY_STOP_MIN_SECONDS:
                        patience_frames = max(1, int(round(config.EARLY_STOP_PATIENCE_SECONDS * max(fps, 1.0))))
                        candidate_stop_frame = frame_index + patience_frames
                        if early_stop_after_frame is None:
                            early_stop_after_frame = candidate_stop_frame
                        else:
                            early_stop_after_frame = min(early_stop_after_frame, candidate_stop_frame)

                if new_visual_matches:
                    strongest_match = max(new_visual_matches, key=lambda match: match.score)
                    last_detection_summary = f"Objeto similar detectado | score {strongest_match.score:.3f}"
                    last_detection_timestamp = strongest_match.timestamp_label
                    last_detection_score = strongest_match.score
                elif best_matches:
                    best_match = best_matches[0]
                    last_detection_summary = f"Sin match nuevo | mejor evidencia en {best_match.timestamp_label}"
                    last_detection_timestamp = best_match.timestamp_label
                    last_detection_score = best_match.score
                else:
                    last_detection_summary = "Buscando coincidencias visuales..."
                    last_detection_timestamp = "-"
                    last_detection_score = 0.0

                score_trace.append((timestamp_seconds, last_detection_score))
                if len(score_trace) > config.TRACE_MAX_POINTS:
                    score_trace = score_trace[-config.TRACE_MAX_POINTS :]

                if processed_frames % 50 == 0:
                    print(
                        f"Procesados {processed_frames} frames muestreados "
                        f"({timestamp_seconds:.1f}s) | Hallazgos actuales: {len(best_matches)}",
                        flush=True,
                    )

            preview_frame: np.ndarray | None = None
            if self.preview_enabled or config.SAVE_ANNOTATED_VIDEO or self.preview_callback is not None:
                preview_frame = self._render_preview(
                    frame=frame,
                    current_matches=current_visual_matches,
                    tracked_people=current_tracked_people,
                    best_matches=best_matches,
                    frame_index=frame_index,
                    processed_frames=processed_frames,
                    total_frames=total_frames,
                    fps=fps,
                    timestamp_seconds=timestamp_seconds,
                    is_sampled_frame=is_sampled_frame,
                    last_detection_summary=last_detection_summary,
                    last_detection_timestamp=last_detection_timestamp,
                    last_detection_score=last_detection_score,
                    person_detection_summary=person_detection_summary,
                    score_trace=score_trace,
                    zone_counter=zone_counter,
                )
                self._write_preview_frame(preview_frame, fps)

            self._emit_runtime_update(
                preview_frame=preview_frame,
                frame_index=frame_index,
                total_frames=total_frames,
                timestamp_seconds=timestamp_seconds,
                processed_frames=processed_frames,
                best_matches=best_matches,
                current_matches=current_visual_matches,
                tracked_people=current_tracked_people,
                last_detection_summary=last_detection_summary,
                last_detection_timestamp=last_detection_timestamp,
                last_detection_score=last_detection_score,
                person_detection_summary=person_detection_summary,
                is_sampled_frame=is_sampled_frame,
            )

            if self.preview_enabled and preview_frame is not None and not self._show_preview(preview_frame):
                interrupted_by_user = True
                print("Analisis detenido por el usuario.", flush=True)
                break

            if early_stop_after_frame is not None and frame_index >= early_stop_after_frame:
                stopped_after_confident_person_match = True
                print(
                    "Analisis detenido temprano: objeto y persona cercana confirmados.",
                    flush=True,
                )
                break

        video_capture.release()
        self._close_preview()

        best_matches.sort(key=lambda item: item.score, reverse=True)
        for rank, match in enumerate(best_matches, start=1):
            match.rank = rank

        self._save_evidence(best_matches)
        self._export_evidence_clips(best_matches, fps)
        report = self._build_report(
            matches=best_matches,
            total_frames=total_frames,
            processed_frames=processed_frames,
            fps=fps,
            total_seconds=total_seconds,
            interrupted_by_user=interrupted_by_user,
            stopped_after_confident_person_match=stopped_after_confident_person_match,
            zone_counter=zone_counter,
        )
        self._write_reports(report)
        return report

    def _ensure_environment(self) -> None:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.CROPS_DIR.mkdir(parents=True, exist_ok=True)
        config.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        config.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        config.PERSONS_DIR.mkdir(parents=True, exist_ok=True)
        self._clear_previous_outputs()

    def _clear_previous_outputs(self) -> None:
        for pattern in ("*.jpg", "*.json", "*.csv", "*.txt", "*.mp4"):
            for file_path in config.OUTPUT_DIR.glob(pattern):
                file_path.unlink(missing_ok=True)

        for directory in (config.CROPS_DIR, config.ANNOTATIONS_DIR, config.CLIPS_DIR, config.PERSONS_DIR):
            for file_path in directory.glob("*.jpg"):
                file_path.unlink(missing_ok=True)
            for file_path in directory.glob("*.mp4"):
                file_path.unlink(missing_ok=True)

    def _resize_for_processing(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        if width <= config.PROCESSING_MAX_WIDTH:
            return frame, 1.0

        ratio = config.PROCESSING_MAX_WIDTH / width
        new_size = (int(width * ratio), int(height * ratio))
        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return resized, ratio

    def _resize_for_preview(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= config.PREVIEW_MAX_WIDTH:
            return frame

        ratio = config.PREVIEW_MAX_WIDTH / width
        resized = cv2.resize(
            frame,
            (int(width * ratio), int(height * ratio)),
            interpolation=cv2.INTER_AREA,
        )
        return resized

    def _build_query_thumbnail(self, query_image: np.ndarray) -> np.ndarray:
        thumbnail_width = 170
        scale = thumbnail_width / max(query_image.shape[1], 1)
        thumbnail_height = max(1, int(query_image.shape[0] * scale))
        return cv2.resize(
            query_image,
            (thumbnail_width, thumbnail_height),
            interpolation=cv2.INTER_AREA,
        )

    def _is_person_detection_active(self) -> bool:
        if not config.ENABLE_PERSON_DETECTION or self.person_detector is None:
            return False

        if config.PERSON_DETECTION_TRIGGER_MODE == "always":
            return True

        return self.object_triggered_person_detection

    def _build_person_detection_summary(self, visible_people: int, detection_active: bool) -> str:
        if not config.ENABLE_PERSON_DETECTION:
            return "Personas YOLO deshabilitado"
        if self.person_detector is None:
            if self.person_detector_error:
                return f"YOLO no disponible: {self.person_detector_error}"
            return "YOLO no disponible"
        if detection_active:
            return f"Tracking personas activo | visibles {visible_people}"
        return "Tracking personas en espera del primer objeto"

    def _update_person_tracking(self, frame: np.ndarray, frame_index: int) -> list[PersonTrack]:
        should_infer_now = frame_index % config.PERSON_DETECTION_FRAME_STEP == 0
        if should_infer_now:
            detections = self._detect_people(frame)
            self._merge_person_tracks(detections, frame_index, frame.shape[:2])
        else:
            self._prune_person_tracks(frame_index)
        return self._get_active_person_tracks(frame_index)

    def _detect_people(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        if self.person_detector is None:
            return []

        try:
            results = self.person_detector.predict(
                source=frame,
                classes=[0],
                conf=config.PERSON_CONFIDENCE_THRESHOLD,
                device=config.PERSON_DETECTION_DEVICE,
                verbose=False,
            )
        except Exception as error:  # pragma: no cover - runtime specific
            self.person_detector_error = f"Fallo en inferencia YOLO: {error}"
            return []

        detections: list[tuple[tuple[int, int, int, int], float]] = []
        if not results:
            return detections

        frame_height, frame_width = frame.shape[:2]
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = (
                max(0, min(int(x1), frame_width - 1)),
                max(0, min(int(y1), frame_height - 1)),
                max(1, min(int(x2), frame_width)),
                max(1, min(int(y2), frame_height)),
            )
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            detections.append((bbox, confidence))
        return detections

    def _merge_person_tracks(
        self,
        detections: list[tuple[tuple[int, int, int, int], float]],
        frame_index: int,
        frame_shape: tuple[int, int],
    ) -> None:
        active_tracks = {
            track_id: track
            for track_id, track in self.person_tracks.items()
            if frame_index - track.last_seen_frame <= config.PERSON_TRACK_MAX_AGE
        }
        updated_tracks: dict[int, PersonTrack] = {}
        unmatched_track_ids = set(active_tracks.keys())

        for bbox, confidence in sorted(detections, key=lambda item: item[1], reverse=True):
            matched_track_id: int | None = None
            best_score = float("inf")

            for track_id in list(unmatched_track_ids):
                track = active_tracks[track_id]
                distance = self._centroid_distance(track.bbox, bbox)
                iou = self._iou(track.bbox, bbox)
                if distance > config.PERSON_TRACK_MAX_DISTANCE and iou < config.PERSON_TRACK_MIN_IOU:
                    continue

                match_score = distance - (iou * 160.0)
                if match_score < best_score:
                    best_score = match_score
                    matched_track_id = track_id

            if matched_track_id is None:
                track_id = self.next_person_track_id
                self.next_person_track_id += 1
                self.total_person_tracks_observed = max(self.total_person_tracks_observed, track_id)
            else:
                track_id = matched_track_id
                unmatched_track_ids.remove(track_id)

            updated_tracks[track_id] = PersonTrack(
                track_id=track_id,
                bbox=bbox,
                confidence=confidence,
                zone_id=self._resolve_zone_id(frame_shape, bbox),
                last_seen_frame=frame_index,
            )

        for track_id in unmatched_track_ids:
            updated_tracks[track_id] = active_tracks[track_id]

        self.person_tracks = updated_tracks

    def _prune_person_tracks(self, frame_index: int) -> None:
        self.person_tracks = {
            track_id: track
            for track_id, track in self.person_tracks.items()
            if frame_index - track.last_seen_frame <= config.PERSON_TRACK_MAX_AGE
        }

    def _get_active_person_tracks(self, frame_index: int) -> list[PersonTrack]:
        self._prune_person_tracks(frame_index)
        return sorted(
            self.person_tracks.values(),
            key=lambda track: (frame_index - track.last_seen_frame, track.track_id),
        )

    def _associate_people_to_match(
        self,
        tracked_people: list[PersonTrack],
        frame_shape: tuple[int, int],
        object_bbox: tuple[int, int, int, int],
    ) -> tuple[list[PersonSnapshot], int, int]:
        if not tracked_people:
            return [], 0, 0

        associated_people: list[PersonSnapshot] = []
        nearby_person_count = 0

        for track in tracked_people:
            is_near_object = self._is_person_near_object(track.bbox, object_bbox)
            nearby_person_count += int(is_near_object)
            associated_people.append(
                PersonSnapshot(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=round(track.confidence, 4),
                    zone_id=self._resolve_zone_id(frame_shape, track.bbox),
                    is_near_object=is_near_object,
                )
            )

        associated_people.sort(
            key=lambda person: (
                not person.is_near_object,
                self._centroid_distance(person.bbox, object_bbox),
                person.track_id,
            )
        )
        return (
            associated_people[: config.MAX_ASSOCIATED_PEOPLE_PER_MATCH],
            len(tracked_people),
            nearby_person_count,
        )

    def _is_person_near_object(
        self,
        person_bbox: tuple[int, int, int, int],
        object_bbox: tuple[int, int, int, int],
    ) -> bool:
        if self._iou(person_bbox, object_bbox) > 0:
            return True
        return self._centroid_distance(person_bbox, object_bbox) <= config.PERSON_ASSOCIATION_DISTANCE

    def _centroid_distance(
        self,
        first_box: tuple[int, int, int, int],
        second_box: tuple[int, int, int, int],
    ) -> float:
        first_center_x = (first_box[0] + first_box[2]) / 2
        first_center_y = (first_box[1] + first_box[3]) / 2
        second_center_x = (second_box[0] + second_box[2]) / 2
        second_center_y = (second_box[1] + second_box[3]) / 2
        return float(np.hypot(first_center_x - second_center_x, first_center_y - second_center_y))

    def _to_investigation_match(
        self,
        candidate: MatchCandidate,
        original_frame: np.ndarray,
        resize_ratio: float,
        frame_index: int,
        timestamp_seconds: float,
    ) -> InvestigationMatch:
        x1, y1, x2, y2 = candidate.bbox

        if resize_ratio != 1.0:
            x1 = int(round(x1 / resize_ratio))
            y1 = int(round(y1 / resize_ratio))
            x2 = int(round(x2 / resize_ratio))
            y2 = int(round(y2 / resize_ratio))

        x1 = max(0, min(x1, original_frame.shape[1] - 1))
        y1 = max(0, min(y1, original_frame.shape[0] - 1))
        x2 = max(x1 + 1, min(x2, original_frame.shape[1]))
        y2 = max(y1 + 1, min(y2, original_frame.shape[0]))

        return InvestigationMatch(
            rank=0,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            timestamp_label=self._format_seconds(timestamp_seconds),
            score=round(candidate.final_score, 4),
            base_score=round(candidate.final_score, 4),
            template_score=round(candidate.template_score, 4),
            color_score=round(candidate.color_score, 4),
            feature_score=round(candidate.feature_score, 4),
            structure_score=round(candidate.structure_score, 4),
            scale=round(candidate.scale, 2),
            bbox=(x1, y1, x2, y2),
            zone_id=self._resolve_zone_id(original_frame.shape[:2], (x1, y1, x2, y2)),
            _frame=original_frame.copy(),
        )

    def _compute_contextual_match_score(self, match: InvestigationMatch) -> float:
        score = match.base_score
        if match.nearby_person_count > 0:
            score += config.PERSON_NEAR_MATCH_BONUS
        elif match.person_count > 0:
            score += config.PERSON_VISIBLE_MATCH_BONUS

        score -= match.timestamp_seconds * config.EARLY_MATCH_TIME_PENALTY_PER_SECOND
        return round(max(score, 0.0), 4)

    def _merge_match(self, best_matches: list[InvestigationMatch], new_match: InvestigationMatch) -> None:
        for index, existing_match in enumerate(best_matches):
            time_distance = abs(existing_match.timestamp_seconds - new_match.timestamp_seconds)
            iou = self._iou(existing_match.bbox, new_match.bbox)

            if (
                time_distance <= config.TEMPORAL_DUPLICATE_WINDOW_SECONDS
                and iou >= config.IOU_DUPLICATE_THRESHOLD
            ):
                if new_match.score > existing_match.score:
                    best_matches[index] = new_match
                return

        best_matches.append(new_match)
        best_matches.sort(key=lambda item: item.score, reverse=True)
        del best_matches[config.MAX_RESULTS :]

    def _save_evidence(self, matches: list[InvestigationMatch]) -> None:
        for match in matches:
            x1, y1, x2, y2 = match.bbox
            crop = match._frame[y1:y2, x1:x2]

            base_name = (
                f"rank_{match.rank:02d}"
                f"_t_{match.timestamp_seconds:07.2f}s"
                f"_f_{match.frame_index:05d}"
                f"_score_{match.score:.3f}"
            )

            crop_path = config.CROPS_DIR / f"{base_name}.jpg"
            annotated_path = config.ANNOTATIONS_DIR / f"{base_name}.jpg"

            annotated_frame = match._frame.copy()
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), config.BRAND_GOLD, 3)
            cv2.putText(
                annotated_frame,
                f"score={match.score:.3f} | {match.timestamp_label}",
                (x1, max(30, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                config.BRAND_TEXT_ACCENT,
                2,
            )

            for person in match.associated_people:
                px1, py1, px2, py2 = person.bbox
                person_color = (
                    config.PERSON_BOX_ASSOCIATED_COLOR
                    if person.is_near_object
                    else config.PERSON_BOX_COLOR
                )
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), person_color, 2)
                cv2.putText(
                    annotated_frame,
                    f"Persona #{person.track_id:02d}",
                    (px1, max(28, py1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.66,
                    person_color,
                    2,
                )

                person_crop = match._frame[py1:py2, px1:px2]
                if person_crop.size == 0:
                    continue
                role = "near" if person.is_near_object else "scene"
                person_path = config.PERSONS_DIR / f"{base_name}_person_{person.track_id:02d}_{role}.jpg"
                cv2.imwrite(str(person_path), person_crop)
                person.crop_path = str(person_path)

            if crop.size > 0:
                cv2.imwrite(str(crop_path), crop)
            cv2.imwrite(str(annotated_path), annotated_frame)

            match.crop_path = str(crop_path)
            match.annotated_frame_path = str(annotated_path)

    def _export_evidence_clips(self, matches: list[InvestigationMatch], fps: float) -> None:
        for match in matches:
            capture = cv2.VideoCapture(str(config.VIDEO_PATH))
            if not capture.isOpened():
                continue

            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_fps = capture.get(cv2.CAP_PROP_FPS) or fps or 20.0

            start_seconds = max(0.0, match.timestamp_seconds - config.EVIDENCE_CLIP_SECONDS_BEFORE)
            end_seconds = match.timestamp_seconds + config.EVIDENCE_CLIP_SECONDS_AFTER
            start_frame = max(0, int(start_seconds * output_fps))
            end_frame = min(total_frames - 1, int(end_seconds * output_fps))

            base_name = (
                f"rank_{match.rank:02d}"
                f"_t_{match.timestamp_seconds:07.2f}s"
                f"_f_{match.frame_index:05d}"
                f"_score_{match.score:.3f}"
            )
            clip_path = config.CLIPS_DIR / f"{base_name}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(clip_path),
                fourcc,
                output_fps,
                (frame_width, frame_height),
            )
            if not writer.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame_index = start_frame
            x1, y1, x2, y2 = match.bbox

            while current_frame_index <= end_frame:
                has_frame, frame = capture.read()
                if not has_frame:
                    break

                annotated = frame.copy()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (46, 255, 102), 3)
                cv2.putText(
                    annotated,
                    f"Match {match.score:.3f} | {match.zone_id} | {match.timestamp_label}",
                    (x1, max(36, y1 - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.80,
                    (180, 255, 198),
                    2,
                )
                for person in match.associated_people:
                    px1, py1, px2, py2 = person.bbox
                    person_color = (
                        config.PERSON_BOX_ASSOCIATED_COLOR
                        if person.is_near_object
                        else config.PERSON_BOX_COLOR
                    )
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), person_color, 2)
                    cv2.putText(
                        annotated,
                        f"Persona #{person.track_id:02d}",
                        (px1, max(28, py1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.66,
                        person_color,
                        2,
                    )
                clip_time = current_frame_index / max(output_fps, 1.0)
                cv2.putText(
                    annotated,
                    f"Clip {self._format_seconds(clip_time)}",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                writer.write(annotated)
                current_frame_index += 1

            writer.release()
            capture.release()
            match.clip_path = str(clip_path)

    def _build_report(
        self,
        matches: list[InvestigationMatch],
        total_frames: int,
        processed_frames: int,
        fps: float,
        total_seconds: float,
        interrupted_by_user: bool,
        stopped_after_confident_person_match: bool,
        zone_counter: Counter[str],
    ) -> dict[str, object]:
        return {
            "query_image_path": str(config.QUERY_IMAGE_PATH),
            "video_path": str(config.VIDEO_PATH),
            "output_dir": str(config.OUTPUT_DIR),
            "annotated_video_path": str(config.ANNOTATED_VIDEO_PATH) if config.SAVE_ANNOTATED_VIDEO else "",
            "clips_dir": str(config.CLIPS_DIR),
            "persons_dir": str(config.PERSONS_DIR),
            "settings": {
                "frame_step": config.FRAME_STEP,
                "similarity_threshold": config.SIMILARITY_THRESHOLD,
                "max_results": config.MAX_RESULTS,
                "processing_max_width": config.PROCESSING_MAX_WIDTH,
                "preview_max_width": config.PREVIEW_MAX_WIDTH,
                "save_annotated_video": config.SAVE_ANNOTATED_VIDEO,
                "zone_grid_rows": config.ZONE_GRID_ROWS,
                "zone_grid_cols": config.ZONE_GRID_COLS,
                "template_scales": list(config.TEMPLATE_SCALES),
                "person_detection_enabled": config.ENABLE_PERSON_DETECTION,
                "person_detection_trigger_mode": config.PERSON_DETECTION_TRIGGER_MODE,
                "person_detection_frame_step": config.PERSON_DETECTION_FRAME_STEP,
                "person_confidence_threshold": config.PERSON_CONFIDENCE_THRESHOLD,
                "early_stop_on_person_match": config.EARLY_STOP_ON_PERSON_MATCH,
                "early_stop_min_match_score": config.EARLY_STOP_MIN_MATCH_SCORE,
                "early_stop_patience_seconds": config.EARLY_STOP_PATIENCE_SECONDS,
                "yolo_model_path": str(config.YOLO_MODEL_PATH),
            },
            "person_tracking_summary": {
                "enabled": config.ENABLE_PERSON_DETECTION,
                "available": self.person_detector is not None,
                "status": self.person_detector_error or "activo",
                "trigger_mode": config.PERSON_DETECTION_TRIGGER_MODE,
                "tracks_observed": self.total_person_tracks_observed,
                "max_people_visible": self.max_people_visible_seen,
            },
            "video_summary": {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "fps": round(fps, 3),
                "duration_seconds": round(total_seconds, 3),
                "interrupted_by_user": interrupted_by_user,
                "early_stop_triggered": stopped_after_confident_person_match,
            },
            "zone_summary": dict(zone_counter.most_common()),
            "matches_found": len(matches),
            "matches": [
                {
                    "rank": match.rank,
                    "frame_index": match.frame_index,
                    "timestamp_seconds": round(match.timestamp_seconds, 3),
                    "timestamp_label": match.timestamp_label,
                    "score": match.score,
                    "base_score": match.base_score,
                    "template_score": match.template_score,
                    "color_score": match.color_score,
                    "feature_score": match.feature_score,
                    "structure_score": match.structure_score,
                    "scale": match.scale,
                    "zone_id": match.zone_id,
                    "person_count": match.person_count,
                    "nearby_person_count": match.nearby_person_count,
                    "associated_track_ids": [person.track_id for person in match.associated_people],
                    "associated_people": [
                        {
                            "track_id": person.track_id,
                            "bbox": list(person.bbox),
                            "confidence": round(person.confidence, 4),
                            "zone_id": person.zone_id,
                            "is_near_object": person.is_near_object,
                            "crop_path": person.crop_path,
                        }
                        for person in match.associated_people
                    ],
                    "bbox": list(match.bbox),
                    "crop_path": match.crop_path,
                    "clip_path": match.clip_path,
                    "annotated_frame_path": match.annotated_frame_path,
                }
                for match in matches
            ],
        }

    def _write_reports(self, report: dict[str, object]) -> None:
        with config.REPORT_JSON_PATH.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        matches = report["matches"]
        with config.REPORT_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "rank",
                    "frame_index",
                    "timestamp_seconds",
                    "timestamp_label",
                    "score",
                    "base_score",
                    "template_score",
                    "color_score",
                    "feature_score",
                    "structure_score",
                    "scale",
                    "zone_id",
                    "person_count",
                    "nearby_person_count",
                    "associated_track_ids",
                    "associated_people",
                    "bbox",
                    "crop_path",
                    "clip_path",
                    "annotated_frame_path",
                ],
            )
            writer.writeheader()
            writer.writerows(matches)

        with config.SUMMARY_PATH.open("w", encoding="utf-8") as file:
            file.write("Resumen de busqueda visual\n")
            file.write(f"Query: {report['query_image_path']}\n")
            file.write(f"Video: {report['video_path']}\n")
            file.write(f"Video anotado: {report['annotated_video_path']}\n")
            file.write(f"Clips de evidencia: {report['clips_dir']}\n")
            file.write(f"Crops de personas: {report['persons_dir']}\n")
            file.write(f"Hallazgos: {report['matches_found']}\n\n")
            person_summary = report["person_tracking_summary"]
            file.write(
                "Tracking personas: "
                f"{person_summary['status']} | tracks observados={person_summary['tracks_observed']} "
                f"| max visibles={person_summary['max_people_visible']}\n\n"
            )
            file.write("Zonas activas:\n")
            for zone_id, count in report["zone_summary"].items():
                file.write(f"- {zone_id}: {count}\n")
            file.write("\n")
            for match in matches:
                file.write(
                    f"#{match['rank']:02d} | tiempo={match['timestamp_label']} "
                    f"| score={match['score']:.3f} | base={match['base_score']:.3f} | zona={match['zone_id']} "
                    f"| personas={match['person_count']} | cercanas={match['nearby_person_count']} "
                    f"| bbox={match['bbox']}\n"
                )

    def _render_preview(
        self,
        frame: np.ndarray,
        current_matches: list[InvestigationMatch],
        tracked_people: list[PersonTrack],
        best_matches: list[InvestigationMatch],
        frame_index: int,
        processed_frames: int,
        total_frames: int,
        fps: float,
        timestamp_seconds: float,
        is_sampled_frame: bool,
        last_detection_summary: str,
        last_detection_timestamp: str,
        last_detection_score: float,
        person_detection_summary: str,
        score_trace: list[tuple[float, float]],
        zone_counter: Counter[str],
    ) -> np.ndarray:
        preview_frame = self._resize_for_preview(frame.copy())
        overlay = preview_frame.copy()
        scale_x = preview_frame.shape[1] / max(frame.shape[1], 1)
        scale_y = preview_frame.shape[0] / max(frame.shape[0], 1)
        highlighted_person_ids = {
            person.track_id
            for match in current_matches
            for person in match.associated_people
            if person.is_near_object
        }

        if config.DRAW_ZONE_GRID:
            self._draw_zone_grid(overlay)

        for person in tracked_people:
            px1, py1, px2, py2 = person.bbox
            scaled_bbox = (
                int(px1 * scale_x),
                int(py1 * scale_y),
                int(px2 * scale_x),
                int(py2 * scale_y),
            )
            self._draw_person_box(
                overlay=overlay,
                bbox=scaled_bbox,
                track_id=person.track_id,
                confidence=person.confidence,
                is_associated=person.track_id in highlighted_person_ids,
            )

        for match in current_matches:
            x1, y1, x2, y2 = match.bbox
            scaled_bbox = (
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            )
            self._draw_detection_box(
                overlay=overlay,
                bbox=scaled_bbox,
                score=match.score,
                timestamp_label=match.timestamp_label,
                rank_hint=self._rank_hint(best_matches, match),
                zone_id=match.zone_id,
            )

        self._draw_header_panel(
            overlay=overlay,
            frame_index=frame_index,
            processed_frames=processed_frames,
            total_frames=total_frames,
            fps=fps,
            timestamp_seconds=timestamp_seconds,
            is_sampled_frame=is_sampled_frame,
            current_matches=current_matches,
            tracked_people=tracked_people,
            last_detection_summary=last_detection_summary,
            last_detection_timestamp=last_detection_timestamp,
            last_detection_score=last_detection_score,
            person_detection_summary=person_detection_summary,
        )
        self._draw_query_panel(overlay)
        self._draw_trace_panel(
            overlay=overlay,
            score_trace=score_trace,
            zone_counter=zone_counter,
            best_matches=best_matches,
            last_detection_timestamp=last_detection_timestamp,
            last_detection_score=last_detection_score,
        )

        preview_frame = cv2.addWeighted(overlay, 0.92, preview_frame, 0.08, 0.0)
        self._draw_footer(preview_frame)
        return preview_frame

    def _draw_detection_box(
        self,
        overlay: np.ndarray,
        bbox: tuple[int, int, int, int],
        score: float,
        timestamp_label: str,
        rank_hint: str,
        zone_id: str,
    ) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), config.BRAND_GOLD, 3)

        label = f"{zone_id} | Objeto similar {score:.3f}"
        if rank_hint:
            label = f"{rank_hint} | {label}"

        label_y = max(34, y1 - 16)
        label_width = min(overlay.shape[1] - x1 - 10, 300)
        cv2.rectangle(
            overlay,
            (x1, label_y - 24),
            (x1 + label_width, label_y + 10),
            config.BRAND_PANEL,
            thickness=-1,
        )
        cv2.putText(
            overlay,
            label[:38],
            (x1 + 8, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            config.BRAND_TEXT_PRIMARY,
            2,
        )
        cv2.putText(
            overlay,
            f"Tiempo {timestamp_label}",
            (x1 + 8, label_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            config.BRAND_GOLD_SOFT,
            1,
        )

    def _draw_person_box(
        self,
        overlay: np.ndarray,
        bbox: tuple[int, int, int, int],
        track_id: int,
        confidence: float,
        is_associated: bool,
    ) -> None:
        x1, y1, x2, y2 = bbox
        color = config.PERSON_BOX_ASSOCIATED_COLOR if is_associated else config.PERSON_BOX_COLOR
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"Persona #{track_id:02d} | {confidence:.2f}"
        if is_associated:
            label = f"Asociada #{track_id:02d} | {confidence:.2f}"

        label_y = min(max(32, y1 + 22), max(32, overlay.shape[0] - 10))
        cv2.rectangle(
            overlay,
            (x1, label_y - 22),
            (min(overlay.shape[1] - 10, x1 + 200), label_y + 6),
            config.BRAND_PANEL_ALT,
            thickness=-1,
        )
        cv2.putText(
            overlay,
            label[:32],
            (x1 + 6, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
        )

    def _draw_header_panel(
        self,
        overlay: np.ndarray,
        frame_index: int,
        processed_frames: int,
        total_frames: int,
        fps: float,
        timestamp_seconds: float,
        is_sampled_frame: bool,
        current_matches: list[InvestigationMatch],
        tracked_people: list[PersonTrack],
        last_detection_summary: str,
        last_detection_timestamp: str,
        last_detection_score: float,
        person_detection_summary: str,
    ) -> None:
        width = overlay.shape[1]
        cv2.rectangle(overlay, (0, 0), (width, 122), config.BRAND_PANEL, thickness=-1)
        cv2.rectangle(overlay, (0, 118), (width, 122), config.BRAND_GOLD, thickness=-1)

        progress = min(max(frame_index / max(total_frames - 1, 1), 0.0), 1.0)
        progress_width = int((width - 40) * progress)
        cv2.rectangle(overlay, (20, 84), (width - 20, 100), config.BRAND_BLUE, thickness=-1)
        cv2.rectangle(overlay, (20, 84), (20 + progress_width, 100), config.BRAND_GOLD, thickness=-1)

        cv2.putText(
            overlay,
            config.PREVIEW_TITLE,
            (22, 34),
            cv2.FONT_HERSHEY_DUPLEX,
            0.92,
            config.BRAND_TEXT_PRIMARY,
            2,
        )
        cv2.putText(
            overlay,
            config.PREVIEW_SUBTITLE,
            (22, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            config.BRAND_TEXT_MUTED,
            1,
        )

        engine_status = "FRAME ANALIZADO" if is_sampled_frame else "VISUALIZACION EN VIVO"
        frame_status = f"Frame {frame_index + 1}/{total_frames} | Muestreados {processed_frames}"
        time_status = f"Tiempo {self._format_seconds(timestamp_seconds)} | FPS {fps:.2f}"
        current_status = (
            f"Matches actuales {len(current_matches)} | "
            f"Personas visibles {len(tracked_people)} | "
            f"Ultimo fuerte {last_detection_timestamp} | Score {last_detection_score:.3f}"
        )

        cv2.putText(
            overlay,
            engine_status,
            (max(22, width - 360), 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            config.BRAND_TEXT_ACCENT,
            2,
        )
        cv2.putText(
            overlay,
            frame_status,
            (max(22, width - 340), 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            config.BRAND_TEXT_MUTED,
            1,
        )
        cv2.putText(
            overlay,
            time_status,
            (max(22, width - 340), 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            config.BRAND_TEXT_MUTED,
            1,
        )
        cv2.putText(
            overlay,
            current_status[: max(20, width // 12)],
            (22, 114),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            config.BRAND_GOLD_SOFT if current_matches else config.BRAND_TEXT_MUTED,
            1,
        )
        cv2.putText(
            overlay,
            person_detection_summary[: max(24, width // 10)],
            (max(22, width - 340), 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            config.BRAND_BLUE_SOFT if tracked_people else config.BRAND_TEXT_MUTED,
            1,
        )

    def _draw_query_panel(self, overlay: np.ndarray) -> None:
        if not config.DISPLAY_QUERY_THUMBNAIL or self.query_thumbnail is None:
            return

        thumb = self.query_thumbnail
        height, width = thumb.shape[:2]
        margin = 22
        x1 = overlay.shape[1] - width - margin
        y1 = 136
        x2 = x1 + width
        y2 = y1 + height

        cv2.rectangle(overlay, (x1 - 10, y1 - 38), (x2 + 10, y2 + 10), config.BRAND_PANEL_ALT, thickness=-1)
        cv2.rectangle(overlay, (x1 - 10, y1 - 38), (x2 + 10, y2 + 10), config.BRAND_GOLD, thickness=1)
        cv2.putText(
            overlay,
            "OBJETO DE REFERENCIA",
            (x1 - 4, y1 - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            config.BRAND_TEXT_ACCENT,
            2,
        )
        overlay[y1:y2, x1:x2] = thumb

    def _draw_trace_panel(
        self,
        overlay: np.ndarray,
        score_trace: list[tuple[float, float]],
        zone_counter: Counter[str],
        best_matches: list[InvestigationMatch],
        last_detection_timestamp: str,
        last_detection_score: float,
    ) -> None:
        panel_width = min(360, overlay.shape[1] - 44)
        panel_height = 240
        x1 = overlay.shape[1] - panel_width - 22
        y1 = min(max(310, 136), max(136, overlay.shape[0] - panel_height - 60))
        x2 = x1 + panel_width
        y2 = y1 + panel_height

        cv2.rectangle(overlay, (x1, y1), (x2, y2), config.BRAND_PANEL_ALT, thickness=-1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), config.BRAND_GOLD, thickness=1)
        cv2.putText(
            overlay,
            "TRAZA TEMPORAL Y ZONAS",
            (x1 + 12, y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            config.BRAND_TEXT_ACCENT,
            2,
        )

        chart_x1 = x1 + 14
        chart_y1 = y1 + 42
        chart_x2 = x2 - 14
        chart_y2 = y1 + 128
        cv2.rectangle(overlay, (chart_x1, chart_y1), (chart_x2, chart_y2), config.BRAND_BLUE, thickness=-1)
        cv2.rectangle(overlay, (chart_x1, chart_y1), (chart_x2, chart_y2), config.BRAND_GOLD, thickness=1)

        if len(score_trace) >= 2:
            max_score = max(max(score for _, score in score_trace), config.SIMILARITY_THRESHOLD)
            min_score = min(score for _, score in score_trace)
            score_range = max(max_score - min_score, 1e-6)
            points: list[tuple[int, int]] = []
            for index, (_, score) in enumerate(score_trace):
                px = chart_x1 + int((index / max(len(score_trace) - 1, 1)) * (chart_x2 - chart_x1))
                normalized = (score - min_score) / score_range
                py = chart_y2 - int(normalized * (chart_y2 - chart_y1))
                points.append((px, py))

            for index in range(1, len(points)):
                cv2.line(overlay, points[index - 1], points[index], config.BRAND_GOLD, 2)

            cv2.putText(
                overlay,
                f"Score actual {score_trace[-1][1]:.3f}",
                (chart_x1 + 4, chart_y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                config.BRAND_TEXT_PRIMARY,
                1,
            )
        else:
            cv2.putText(
                overlay,
                "Sin suficiente historial para traza",
                (chart_x1 + 8, chart_y1 + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                config.BRAND_TEXT_MUTED,
                1,
            )

        best_text = "Top score historico: 0.000"
        if best_matches:
            best_text = f"Top score historico: {best_matches[0].score:.3f} en {best_matches[0].timestamp_label}"
        cv2.putText(
            overlay,
            best_text[:40],
            (x1 + 14, y1 + 158),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            config.BRAND_TEXT_PRIMARY,
            1,
        )
        cv2.putText(
            overlay,
            f"Ultima fuerte: {last_detection_timestamp} | {last_detection_score:.3f}",
            (x1 + 14, y1 + 178),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            config.BRAND_GOLD_SOFT,
            1,
        )
        cv2.putText(
            overlay,
            "Zonas mas activas",
            (x1 + 14, y1 + 202),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            config.BRAND_TEXT_MUTED,
            1,
        )
        zone_lines = zone_counter.most_common(4)
        if not zone_lines:
            zone_lines = [("sin actividad", 0)]
        for index, (zone_id, count) in enumerate(zone_lines):
            text = f"{zone_id}: {count} detecciones"
            cv2.putText(
                overlay,
                text[:34],
                (x1 + 18, y1 + 224 + index * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.49,
                config.BRAND_TEXT_PRIMARY,
                1,
            )

    def _draw_zone_grid(self, overlay: np.ndarray) -> None:
        height, width = overlay.shape[:2]
        row_height = height / config.ZONE_GRID_ROWS
        col_width = width / config.ZONE_GRID_COLS

        for row in range(1, config.ZONE_GRID_ROWS):
            y = int(row * row_height)
            cv2.line(overlay, (0, y), (width, y), config.BRAND_GRID, 1)
        for col in range(1, config.ZONE_GRID_COLS):
            x = int(col * col_width)
            cv2.line(overlay, (x, 0), (x, height), config.BRAND_GRID, 1)

        for row in range(config.ZONE_GRID_ROWS):
            for col in range(config.ZONE_GRID_COLS):
                zone_id = f"{chr(65 + row)}{col + 1}"
                x = int(col * col_width) + 10
                y = int(row * row_height) + 24
                cv2.putText(
                    overlay,
                    zone_id,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    config.BRAND_BLUE_SOFT,
                    1,
                )

    def _draw_footer(self, frame: np.ndarray) -> None:
        footer_text = "Innova Monitoring | Q o ESC para detener. El sistema guarda crops, clips, reporte y video anotado."
        y = frame.shape[0] - 18
        cv2.rectangle(frame, (0, frame.shape[0] - 44), (frame.shape[1], frame.shape[0]), config.BRAND_PANEL, thickness=-1)
        cv2.putText(frame, footer_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, config.BRAND_TEXT_MUTED, 1)

    def _resolve_zone_id(
        self,
        frame_shape: tuple[int, int],
        bbox: tuple[int, int, int, int],
    ) -> str:
        frame_height, frame_width = frame_shape
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        col = min(config.ZONE_GRID_COLS - 1, max(0, int(center_x / max(frame_width, 1) * config.ZONE_GRID_COLS)))
        row = min(config.ZONE_GRID_ROWS - 1, max(0, int(center_y / max(frame_height, 1) * config.ZONE_GRID_ROWS)))
        return f"{chr(65 + row)}{col + 1}"

    def _rank_hint(self, best_matches: list[InvestigationMatch], current_match: InvestigationMatch) -> str:
        for rank, match in enumerate(best_matches[:5], start=1):
            if (
                abs(match.timestamp_seconds - current_match.timestamp_seconds) <= 0.05
                and self._iou(match.bbox, current_match.bbox) >= 0.80
            ):
                return f"Top #{rank:02d}"
        return ""

    def _initialize_preview_window(self) -> None:
        try:
            cv2.namedWindow(config.PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        except cv2.error:
            self.preview_enabled = False
            print("Vista previa deshabilitada: OpenCV GUI no disponible en este entorno.", flush=True)

    def _show_preview(self, preview_frame: np.ndarray) -> bool:
        try:
            cv2.imshow(config.PREVIEW_WINDOW_NAME, preview_frame)
            key = cv2.waitKey(config.PREVIEW_WAIT_MS) & 0xFF
        except cv2.error:
            self.preview_enabled = False
            return True

        return key not in (27, ord("q"), ord("Q"))

    def _write_preview_frame(self, preview_frame: np.ndarray, fps: float) -> None:
        if not config.SAVE_ANNOTATED_VIDEO:
            return

        if self.preview_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.preview_writer = cv2.VideoWriter(
                str(config.ANNOTATED_VIDEO_PATH),
                fourcc,
                fps if fps > 0 else 20.0,
                (preview_frame.shape[1], preview_frame.shape[0]),
            )
            if not self.preview_writer.isOpened():
                print("No se pudo crear el video anotado de salida.", flush=True)
                self.preview_writer = None
                return

        self.preview_writer.write(preview_frame)

    def _close_preview(self) -> None:
        if self.preview_writer is not None:
            self.preview_writer.release()
            self.preview_writer = None

        if self.preview_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    def _emit_runtime_update(
        self,
        preview_frame: np.ndarray | None,
        frame_index: int,
        total_frames: int,
        timestamp_seconds: float,
        processed_frames: int,
        best_matches: list[InvestigationMatch],
        current_matches: list[InvestigationMatch],
        tracked_people: list[PersonTrack],
        last_detection_summary: str,
        last_detection_timestamp: str,
        last_detection_score: float,
        person_detection_summary: str,
        is_sampled_frame: bool,
    ) -> None:
        if self.status_callback is not None and is_sampled_frame:
            progress_ratio = min(max((frame_index + 1) / max(total_frames, 1), 0.0), 1.0)
            self.status_callback(
                {
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "processed_frames": processed_frames,
                    "timestamp_seconds": timestamp_seconds,
                    "timestamp_label": self._format_seconds(timestamp_seconds),
                    "matches_found": len(best_matches),
                    "current_matches": len(current_matches),
                    "people_visible": len(tracked_people),
                    "last_detection_summary": last_detection_summary,
                    "last_detection_timestamp": last_detection_timestamp,
                    "last_detection_score": last_detection_score,
                    "person_detection_summary": person_detection_summary,
                    "progress_ratio": progress_ratio,
                }
            )

        if preview_frame is not None and self.preview_callback is not None and (
            (is_sampled_frame and processed_frames % config.PREVIEW_CALLBACK_SAMPLE_INTERVAL == 0)
            or frame_index == 0
            or bool(current_matches)
            or bool(tracked_people)
            or frame_index == total_frames - 1
        ):
            self.preview_callback(preview_frame.copy())

    def _format_seconds(self, seconds: float) -> str:
        return str(timedelta(seconds=int(seconds)))

    def _iou(
        self,
        first_box: tuple[int, int, int, int],
        second_box: tuple[int, int, int, int],
    ) -> float:
        first_x1, first_y1, first_x2, first_y2 = first_box
        second_x1, second_y1, second_x2, second_y2 = second_box

        intersection_x1 = max(first_x1, second_x1)
        intersection_y1 = max(first_y1, second_y1)
        intersection_x2 = min(first_x2, second_x2)
        intersection_y2 = min(first_y2, second_y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height

        first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
        second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
        union_area = first_area + second_area - intersection_area

        if union_area == 0:
            return 0.0
        return intersection_area / union_area
