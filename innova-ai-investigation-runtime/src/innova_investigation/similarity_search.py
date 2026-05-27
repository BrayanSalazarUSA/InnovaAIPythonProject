from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass(slots=True)
class QuerySignature:
    image: np.ndarray
    gray: np.ndarray
    edges: np.ndarray
    histogram: np.ndarray
    scaled_templates: tuple[tuple[float, int, int, np.ndarray, np.ndarray], ...]
    keypoints: tuple[cv2.KeyPoint, ...]
    keypoints_count: int
    descriptors: np.ndarray | None
    width: int
    height: int
    aspect_ratio: float


@dataclass(slots=True)
class MatchCandidate:
    bbox: tuple[int, int, int, int]
    scale: float
    template_score: float
    color_score: float
    feature_score: float
    structure_score: float
    final_score: float


class SimilaritySearcher:
    def __init__(self) -> None:
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def build_query_signature(self, image: np.ndarray) -> QuerySignature:
        prepared = self._prepare_image(image)
        gray = self._prepare_gray(prepared)
        edges = self._compute_edges(gray)
        histogram = self._compute_histogram(prepared)
        scaled_templates = self._build_scaled_templates(gray, edges)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        height, width = prepared.shape[:2]

        return QuerySignature(
            image=prepared,
            gray=gray,
            edges=edges,
            histogram=histogram,
            scaled_templates=scaled_templates,
            keypoints=tuple(keypoints),
            keypoints_count=len(keypoints),
            descriptors=descriptors,
            width=width,
            height=height,
            aspect_ratio=width / max(height, 1),
        )

    def search(self, frame: np.ndarray, query: QuerySignature) -> list[MatchCandidate]:
        if frame.size == 0:
            return []

        frame_gray = self._prepare_gray(frame)
        frame_edges = self._compute_edges(frame_gray)

        raw_candidates = self._propose_regions(frame_gray, frame_edges, query)

        scored_candidates: list[MatchCandidate] = []
        for bbox, scale, template_score in raw_candidates[: config.MAX_SCORING_CANDIDATES]:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            template_weight, color_weight, feature_weight, structure_weight = self._score_weights(query)
            color_score = self._compare_histograms(query, crop)
            structure_score = self._compare_structure(query, crop)
            max_possible_score = float(
                (template_weight * template_score)
                + (color_weight * color_score)
                + (feature_weight * 1.0)
                + (structure_weight * structure_score)
            )
            if max_possible_score < config.SIMILARITY_THRESHOLD:
                continue

            feature_score = self._compare_features(query, crop)
            final_score = float(
                (template_weight * template_score)
                + (color_weight * color_score)
                + (feature_weight * feature_score)
                + (structure_weight * structure_score)
            )

            if final_score < config.SIMILARITY_THRESHOLD:
                continue

            scored_candidates.append(
                MatchCandidate(
                    bbox=bbox,
                    scale=scale,
                    template_score=template_score,
                    color_score=color_score,
                    feature_score=feature_score,
                    structure_score=structure_score,
                    final_score=final_score,
                )
            )

        scored_candidates.sort(key=lambda candidate: candidate.final_score, reverse=True)
        deduplicated = self._deduplicate_candidates(scored_candidates)
        return deduplicated[: config.TOP_CANDIDATES_PER_FRAME]

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        longest_side = max(height, width)
        if longest_side <= 320:
            return image.copy()

        scale = 320 / longest_side
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def _prepare_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _compute_edges(self, gray_image: np.ndarray) -> np.ndarray:
        return cv2.Canny(gray_image, 60, 160)

    def _compute_histogram(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        return cv2.normalize(histogram, None).flatten()

    def _build_scaled_templates(
        self,
        query_gray: np.ndarray,
        query_edges: np.ndarray,
    ) -> tuple[tuple[float, int, int, np.ndarray, np.ndarray], ...]:
        query_height, query_width = query_gray.shape[:2]
        templates: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
        for scale in config.TEMPLATE_SCALES:
            template_width = max(int(query_width * scale), config.MIN_WINDOW_SIZE)
            template_height = max(int(query_height * scale), config.MIN_WINDOW_SIZE)
            template_gray = cv2.resize(
                query_gray,
                (template_width, template_height),
                interpolation=cv2.INTER_AREA,
            )
            template_edges = cv2.resize(
                query_edges,
                (template_width, template_height),
                interpolation=cv2.INTER_AREA,
            )
            templates.append((scale, template_width, template_height, template_gray, template_edges))
        return tuple(templates)

    def _propose_regions(
        self,
        frame_gray: np.ndarray,
        frame_edges: np.ndarray,
        query: QuerySignature,
    ) -> list[tuple[tuple[int, int, int, int], float, float]]:
        height, width = frame_gray.shape[:2]
        candidates: list[tuple[tuple[int, int, int, int], float, float]] = []

        for scale, template_width, template_height, template_gray, template_edges in query.scaled_templates:
            if template_width >= width or template_height >= height:
                continue

            gray_response = cv2.matchTemplate(
                frame_gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED,
            )
            edge_response = cv2.matchTemplate(
                frame_edges,
                template_edges,
                cv2.TM_CCOEFF_NORMED,
            )
            response = (0.65 * gray_response) + (0.35 * edge_response)
            candidates.extend(
                self._extract_top_matches(response, template_width, template_height, scale)
            )

        contour_candidates = self._extract_contour_proposals(frame_gray, frame_edges, query)
        candidates.extend(contour_candidates)
        candidates.sort(key=lambda item: item[2], reverse=True)
        return self._deduplicate_raw_candidates(candidates)

    def _extract_contour_proposals(
        self,
        frame_gray: np.ndarray,
        frame_edges: np.ndarray,
        query: QuerySignature,
    ) -> list[tuple[tuple[int, int, int, int], float, float]]:
        height, width = frame_gray.shape[:2]
        frame_area = float(height * width)
        if frame_area <= 0:
            return []

        dilated_edges = cv2.dilate(frame_edges, np.ones((5, 5), np.uint8), iterations=1)
        thresholded = cv2.adaptiveThreshold(
            frame_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            6,
        )
        merged_mask = cv2.bitwise_or(dilated_edges, thresholded)
        merged_mask = cv2.morphologyEx(
            merged_mask,
            cv2.MORPH_CLOSE,
            np.ones((7, 7), np.uint8),
            iterations=1,
        )

        contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[tuple[int, int, int, int], float, float]] = []

        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < config.MIN_WINDOW_SIZE or box_height < config.MIN_WINDOW_SIZE:
                continue

            area_ratio = (box_width * box_height) / frame_area
            if area_ratio < config.CONTOUR_MIN_AREA_RATIO or area_ratio > config.CONTOUR_MAX_AREA_RATIO:
                continue

            bbox = self._expand_bbox(
                (x, y, x + box_width, y + box_height),
                width,
                height,
                config.CONTOUR_EXPAND_RATIO,
            )
            x1, y1, x2, y2 = bbox
            proposal_width = x2 - x1
            proposal_height = y2 - y1
            if proposal_width < config.MIN_WINDOW_SIZE or proposal_height < config.MIN_WINDOW_SIZE:
                continue

            aspect_score = self._aspect_similarity(proposal_width / max(proposal_height, 1), query.aspect_ratio)
            if aspect_score < config.ASPECT_RATIO_TOLERANCE:
                continue

            edge_density = float(np.count_nonzero(frame_edges[y1:y2, x1:x2])) / max(proposal_width * proposal_height, 1)
            size_score = self._size_similarity(proposal_width, proposal_height, query.width, query.height)
            quick_score = float(
                np.clip((0.45 * aspect_score) + (0.35 * size_score) + (0.20 * min(edge_density * 8.0, 1.0)), 0.0, 1.0)
            )
            scale = proposal_width / max(query.width, 1)
            candidates.append((bbox, scale, quick_score))

        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates[: config.CONTOUR_PROPOSAL_LIMIT]

    def _extract_top_matches(
        self,
        response: np.ndarray,
        template_width: int,
        template_height: int,
        scale: float,
    ) -> list[tuple[tuple[int, int, int, int], float, float]]:
        response_copy = response.copy()
        matches: list[tuple[tuple[int, int, int, int], float, float]] = []

        for _ in range(config.TOP_TEMPLATE_MATCHES_PER_SCALE):
            _, max_value, _, max_location = cv2.minMaxLoc(response_copy)
            if max_value < config.MIN_TEMPLATE_SCORE:
                break

            x1, y1 = max_location
            bbox = (x1, y1, x1 + template_width, y1 + template_height)
            matches.append((bbox, scale, float(max_value)))

            suppression_margin = max(6, int(min(template_width, template_height) * 0.35))
            start_x = max(0, x1 - suppression_margin)
            start_y = max(0, y1 - suppression_margin)
            end_x = min(response_copy.shape[1], x1 + suppression_margin)
            end_y = min(response_copy.shape[0], y1 + suppression_margin)
            response_copy[start_y:end_y, start_x:end_x] = -1.0

        return matches

    def _compare_histograms(self, query: QuerySignature, crop: np.ndarray) -> float:
        crop_resized = cv2.resize(
            crop,
            (query.width, query.height),
            interpolation=cv2.INTER_AREA,
        )
        crop_histogram = self._compute_histogram(crop_resized)
        score = cv2.compareHist(
            query.histogram.astype(np.float32),
            crop_histogram.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))

    def _compare_features(self, query: QuerySignature, crop: np.ndarray) -> float:
        if query.descriptors is None or query.keypoints_count == 0:
            return 0.0

        crop_resized = cv2.resize(
            crop,
            (query.width, query.height),
            interpolation=cv2.INTER_AREA,
        )
        crop_gray = self._prepare_gray(crop_resized)
        crop_keypoints, crop_descriptors = self.orb.detectAndCompute(crop_gray, None)

        if crop_descriptors is None or len(crop_keypoints) == 0:
            return 0.0

        knn_matches = self.matcher.knnMatch(query.descriptors, crop_descriptors, k=2)
        ratio_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.78 * second.distance:
                ratio_matches.append(first)

        # Evita inflar el score cuando muchos descriptores de la query
        # caen sobre el mismo punto del crop.
        unique_matches_by_train_index: dict[int, cv2.DMatch] = {}
        for match in ratio_matches:
            current_best = unique_matches_by_train_index.get(match.trainIdx)
            if current_best is None or match.distance < current_best.distance:
                unique_matches_by_train_index[match.trainIdx] = match

        good_matches = list(unique_matches_by_train_index.values())
        if len(good_matches) < config.MIN_FEATURE_MATCHES:
            return 0.0

        query_points = np.float32(
            [query.keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        crop_points = np.float32([crop_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(query_points, crop_points, cv2.RANSAC, 4.0)
        if homography is None or inlier_mask is None:
            return 0.0

        inlier_mask = inlier_mask.ravel().astype(bool)
        inlier_matches = [match for match, is_inlier in zip(good_matches, inlier_mask) if is_inlier]
        if len(inlier_matches) < config.MIN_FEATURE_MATCHES:
            return 0.0

        inlier_ratio = len(inlier_matches) / len(good_matches)
        normalized_coverage = len(inlier_matches) / max(min(query.keypoints_count, len(crop_keypoints)), 1)
        average_distance = float(np.mean([match.distance for match in inlier_matches]))
        distance_score = float(np.clip(1.0 - (average_distance / 70.0), 0.0, 1.0))
        spread_score = self._keypoint_spread(crop_keypoints, inlier_matches, query.width * query.height)

        feature_score = (
            (0.45 * inlier_ratio)
            + (0.30 * min(normalized_coverage * 1.5, 1.0))
            + (0.15 * distance_score)
            + (0.10 * spread_score)
        )
        return float(np.clip(feature_score, 0.0, 1.0))

    def _compare_structure(self, query: QuerySignature, crop: np.ndarray) -> float:
        crop_resized = cv2.resize(
            crop,
            (query.width, query.height),
            interpolation=cv2.INTER_AREA,
        )
        crop_gray = self._prepare_gray(crop_resized)
        crop_edges = self._compute_edges(crop_gray)
        edge_fill_query = float(np.count_nonzero(query.edges)) / max(query.width * query.height, 1)
        edge_fill_crop = float(np.count_nonzero(crop_edges)) / max(query.width * query.height, 1)

        gray_score = cv2.matchTemplate(
            crop_gray,
            query.gray,
            cv2.TM_CCOEFF_NORMED,
        )[0][0]
        edge_score = cv2.matchTemplate(
            crop_edges,
            query.edges,
            cv2.TM_CCOEFF_NORMED,
        )[0][0]
        crop_aspect_ratio = crop.shape[1] / max(crop.shape[0], 1)
        aspect_score = 1.0 - min(
            abs(crop_aspect_ratio - query.aspect_ratio) / max(query.aspect_ratio, 1e-6),
            1.0,
        )
        edge_fill_score = 1.0 - min(abs(edge_fill_crop - edge_fill_query) / max(edge_fill_query, 0.05), 1.0)

        combined_score = (
            (0.34 * gray_score)
            + (0.28 * edge_score)
            + (0.18 * aspect_score)
            + (0.20 * edge_fill_score)
        )
        return float(np.clip((combined_score + 1.0) / 2.0, 0.0, 1.0))

    def _score_weights(self, query: QuerySignature) -> tuple[float, float, float, float]:
        if query.keypoints_count >= 40:
            return 0.40, config.SIMILARITY_COLOR_WEIGHT_STRONG, 0.28, 0.16
        if query.keypoints_count >= 20:
            return 0.40, config.SIMILARITY_COLOR_WEIGHT_MEDIUM, 0.22, 0.20
        return 0.36, config.SIMILARITY_COLOR_WEIGHT_LOW_TEXTURE, 0.08, 0.44

    def _deduplicate_raw_candidates(
        self,
        candidates: list[tuple[tuple[int, int, int, int], float, float]],
    ) -> list[tuple[tuple[int, int, int, int], float, float]]:
        deduplicated: list[tuple[tuple[int, int, int, int], float, float]] = []

        for candidate in candidates:
            bbox, _, score = candidate
            should_add = True
            for index, existing in enumerate(deduplicated):
                existing_bbox, _, existing_score = existing
                if self._iou(bbox, existing_bbox) >= 0.45:
                    if score > existing_score:
                        deduplicated[index] = candidate
                    should_add = False
                    break
            if should_add:
                deduplicated.append(candidate)

        return deduplicated

    def _deduplicate_candidates(self, candidates: list[MatchCandidate]) -> list[MatchCandidate]:
        deduplicated: list[MatchCandidate] = []

        for candidate in candidates:
            should_add = True
            for index, existing in enumerate(deduplicated):
                if self._iou(candidate.bbox, existing.bbox) >= 0.45:
                    if candidate.final_score > existing.final_score:
                        deduplicated[index] = candidate
                    should_add = False
                    break
            if should_add:
                deduplicated.append(candidate)

        return deduplicated

    def _keypoint_spread(
        self,
        crop_keypoints: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
        reference_area: int,
    ) -> float:
        if not matches:
            return 0.0

        points = np.float32([crop_keypoints[match.trainIdx].pt for match in matches])
        if len(points) < 2:
            return 0.0

        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        spread_area = max(0.0, (max_x - min_x) * (max_y - min_y))
        return float(np.clip(spread_area / max(reference_area, 1), 0.0, 1.0))

    def _expand_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        expand_ratio: float,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1
        expand_x = int(round(box_width * expand_ratio))
        expand_y = int(round(box_height * expand_ratio))
        return (
            max(0, x1 - expand_x),
            max(0, y1 - expand_y),
            min(frame_width, x2 + expand_x),
            min(frame_height, y2 + expand_y),
        )

    def _aspect_similarity(self, candidate_ratio: float, query_ratio: float) -> float:
        if candidate_ratio <= 0 or query_ratio <= 0:
            return 0.0
        return float(np.clip(1.0 - abs(candidate_ratio - query_ratio) / max(query_ratio, 1e-6), 0.0, 1.0))

    def _size_similarity(
        self,
        candidate_width: int,
        candidate_height: int,
        query_width: int,
        query_height: int,
    ) -> float:
        width_ratio = candidate_width / max(query_width, 1)
        height_ratio = candidate_height / max(query_height, 1)
        width_score = 1.0 - min(abs(width_ratio - 1.0), 1.0)
        height_score = 1.0 - min(abs(height_ratio - 1.0), 1.0)
        return float(np.clip((width_score + height_score) / 2.0, 0.0, 1.0))

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
