from __future__ import annotations

import numpy as np

from .color_rules import looks_like_color
from .detectors import crop_detection
from .models import Detection, EventRule


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def rule_matches(rule: EventRule, detection: Detection, frame: np.ndarray) -> tuple[bool, dict[str, object]]:
    """Evaluate one rule against one detection.

    The returned dict is stored in event metadata so you can later understand
    why an event was saved.
    """

    if detection.confidence < rule.min_confidence:
        return False, {"reason": "confidence_below_rule"}

    if rule.class_names and detection.class_name not in set(rule.class_names):
        return False, {"reason": "class_not_in_rule"}

    if rule.type in ("class", "custom_model"):
        return True, {"reason": "class_match", "model_name": detection.model_name}

    crop = crop_detection(frame, detection)
    target_color = (rule.color or "").strip().lower()
    if target_color == "grey":
        target_color = "gray"

    if rule.type == "vehicle_color":
        if detection.class_name not in VEHICLE_CLASSES:
            return False, {"reason": "not_vehicle"}
        threshold = rule.color_threshold if rule.color_threshold is not None else 0.32
        ok, score, profile = looks_like_color(crop, target_color, region="vehicle_body", threshold=threshold)
        dominant_color = str(profile.get("dominant_color") or "")
        compatible_colors = {target_color}
        if target_color == "white":
            compatible_colors.add("silver")
        if target_color == "gray":
            compatible_colors.add("silver")
        if target_color == "silver":
            compatible_colors.add("gray")
        if dominant_color not in compatible_colors:
            ok = False
        return ok, {
            "reason": "vehicle_color",
            "target_color": target_color,
            "color_score": score,
            "color_threshold": threshold,
            "dominant_color": dominant_color,
            "color_profile": profile,
        }

    if rule.type == "person_upper_color":
        if detection.class_name != "person":
            return False, {"reason": "not_person"}
        threshold = rule.color_threshold if rule.color_threshold is not None else 0.16
        ok, score, profile = looks_like_color(crop, target_color, region="upper", threshold=threshold)
        return ok, {
            "reason": "person_upper_color",
            "target_color": target_color,
            "color_score": score,
            "color_threshold": threshold,
            "color_profile": profile,
        }

    return False, {"reason": "unknown_rule_type"}
