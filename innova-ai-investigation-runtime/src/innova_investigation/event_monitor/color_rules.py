from __future__ import annotations

import cv2
import numpy as np


SUPPORTED_COLORS = {"white", "black", "gray", "grey", "silver", "red", "blue", "green", "yellow"}


def _crop_region(image: np.ndarray, region: str) -> np.ndarray:
    """Return the part of a crop that best represents the requested rule.

    For a person shirt rule, the top half is usually more useful than the
    whole body because pants/shoes/background can dominate the color.
    """

    if image.size == 0:
        return image
    height, width = image.shape[:2]
    if region == "upper":
        return image[: max(1, int(height * 0.55)), :]
    if region == "center":
        x1 = int(width * 0.15)
        x2 = int(width * 0.85)
        y1 = int(height * 0.15)
        y2 = int(height * 0.85)
        return image[y1:y2, x1:x2]
    if region == "vehicle_body":
        # Most vehicle crops include road, plants, wheels and black bumpers near
        # the bottom. For color rules we favor the upper/middle body panels,
        # accepting that this may miss some odd angles but reduces false colors.
        x1 = int(width * 0.08)
        x2 = int(width * 0.92)
        y1 = int(height * 0.12)
        y2 = int(height * 0.68)
        return image[y1:y2, x1:x2]
    return image


def _valid_pixel_mask(hsv: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Ignore extreme highlights/shadows that often come from glare or windows."""

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    return (value > 10) & ~((gray > 248) & (saturation < 8))


def _color_mask(hsv: np.ndarray, gray: np.ndarray, color: str) -> np.ndarray:
    color_key = color.strip().lower()
    if color_key == "grey":
        color_key = "gray"
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    if color_key == "white":
        return (gray > 168) & (sat < 78)
    if color_key == "black":
        return (gray < 72) & (val < 92)
    if color_key in {"gray", "silver"}:
        # Gray and silver are low-saturation colors. Silver tends to be brighter
        # than gray, but camera exposure can blur that line, so both overlap.
        if color_key == "silver":
            return (sat < 58) & (gray >= 132) & (gray <= 205)
        return (sat < 62) & (gray >= 72) & (gray <= 155)
    if color_key == "red":
        return ((hue < 10) | (hue > 170)) & (sat > 70) & (val > 55)
    if color_key == "blue":
        return (hue > 90) & (hue < 135) & (sat > 55) & (val > 45)
    if color_key == "green":
        return (hue > 35) & (hue < 90) & (sat > 55) & (val > 45)
    if color_key == "yellow":
        return (hue > 18) & (hue < 38) & (sat > 65) & (val > 65)
    return np.zeros(gray.shape, dtype=bool)


def color_profile(image_bgr: np.ndarray, *, region: str = "full") -> dict[str, object]:
    """Return explainable color scores for a crop.

    The scores are intentionally transparent. They are not a neural classifier,
    but they let us tune the rules using real CCTV examples from each property.
    """

    crop = _crop_region(image_bgr, region)
    if crop.size == 0:
        return {"dominant_color": "", "scores": {}, "valid_pixel_ratio": 0.0}

    # A small blur reduces compression noise from RTSP streams.
    crop = cv2.GaussianBlur(crop, (5, 5), 0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    valid_mask = _valid_pixel_mask(hsv, gray)
    valid_count = max(1, int(np.count_nonzero(valid_mask)))

    scores: dict[str, float] = {}
    for color in ("white", "black", "gray", "silver", "red", "blue", "green", "yellow"):
        mask = _color_mask(hsv, gray, color) & valid_mask
        scores[color] = round(float(np.count_nonzero(mask)) / float(valid_count), 4)

    dominant_color = _dominant_color(scores)
    return {
        "dominant_color": dominant_color,
        "scores": scores,
        "valid_pixel_ratio": round(float(valid_count) / float(max(1, gray.size)), 4),
    }


def _dominant_color(scores: dict[str, float]) -> str:
    if not scores:
        return ""
    # Neutral CCTV colors are noisy. Prefer white when a large part of the
    # vehicle is bright/low saturation, otherwise shadows turn white vehicles
    # into false gray/black events.
    if scores.get("white", 0.0) >= 0.28:
        return "white"
    if scores.get("black", 0.0) >= 0.42:
        return "black"
    chromatic = {key: scores.get(key, 0.0) for key in ("red", "blue", "green", "yellow")}
    chromatic_color = max(chromatic, key=chromatic.get)
    if chromatic[chromatic_color] >= 0.28:
        return chromatic_color
    if scores.get("silver", 0.0) >= scores.get("gray", 0.0):
        return "silver"
    return max(scores, key=scores.get)


def color_match_score(image_bgr: np.ndarray, color: str, *, region: str = "full") -> float:
    """Return an approximate 0..1 score for a color inside an image crop."""

    color_key = color.strip().lower()
    if color_key == "grey":
        color_key = "gray"
    profile = color_profile(image_bgr, region=region)
    scores = profile.get("scores") or {}
    return float(scores.get(color_key, 0.0))


def looks_like_color(
    image_bgr: np.ndarray,
    color: str,
    *,
    region: str = "full",
    threshold: float = 0.18,
) -> tuple[bool, float, dict[str, object]]:
    """Return whether the crop matches a target color, score and diagnostics."""

    profile = color_profile(image_bgr, region=region)
    color_key = color.strip().lower()
    if color_key == "grey":
        color_key = "gray"
    scores = profile.get("scores") or {}
    score = float(scores.get(color_key, 0.0))
    return score >= threshold, round(score, 4), profile
