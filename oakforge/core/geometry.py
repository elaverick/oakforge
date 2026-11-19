"""Reusable geometric operations for branch and foliage generation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def normalize(vector: np.ndarray, fallback: Iterable[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    """Return a unit vector while safely handling degenerate directions."""
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return vector / magnitude


def cubic_bezier_points(
    start: np.ndarray,
    control_one: np.ndarray,
    control_two: np.ndarray,
    end: np.ndarray,
    segment_count: int,
) -> list[np.ndarray]:
    """Sample a smooth growth path between a branch base and its tip."""
    sampled_points: list[np.ndarray] = []
    for sample_index in range(segment_count + 1):
        amount = sample_index / segment_count
        inverse_amount = 1.0 - amount
        point = (
            inverse_amount ** 3 * start
            + 3.0 * inverse_amount ** 2 * amount * control_one
            + 3.0 * inverse_amount * amount ** 2 * control_two
            + amount ** 3 * end
        )
        sampled_points.append(point.astype(np.float64))
    return sampled_points


def perpendicular_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build two stable perpendicular axes around a branch direction."""
    direction = normalize(direction)
    reference_axis = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, reference_axis))) > 0.92:
        reference_axis = np.array([1.0, 0.0, 0.0])
    side_axis = normalize(np.cross(direction, reference_axis), (0.0, 0.0, 1.0))
    normal_axis = normalize(np.cross(side_axis, direction), (1.0, 0.0, 0.0))
    return side_axis, normal_axis


def rotate_around_axis(vector: np.ndarray, axis: np.ndarray, angle_radians: float) -> np.ndarray:
    """Rotate a direction around an axis for radial branch distribution."""
    axis = normalize(axis)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return normalize(
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * np.dot(axis, vector) * (1.0 - cosine)
    )


def transported_frame(
    direction: np.ndarray,
    previous_side_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Carry a stable cross-section frame along a curved growth path.

    Parallel transport avoids the sudden twisting that occurs when every ring
    independently chooses a new world-space reference axis.
    """
    direction = normalize(direction)
    if previous_side_axis is None:
        return perpendicular_frame(direction)

    side_axis = previous_side_axis - direction * np.dot(previous_side_axis, direction)
    if np.linalg.norm(side_axis) <= 1e-8:
        return perpendicular_frame(direction)

    side_axis = normalize(side_axis)
    normal_axis = normalize(np.cross(side_axis, direction))
    return side_axis, normal_axis
