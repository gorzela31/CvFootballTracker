#!/usr/bin/env python3
"""
Generate a reference (annotation-derived) soccer-pitch homography from
SoccerNet-style semantic line annotations.

Input:
  - broadcast image (JPG/PNG)
  - JSON with semantic pitch-line annotations in normalized coordinates

Output:
  - JSON with H_pitch_to_image and H_image_to_pitch
  - PNG visualization with the pitch projection and mapped demo points

Coordinate system on the pitch (same convention as the SoccerNet calibration
baseline used in the supplied project):
  X = 0 at the center line, negative toward the left goal, positive toward right
  Y = 0 on the pitch center axis, negative toward the top touchline,
      positive toward the bottom touchline
  units = meters

Example:
    python generate_homography_gt.py \
        --image 00000.jpg \
        --annotation 00000.json \
        --output-dir gt_demo

To map your own image points later, e.g. bottom-centers of player boxes:
    python generate_homography_gt.py \
        --image 00000.jpg \
        --annotation 00000.json \
        --output-dir gt_demo \
        --points 145,260 665,305

Dependencies:
    pip install numpy opencv-python matplotlib scipy

SciPy is optional. If available, the initial line-DLT homography is refined by
robust nonlinear least squares to minimize point-to-projected-line error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:  # refinement is optional
    least_squares = None


# SoccerNet / IFAB canonical dimensions used by the supplied baseline.
PENALTY_AREA_WIDTH = 40.32
PENALTY_AREA_LENGTH = 16.5
GOAL_AREA_WIDTH = 18.32
GOAL_AREA_LENGTH = 5.5
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_MARK_DISTANCE = 11.0


Point2 = Tuple[float, float]
LineEndpoints = Tuple[Point2, Point2]


def build_pitch_line_endpoints(
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Dict[str, LineEndpoints]:
    """Return 2D metric endpoints for all straight lawn-plane SoccerNet classes."""
    x_l = -pitch_length / 2.0
    x_r = pitch_length / 2.0
    y_t = -pitch_width / 2.0
    y_b = pitch_width / 2.0

    x_l_pen = x_l + PENALTY_AREA_LENGTH
    x_r_pen = x_r - PENALTY_AREA_LENGTH
    y_pen_t = -PENALTY_AREA_WIDTH / 2.0
    y_pen_b = PENALTY_AREA_WIDTH / 2.0

    x_l_goal = x_l + GOAL_AREA_LENGTH
    x_r_goal = x_r - GOAL_AREA_LENGTH
    y_goal_t = -GOAL_AREA_WIDTH / 2.0
    y_goal_b = GOAL_AREA_WIDTH / 2.0

    return {
        # Pitch boundary + halfway line
        "Side line top": ((x_l, y_t), (x_r, y_t)),
        "Side line bottom": ((x_l, y_b), (x_r, y_b)),
        "Side line left": ((x_l, y_t), (x_l, y_b)),
        "Side line right": ((x_r, y_t), (x_r, y_b)),
        "Middle line": ((0.0, y_t), (0.0, y_b)),

        # Left penalty area
        "Big rect. left top": ((x_l, y_pen_t), (x_l_pen, y_pen_t)),
        "Big rect. left bottom": ((x_l, y_pen_b), (x_l_pen, y_pen_b)),
        "Big rect. left main": ((x_l_pen, y_pen_t), (x_l_pen, y_pen_b)),

        # Right penalty area
        "Big rect. right top": ((x_r_pen, y_pen_t), (x_r, y_pen_t)),
        "Big rect. right bottom": ((x_r_pen, y_pen_b), (x_r, y_pen_b)),
        "Big rect. right main": ((x_r_pen, y_pen_t), (x_r_pen, y_pen_b)),

        # Left goal area
        "Small rect. left top": ((x_l, y_goal_t), (x_l_goal, y_goal_t)),
        "Small rect. left bottom": ((x_l, y_goal_b), (x_l_goal, y_goal_b)),
        "Small rect. left main": ((x_l_goal, y_goal_t), (x_l_goal, y_goal_b)),

        # Right goal area
        "Small rect. right top": ((x_r_goal, y_goal_t), (x_r, y_goal_t)),
        "Small rect. right bottom": ((x_r_goal, y_goal_b), (x_r, y_goal_b)),
        "Small rect. right main": ((x_r_goal, y_goal_t), (x_r_goal, y_goal_b)),
    }


def homogeneous_line_from_endpoints(a: Point2, b: Point2) -> np.ndarray:
    p1 = np.array([a[0], a[1], 1.0], dtype=np.float64)
    p2 = np.array([b[0], b[1], 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    n = np.linalg.norm(line[:2])
    if n < 1e-12:
        raise ValueError("Degenerate line endpoints")
    return line / n


def fit_image_line(points_px: np.ndarray) -> np.ndarray:
    """TLS fit of a homogeneous image line to >=2 pixel points."""
    if len(points_px) < 2:
        raise ValueError("At least two points are needed to fit a line")
    a = np.column_stack([points_px, np.ones(len(points_px))])
    _, _, vh = np.linalg.svd(a)
    line = vh[-1]
    n = np.linalg.norm(line[:2])
    if n < 1e-12:
        raise ValueError("Degenerate image line")
    return line / n


def annotation_points_to_pixels(
    raw_points: Sequence[dict], image_width: int, image_height: int
) -> np.ndarray:
    pts = np.array([[float(p["x"]), float(p["y"])] for p in raw_points], dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 2)

    # SoccerNet annotations are normalized. For convenience, also accept pixels.
    if np.nanmax(np.abs(pts)) <= 1.5:
        pts[:, 0] *= image_width
        pts[:, 1] *= image_height
    return pts


def normalization_transform(points: np.ndarray) -> np.ndarray:
    """Hartley-style similarity transform: centroid -> 0, mean radius -> sqrt(2)."""
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    mean_distance = float(distances.mean())
    scale = math.sqrt(2.0) / mean_distance if mean_distance > 1e-12 else 1.0
    return np.array(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_homography_from_lines(
    correspondences: Sequence[Tuple[np.ndarray, np.ndarray]],
    world_points_for_normalization: np.ndarray,
    image_points_for_normalization: np.ndarray,
) -> np.ndarray:
    """
    Estimate H such that image_point ~ H @ pitch_point.

    A pitch line l maps to image line l' as:
        l' ~ H^{-T} l

    We estimate G = H^{-T} with DLT in the dual projective plane, then recover H.
    The null-space vector is ALWAYS vh[-1] (this intentionally fixes the bug in
    the original baseline supplied with the project).
    """
    if len(correspondences) < 4:
        raise ValueError(f"Need at least 4 lawn-plane line correspondences, got {len(correspondences)}")

    t_pitch = normalization_transform(world_points_for_normalization)
    t_image = normalization_transform(image_points_for_normalization)

    rows: List[List[float]] = []
    for pitch_line, image_line in correspondences:
        # If x_n = T x, then l_n = T^{-T} l.
        src = np.linalg.inv(t_pitch).T @ pitch_line
        dst = np.linalg.inv(t_image).T @ image_line
        src /= max(np.linalg.norm(src[:2]), 1e-12)
        dst /= max(np.linalg.norm(dst[:2]), 1e-12)

        x, y, w = src
        u, v, z = dst

        # Two independent equations for dst ~ G @ src.
        rows.append([0.0, 0.0, 0.0, -z*x, -z*y, -z*w, v*x, v*y, v*w])
        rows.append([z*x, z*y, z*w, 0.0, 0.0, 0.0, -u*x, -u*y, -u*w])

    a = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(a) < 8:
        raise ValueError(
            "Degenerate line configuration: the visible annotated lines do not constrain a unique homography"
        )

    _, _, vh = np.linalg.svd(a)
    g_norm = vh[-1].reshape(3, 3)

    if abs(np.linalg.det(g_norm)) < 1e-12:
        raise ValueError("Estimated dual homography is singular")

    h_norm = np.linalg.inv(g_norm).T
    h = np.linalg.inv(t_image) @ h_norm @ t_pitch

    if abs(h[2, 2]) > 1e-12:
        h /= h[2, 2]
    else:
        h /= np.linalg.norm(h)
    return h


def pack_homography(h: np.ndarray) -> np.ndarray:
    h = h / h[2, 2]
    return np.array(
        [h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1]],
        dtype=np.float64,
    )


def unpack_homography(p: Sequence[float]) -> np.ndarray:
    return np.array(
        [[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]],
        dtype=np.float64,
    )


def line_residuals_px(
    h_pitch_to_image: np.ndarray,
    line_data: Sequence[Tuple[str, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Signed pixel distances from annotation points to projected model lines."""
    try:
        h_inv_t = np.linalg.inv(h_pitch_to_image).T
    except np.linalg.LinAlgError:
        return np.full(sum(len(x[2]) for x in line_data), 1e6, dtype=np.float64)

    residuals: List[float] = []
    for _, pitch_line, annotation_points_px in line_data:
        projected_line = h_inv_t @ pitch_line
        denom = np.linalg.norm(projected_line[:2])
        if denom < 1e-12:
            residuals.extend([1e6] * len(annotation_points_px))
            continue
        projected_line /= denom
        for u, v in annotation_points_px:
            residuals.append(float(projected_line @ np.array([u, v, 1.0])))
    return np.asarray(residuals, dtype=np.float64)


def refine_homography(
    h_initial: np.ndarray,
    line_data: Sequence[Tuple[str, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, bool]:
    """Robust nonlinear refinement. Returns (H, was_refined)."""
    if least_squares is None:
        return h_initial, False

    def residual_fun(params: np.ndarray) -> np.ndarray:
        return line_residuals_px(unpack_homography(params), line_data)

    result = least_squares(
        residual_fun,
        pack_homography(h_initial),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=2000,
    )
    if not result.success:
        return h_initial, False

    h = unpack_homography(result.x)
    h /= h[2, 2]
    return h, True


def transform_points(h: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    hp = np.column_stack([points_xy, np.ones(len(points_xy))])
    out = (h @ hp.T).T
    valid = np.abs(out[:, 2]) > 1e-12
    result = np.full((len(points_xy), 2), np.nan, dtype=np.float64)
    result[valid] = out[valid, :2] / out[valid, 2:3]
    return result


def collect_line_data(
    annotations: dict,
    image_width: int,
    image_height: int,
    pitch_length: float,
    pitch_width: float,
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    List[Tuple[str, np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
]:
    pitch_lines = build_pitch_line_endpoints(pitch_length, pitch_width)
    correspondences: List[Tuple[np.ndarray, np.ndarray]] = []
    line_data: List[Tuple[str, np.ndarray, np.ndarray]] = []
    world_norm_points: List[Point2] = []
    image_norm_points: List[Point2] = []

    for name, raw_points in annotations.items():
        # Goal posts/crossbars are not on Z=0. Circles are curves, not straight lines.
        if name not in pitch_lines:
            continue
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            continue

        pts_px = annotation_points_to_pixels(raw_points, image_width, image_height)
        pts_px = pts_px[np.isfinite(pts_px).all(axis=1)]
        if len(pts_px) < 2:
            continue

        a, b = pitch_lines[name]
        pitch_line = homogeneous_line_from_endpoints(a, b)
        image_line = fit_image_line(pts_px)

        correspondences.append((pitch_line, image_line))
        line_data.append((name, pitch_line, pts_px))
        world_norm_points.extend([a, b])
        image_norm_points.extend([tuple(p) for p in pts_px])

    return (
        correspondences,
        line_data,
        np.asarray(world_norm_points, dtype=np.float64),
        np.asarray(image_norm_points, dtype=np.float64),
    )


def sample_demo_image_points(
    h_image_to_pitch: np.ndarray,
    image_width: int,
    image_height: int,
    pitch_length: float,
    pitch_width: float,
    count: int,
    seed: int,
    pitch_margin_m: float = 5.0,
) -> np.ndarray:
    """Sample random IMAGE pixels whose inverse homography lies inside the pitch."""
    rng = np.random.default_rng(seed)
    selected: List[np.ndarray] = []
    min_separation_px = 0.15 * min(image_width, image_height)

    x_min = -pitch_length / 2.0 + pitch_margin_m
    x_max = pitch_length / 2.0 - pitch_margin_m
    y_min = -pitch_width / 2.0 + pitch_margin_m
    y_max = pitch_width / 2.0 - pitch_margin_m

    for _ in range(200000):
        if len(selected) >= count:
            break
        uv = np.array(
            [rng.uniform(0.0, image_width - 1.0), rng.uniform(0.0, image_height - 1.0)],
            dtype=np.float64,
        )
        xy = transform_points(h_image_to_pitch, uv.reshape(1, 2))[0]
        if not np.isfinite(xy).all():
            continue
        if not (x_min <= xy[0] <= x_max and y_min <= xy[1] <= y_max):
            continue
        if any(np.linalg.norm(uv - prev) < min_separation_px for prev in selected):
            continue
        selected.append(uv)

    if len(selected) < count:
        raise RuntimeError(
            f"Could only sample {len(selected)} valid image points inside the projected pitch; requested {count}"
        )
    return np.asarray(selected, dtype=np.float64)


def parse_cli_points(values: Optional[Sequence[str]]) -> Optional[np.ndarray]:
    if not values:
        return None
    pts: List[Point2] = []
    for value in values:
        try:
            u_str, v_str = value.split(",")
            pts.append((float(u_str), float(v_str)))
        except Exception as exc:
            raise ValueError(f"Invalid point '{value}'. Expected format u,v, e.g. 512,340") from exc
    return np.asarray(pts, dtype=np.float64)


def pitch_drawing_polylines(pitch_length: float, pitch_width: float) -> List[np.ndarray]:
    """Pitch markings for visualization (all in meters on Z=0)."""
    x_l, x_r = -pitch_length / 2.0, pitch_length / 2.0
    y_t, y_b = -pitch_width / 2.0, pitch_width / 2.0
    lines = build_pitch_line_endpoints(pitch_length, pitch_width)

    polylines: List[np.ndarray] = []
    for a, b in lines.values():
        t = np.linspace(0.0, 1.0, 80)
        arr = np.column_stack([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
        polylines.append(arr)

    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    center_circle = np.column_stack(
        [CENTER_CIRCLE_RADIUS * np.cos(theta), CENTER_CIRCLE_RADIUS * np.sin(theta)]
    )
    polylines.append(center_circle)

    # Penalty spots as tiny circles, purely for visualization.
    for cx in (x_l + PENALTY_MARK_DISTANCE, x_r - PENALTY_MARK_DISTANCE):
        r = 0.20
        polylines.append(np.column_stack([cx + r * np.cos(theta), r * np.sin(theta)]))

    return polylines


def draw_top_down_pitch(ax, pitch_length: float, pitch_width: float) -> None:
    for poly in pitch_drawing_polylines(pitch_length, pitch_width):
        ax.plot(poly[:, 0], poly[:, 1], linewidth=1.2)
    ax.set_xlim(-pitch_length / 2.0 - 3.0, pitch_length / 2.0 + 3.0)
    ax.set_ylim(pitch_width / 2.0 + 3.0, -pitch_width / 2.0 - 3.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Top-down pitch / coordinates in meters")
    ax.grid(alpha=0.2)


def create_visualization(
    image_bgr: np.ndarray,
    h_pitch_to_image: np.ndarray,
    line_data: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    pitch_length: float,
    pitch_width: float,
    output_path: Path,
) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = image_bgr.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_img, ax_pitch = axes

    ax_img.imshow(image_rgb)
    ax_img.set_xlim(0, w_img)
    ax_img.set_ylim(h_img, 0)
    ax_img.set_title("Broadcast image: annotations + GT pitch projection + demo points")
    ax_img.axis("off")

    # Manual line annotations actually used for H_GT.
    for name, _, pts_px in line_data:
        ax_img.plot(pts_px[:, 0], pts_px[:, 1], linewidth=2.0, alpha=0.8)

    # Project the canonical pitch model onto the broadcast image.
    for poly_world in pitch_drawing_polylines(pitch_length, pitch_width):
        poly_img = transform_points(h_pitch_to_image, poly_world)
        finite = np.isfinite(poly_img).all(axis=1)
        if finite.any():
            ax_img.plot(poly_img[finite, 0], poly_img[finite, 1], linewidth=1.0, alpha=0.65)

    # Same demo points in image and pitch coordinates.
    for i, (uv, xy) in enumerate(zip(image_points, pitch_points), start=1):
        ax_img.scatter([uv[0]], [uv[1]], s=70, marker="o")
        ax_img.annotate(
            f"P{i}  ({uv[0]:.0f}, {uv[1]:.0f}) px",
            (uv[0], uv[1]),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.7),
        )

    draw_top_down_pitch(ax_pitch, pitch_length, pitch_width)
    for i, xy in enumerate(pitch_points, start=1):
        ax_pitch.scatter([xy[0]], [xy[1]], s=80, marker="o")
        ax_pitch.annotate(
            f"P{i}\n({xy[0]:.2f}, {xy[1]:.2f}) m",
            (xy[0], xy[1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.7),
        )

    fig.suptitle("Annotation-derived H_GT: image pixels → pitch coordinates", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build annotation-derived soccer pitch homography GT and visualize mapped points."
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to image, e.g. 00000.jpg")
    parser.add_argument("--annotation", required=True, type=Path, help="Path to SoccerNet line JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("homography_gt_output"))
    parser.add_argument("--pitch-length", type=float, default=105.0, help="Pitch length in meters")
    parser.add_argument("--pitch-width", type=float, default=68.0, help="Pitch width in meters")
    parser.add_argument("--num-points", type=int, default=2, help="Number of random demo points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for demo points")
    parser.add_argument(
        "--points",
        nargs="*",
        default=None,
        help="Optional image points u,v. If supplied, random points are not used. Example: --points 150,300 600,320",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable optional robust SciPy nonlinear refinement",
    )
    args = parser.parse_args()

    if args.pitch_length <= 0 or args.pitch_width <= 0:
        raise ValueError("Pitch dimensions must be positive")
    if args.num_points < 1:
        raise ValueError("--num-points must be >= 1")

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    image_height, image_width = image_bgr.shape[:2]

    with args.annotation.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    correspondences, line_data, world_norm_pts, image_norm_pts = collect_line_data(
        annotations,
        image_width,
        image_height,
        args.pitch_length,
        args.pitch_width,
    )

    used_classes = [entry[0] for entry in line_data]
    print(f"Image: {image_width} x {image_height}")
    print(f"Usable lawn-plane line classes ({len(used_classes)}):")
    for name in used_classes:
        print(f"  - {name}")

    h_pitch_to_image = estimate_homography_from_lines(
        correspondences, world_norm_pts, image_norm_pts
    )

    refined = False
    if not args.no_refine:
        h_pitch_to_image, refined = refine_homography(h_pitch_to_image, line_data)
        if least_squares is None:
            print("WARNING: SciPy not installed; using DLT without nonlinear refinement.")

    h_image_to_pitch = np.linalg.inv(h_pitch_to_image)
    h_image_to_pitch /= h_image_to_pitch[2, 2]

    residuals = line_residuals_px(h_pitch_to_image, line_data)
    rms_px = float(np.sqrt(np.mean(residuals ** 2)))
    mean_abs_px = float(np.mean(np.abs(residuals)))
    max_abs_px = float(np.max(np.abs(residuals)))

    cli_points = parse_cli_points(args.points)
    if cli_points is not None:
        image_points = cli_points
        if np.any(image_points[:, 0] < 0) or np.any(image_points[:, 0] >= image_width):
            raise ValueError("At least one supplied point has u outside the image")
        if np.any(image_points[:, 1] < 0) or np.any(image_points[:, 1] >= image_height):
            raise ValueError("At least one supplied point has v outside the image")
    else:
        image_points = sample_demo_image_points(
            h_image_to_pitch,
            image_width,
            image_height,
            args.pitch_length,
            args.pitch_width,
            args.num_points,
            args.seed,
        )

    pitch_points = transform_points(h_image_to_pitch, image_points)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "homography_gt.json"
    visualization_path = args.output_dir / "homography_gt_visualization.png"

    output = {
        "image": str(args.image),
        "annotation": str(args.annotation),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "pitch_length_m": float(args.pitch_length),
        "pitch_width_m": float(args.pitch_width),
        "coordinate_system": {
            "origin": "pitch center",
            "x_axis": "negative toward left goal, positive toward right goal",
            "y_axis": "negative toward top touchline, positive toward bottom touchline",
            "units": "meters",
        },
        "used_line_classes": used_classes,
        "refined_with_scipy": bool(refined),
        "quality": {
            "rms_point_to_projected_line_px": rms_px,
            "mean_abs_point_to_projected_line_px": mean_abs_px,
            "max_abs_point_to_projected_line_px": max_abs_px,
        },
        "H_pitch_to_image": h_pitch_to_image.tolist(),
        "H_image_to_pitch": h_image_to_pitch.tolist(),
        "demo_points": [
            {
                "label": f"P{i}",
                "image_px": [float(uv[0]), float(uv[1])],
                "pitch_m": [float(xy[0]), float(xy[1])],
            }
            for i, (uv, xy) in enumerate(zip(image_points, pitch_points), start=1)
        ],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    create_visualization(
        image_bgr=image_bgr,
        h_pitch_to_image=h_pitch_to_image,
        line_data=line_data,
        image_points=image_points,
        pitch_points=pitch_points,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
        output_path=visualization_path,
    )

    print("\nH_pitch_to_image =")
    print(np.array2string(h_pitch_to_image, precision=8, suppress_small=False))
    print("\nH_image_to_pitch =")
    print(np.array2string(h_image_to_pitch, precision=8, suppress_small=False))
    print(
        f"\nQuality: RMS={rms_px:.3f}px, mean|e|={mean_abs_px:.3f}px, max|e|={max_abs_px:.3f}px"
    )
    print(f"Nonlinear refinement: {'yes' if refined else 'no'}")
    print("\nMapped points:")
    for i, (uv, xy) in enumerate(zip(image_points, pitch_points), start=1):
        print(f"  P{i}: image=({uv[0]:.2f}, {uv[1]:.2f}) px -> pitch=({xy[0]:.3f}, {xy[1]:.3f}) m")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {visualization_path}")


if __name__ == "__main__":
    main()
