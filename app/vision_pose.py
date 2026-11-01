"""
Human movement / form analysis from video via 2-D pose estimation.

Uses a YOLOv8-pose model to find the 17 COCO body keypoints on the most
prominent person in each frame, then turns that skeleton into coaching metrics:
joint angles, rep count, squat depth, tempo, and left/right symmetry — plus an
annotated video with the skeleton drawn on.

The showcase movement is the back/air squat (the knee angle is the primary
signal), but the keypoints and angles are generic to any full-body movement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import uniform_filter1d

# COCO keypoint indices (YOLOv8-pose order)
NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

# Bones to draw (pairs of keypoint indices)
SKELETON = [
    (L_SH, R_SH), (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP),
    (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
    (L_HIP, L_KNEE), (L_KNEE, L_ANK), (R_HIP, R_KNEE), (R_KNEE, R_ANK),
]
KP_CONF = 0.3  # minimum per-keypoint confidence to trust a point


@dataclass
class FormResult:
    fps: float
    duration_s: float
    frames_with_person: int
    reps: int
    best_depth_deg: float          # smallest knee angle reached (lower = deeper)
    avg_tempo_s: float             # seconds per rep
    symmetry_pct: float            # 0..100, 100 = perfectly even L/R
    depth_rating: str              # "deep" / "parallel" / "partial" / "shallow"
    knee_series: list[float] = field(default_factory=list)   # smoothed knee angle per frame
    rep_depths: list[float] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)          # {level, text}
    annotated_video: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fps": round(self.fps, 1),
            "duration_s": round(self.duration_s, 2),
            "frames_with_person": self.frames_with_person,
            "reps": self.reps,
            "best_depth_deg": round(self.best_depth_deg, 0),
            "avg_tempo_s": round(self.avg_tempo_s, 2),
            "symmetry_pct": round(self.symmetry_pct, 0),
            "depth_rating": self.depth_rating,
            "knee_series": [round(v, 1) for v in self.knee_series],
            "rep_depths": [round(v, 0) for v in self.rep_depths],
            "flags": self.flags,
            "notes": self.notes,
        }


def _angle(a, b, c) -> float:
    """Angle ABC in degrees (at vertex b), or nan if any point is missing."""
    if a is None or b is None or c is None:
        return float("nan")
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return float("nan")
    cos = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) < 2:
        return x
    return uniform_filter1d(x.astype(float), size=min(win, len(x)), mode="nearest")


def _depth_rating(min_angle: float) -> str:
    if min_angle <= 95:
        return "deep"
    if min_angle <= 110:
        return "parallel"
    if min_angle <= 135:
        return "partial"
    return "shallow"


def analyze_form(input_path: str, output_path: str,
                 model_name: str = "yolov8n-pose.pt",
                 max_frames: Optional[int] = 900, progress=None) -> FormResult:
    from ultralytics import YOLO
    model = YOLO(model_name)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=7,
        macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "24"],
    )

    frames_rgb: list[np.ndarray] = []
    kps_per_frame: list[Optional[np.ndarray]] = []  # (17,3) or None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, verbose=False)[0]
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        chosen = None
        if res.keypoints is not None and res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            main = int(np.argmax(areas))  # most prominent person
            chosen = res.keypoints.data.cpu().numpy()[main]  # (17,3)
        kps_per_frame.append(chosen)

        idx += 1
        if progress and total:
            progress(min(0.6, 0.6 * idx / total))
        if max_frames and idx >= max_frames:
            break
    cap.release()

    seen = [k for k in kps_per_frame if k is not None]
    if len(seen) < 5:
        writer.close()
        raise ValueError(
            "Couldn't find a person clearly enough to analyze. Film the whole "
            "body in frame, side-on, with good lighting."
        )

    def pt(kp, i):
        if kp is None or kp[i, 2] < KP_CONF:
            return None
        return kp[i, :2].astype(float)

    # ---- per-frame joint angles -------------------------------------------
    knee_l, knee_r, hip_l, hip_r = [], [], [], []
    for kp in kps_per_frame:
        knee_l.append(_angle(pt(kp, L_HIP), pt(kp, L_KNEE), pt(kp, L_ANK)))
        knee_r.append(_angle(pt(kp, R_HIP), pt(kp, R_KNEE), pt(kp, R_ANK)))
        hip_l.append(_angle(pt(kp, L_SH), pt(kp, L_HIP), pt(kp, L_KNEE)))
        hip_r.append(_angle(pt(kp, R_SH), pt(kp, R_HIP), pt(kp, R_KNEE)))
    knee_l = np.array(knee_l); knee_r = np.array(knee_r)

    # mean knee angle where available (primary squat signal)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        knee = np.nanmean(np.vstack([knee_l, knee_r]), axis=0)
    # fill gaps by interpolation over frame index
    good = ~np.isnan(knee)
    if good.sum() < 5:
        writer.close()
        raise ValueError("Legs weren't visible enough to measure knee angle.")
    xi = np.arange(len(knee))
    knee = np.interp(xi, xi[good], knee[good])
    win = max(1, int(round(fps / 6)))
    knee_s = _smooth(knee, win)

    # ---- rep detection via the knee-angle cycle ---------------------------
    # Thresholds adapt to this clip's own range of motion: a rep is a dip into
    # the bottom third of the observed range and a recovery past the mid-point.
    lo = float(np.percentile(knee_s, 5))
    rng = float(np.percentile(knee_s, 95) - lo)
    if rng < 25:                          # person isn't really squatting
        down_th, up_th = 125.0, 155.0
        reps, rep_depths = 0, []
    else:
        down_th = lo + 0.30 * rng
        up_th = lo + 0.62 * rng
        reps, rep_depths = _count_reps(knee_s, down_th, up_th)
    best_depth = float(np.min(knee_s))
    avg_tempo = (len(knee_s) / fps) / reps if reps else 0.0

    # ---- left/right symmetry ----------------------------------------------
    both = ~np.isnan(knee_l) & ~np.isnan(knee_r)
    if both.sum() > 3:
        diff = np.abs(knee_l[both] - knee_r[both])
        symmetry = float(max(0.0, 100.0 - np.mean(diff) * 2.0))  # 1deg diff ~= -2%
    else:
        symmetry = float("nan")

    flags = _build_flags(best_depth, symmetry, avg_tempo, reps)

    # ---- annotate ---------------------------------------------------------
    phase_of = _phase_track(knee_s)
    rep_running = _running_rep_count(knee_s, down_th, up_th)
    for i, rgb in enumerate(frames_rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _draw_skeleton(bgr, kps_per_frame[i])
        _draw_hud(bgr, knee_s[i] if i < len(knee_s) else float("nan"),
                  rep_running[i] if i < len(rep_running) else 0,
                  phase_of[i] if i < len(phase_of) else "")
        writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if progress and frames_rgb:
            progress(0.6 + 0.4 * (i + 1) / len(frames_rgb))
    writer.close()

    notes = [
        "Pose from a YOLOv8-pose model (17 body keypoints). Angles are 2-D, so "
        "a clean side-on view gives the most accurate depth and knee readings.",
    ]
    return FormResult(
        fps=fps, duration_s=len(frames_rgb) / fps if fps else 0.0,
        frames_with_person=len(seen), reps=reps,
        best_depth_deg=best_depth, avg_tempo_s=avg_tempo,
        symmetry_pct=symmetry if not np.isnan(symmetry) else 0.0,
        depth_rating=_depth_rating(best_depth),
        knee_series=[float(v) for v in knee_s], rep_depths=rep_depths,
        flags=flags, annotated_video=output_path, notes=notes,
    )


def _count_reps(knee: np.ndarray, down: float = 125.0, up: float = 155.0):
    """A rep = knee angle drops below `down`, then recovers above `up`."""
    reps, depths = 0, []
    state = "up"
    cur_min = 180.0
    for a in knee:
        if state == "up" and a < down:
            state = "down"; cur_min = a
        elif state == "down":
            cur_min = min(cur_min, a)
            if a > up:
                reps += 1; depths.append(float(cur_min)); state = "up"
    return reps, depths


def _running_rep_count(knee: np.ndarray, down: float = 125.0, up: float = 155.0):
    out, state, c = [], "up", 0
    for a in knee:
        if state == "up" and a < down:
            state = "down"
        elif state == "down" and a > up:
            c += 1; state = "up"
        out.append(c)
    return out


def _phase_track(knee: np.ndarray):
    """Label each frame descent / ascent / hold from the knee-angle slope."""
    out = []
    d = np.gradient(knee)
    for g in d:
        if g < -0.6:
            out.append("down")
        elif g > 0.6:
            out.append("up")
        else:
            out.append("hold")
    return out


def _build_flags(best_depth, symmetry, tempo, reps):
    flags = []
    rating = _depth_rating(best_depth)
    if rating in ("deep", "parallel"):
        flags.append({"level": "good", "text": f"Good depth — reached {rating} ({best_depth:.0f}° knee)."})
    elif rating == "partial":
        flags.append({"level": "warn", "text": f"Partial depth ({best_depth:.0f}° knee) — aim to break parallel."})
    else:
        flags.append({"level": "bad", "text": f"Shallow squat ({best_depth:.0f}° knee) — sit lower."})

    if not np.isnan(symmetry):
        if symmetry >= 88:
            flags.append({"level": "good", "text": f"Even left/right drive ({symmetry:.0f}% symmetry)."})
        else:
            flags.append({"level": "warn", "text": f"Uneven left/right ({symmetry:.0f}% symmetry) — check for a weight shift."})

    if reps and tempo < 1.2:
        flags.append({"level": "warn", "text": f"Fast tempo (~{tempo:.1f}s/rep) — controlled reps build more strength."})
    elif reps:
        flags.append({"level": "good", "text": f"Controlled tempo (~{tempo:.1f}s/rep)."})
    return flags


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
_ACCENT = (235, 180, 70)   # BGR amber
_BONE = (210, 170, 90)
_JOINT = (250, 250, 250)


def _draw_skeleton(bgr, kp):
    if kp is None:
        return
    for a, b in SKELETON:
        if kp[a, 2] >= KP_CONF and kp[b, 2] >= KP_CONF:
            pa = tuple(kp[a, :2].astype(int)); pb = tuple(kp[b, :2].astype(int))
            cv2.line(bgr, pa, pb, _BONE, 3, cv2.LINE_AA)
    for i in range(len(kp)):
        if kp[i, 2] >= KP_CONF:
            cv2.circle(bgr, tuple(kp[i, :2].astype(int)), 4, _JOINT, -1, cv2.LINE_AA)


def _draw_hud(bgr, knee_angle, rep_count, phase):
    h, w = bgr.shape[:2]
    panel = bgr.copy()
    cv2.rectangle(panel, (0, 0), (w, 64), (24, 20, 14), -1)
    cv2.addWeighted(panel, 0.5, bgr, 0.5, 0, bgr)
    ka = "--" if (knee_angle is None or np.isnan(knee_angle)) else f"{knee_angle:.0f}"
    cv2.putText(bgr, f"Reps: {rep_count}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, f"Knee: {ka}deg   {phase}", (14, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, _ACCENT, 2, cv2.LINE_AA)
