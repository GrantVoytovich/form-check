# 🏋️ Form Check

Upload a video of a squat (or any full-body movement) and get back a coached
breakdown — **reps, depth, tempo, and left/right symmetry** — plus a video with
the skeleton drawn on. It runs on **2-D pose estimation**: no wearables, markers,
or calibration.

## What it measures

| Metric | How |
| --- | --- |
| **Reps** | Counted from the knee-angle cycle (drop below ~125°, recover above ~155°). |
| **Depth** | Smallest knee angle reached — rated deep / parallel / partial / shallow. |
| **Tempo** | Seconds per rep. |
| **Symmetry** | Left-vs-right knee-angle difference across the set. |
| **Form cues** | Depth, symmetry, and tempo flags. |

It also charts the **knee angle through the set** (each dip is a rep) and writes an
**annotated MP4** with the skeleton and a live rep/angle HUD.

## How it works

1. **YOLOv8-pose** locates 17 body keypoints on the most prominent person each frame.
2. Joint angles come from the hip–knee–ankle (and shoulder–hip–knee) geometry.
3. The knee-angle signal is smoothed; reps, depth, tempo, and symmetry are derived from it.

## Setup

```powershell
cd C:\Users\grant\form-check
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

First run downloads the `yolov8n-pose.pt` weights (~6 MB) automatically.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8010
```

Open <http://localhost:8010>, upload a clip, and wait for the report.

## Tips

- **Film side-on**, whole body in frame, one person, decent lighting.
- The showcase movement is the squat; the keypoints/angles generalize to other
  movements (the rep + depth logic is tuned for the knee).

## Next steps

- Per-exercise profiles (deadlift hip hinge, lunge, push-up elbow angle).
- Knee-valgus / bar-path overlays from a front view.
- A rep-by-rep table with depth and tempo per rep.
