"""Smoke test: run the pose pipeline on the downloaded squat clips."""
import json, time
from app.vision_pose import analyze_form

CLIPS = [
    ("data/test_clips/squat1.mp4", "data/outputs/squat1_form.mp4"),
    ("data/test_clips/squat2.mp4", "data/outputs/squat2_form.mp4"),
]
for src, out in CLIPS:
    print(f"\n=== {src} ===", flush=True)
    t0 = time.time()
    try:
        r = analyze_form(src, out, max_frames=600).to_dict()
        r_small = {k: v for k, v in r.items() if k not in ("knee_series",)}
        r_small["knee_series_len"] = len(r["knee_series"])
        print(json.dumps(r_small, indent=2))
        print(f"elapsed {time.time()-t0:.1f}s -> {out}")
    except Exception as e:
        print("ERROR:", e)
print("\nALL_DONE")
