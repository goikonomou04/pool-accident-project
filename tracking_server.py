from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import cv2
from ultralytics import YOLO
import json
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
import math
from types import SimpleNamespace

sys.path.append("/home/goikonom/ByteTrack")
from yolox.tracker.byte_tracker import BYTETracker

tracker_args = SimpleNamespace(
    track_thresh=0.5,
    match_thresh=0.8,
    track_buffer=30,
    mot20=False
)
tracker = BYTETracker(tracker_args, frame_rate=5)

app = FastAPI()

model = YOLO("yolov8n.pt")
CONF_THR = 0.35
MAX_PERSONS = 8

WARNING_SECS = 4.0
HORIZONTAL_RATIO = 1.15

POSE_MODEL_PATH = "/home/goikonom/diplw/pose_landmarker_lite.task"
base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

JOINTS = [0, 11, 12, 15, 16, 23, 24, 25, 26, 27, 28]
VIS_THR = 0.3

DANGER_THR = 0.7
WARNING_THR = 0.05
LOW_VEL_THR = 2.0
HEAD_LOW_SECS = 1.5
HAND_UP_SECS = 1.5
SMOOTH_ALPHA = 0.7
SAFE_DECAY = 0.85
MEMORY_TTL_SECS = 10.0

track_memory = {}


def point_xy(lm):
    if lm is None:
        return None
    return (float(lm["x"]), float(lm["y"]))


def angle_3pts(a, b, c):
    if a is None or b is None or c is None:
        return None
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    norm_ba = math.hypot(bax, bay)
    norm_bc = math.hypot(bcx, bcy)
    if norm_ba == 0 or norm_bc == 0:
        return None
    cosang = (bax * bcx + bay * bcy) / (norm_ba * norm_bc)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def bbox_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_velocity(prev_center, curr_center):
    if prev_center is None or curr_center is None:
        return 0.0
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    return math.hypot(dx, dy)


def is_horizontal_bbox(x1, y1, x2, y2):
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return (w / h) > HORIZONTAL_RATIO


def get_lm(pose, lm_id):
    if not pose:
        return None
    for lm in pose:
        if lm.get("id") == lm_id:
            return lm
    return None


def pose_landmarks_for_person(img_bgr, x1, y1, x2, y2):
    H, W = img_bgr.shape[:2]
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W))
    y2 = max(0, min(int(y2), H))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
    result = pose_landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    h, w = roi.shape[:2]
    pose = result.pose_landmarks[0]
    landmarks = []
    for i in JOINTS:
        if i >= len(pose):
            continue
        lm = pose[i]
        if float(lm.visibility) < VIS_THR:
            continue
        px = x1 + int(lm.x * w)
        py = y1 + int(lm.y * h)
        landmarks.append({
            "id": i,
            "x": int(px),
            "y": int(py),
            "v": float(lm.visibility)
        })
    return landmarks if landmarks else None


def extract_features(pose, x1, y1, x2, y2):
    nose = get_lm(pose, 0)
    l_sh = get_lm(pose, 11)
    r_sh = get_lm(pose, 12)
    l_wr = get_lm(pose, 15)
    r_wr = get_lm(pose, 16)

    head_visible = nose is not None
    legs_visible = any(get_lm(pose, i) is not None for i in [25, 26, 27, 28])

    head_below_shoulders = False
    if nose is not None and (l_sh is not None or r_sh is not None):
        sh_y_vals = [s["y"] for s in (l_sh, r_sh) if s is not None]
        if sh_y_vals:
            avg_sh_y = sum(sh_y_vals) / len(sh_y_vals)
            if nose["y"] > avg_sh_y:
                head_below_shoulders = True

    hand_above_head = False
    if nose is not None:
        for wr in (l_wr, r_wr):
            if wr is not None and wr["y"] < nose["y"]:
                hand_above_head = True
                break

    return {
        "head_visible": head_visible,
        "legs_visible": legs_visible,
        "head_below_shoulders": head_below_shoulders,
        "hand_above_head": hand_above_head,
    }


def update_track_memory(track_id, pose, x1, y1, x2, y2, now):
    mem = track_memory.get(track_id)
    if mem is None:
        mem = {
            "first_seen": now,
            "last_seen": now,
            "prev_center": None,
            "velocity": 0.0,
            "is_horizontal": False,
            "head_visible": False,
            "legs_visible": False,
            "head_below_shoulders_since": None,
            "hand_above_head_since": None,
            "missing_since": None,
            "risk_score": 0.0,
        }
        track_memory[track_id] = mem

    curr_center = bbox_center(x1, y1, x2, y2)
    velocity = compute_velocity(mem.get("prev_center"), curr_center)
    mem["prev_center"] = curr_center
    mem["velocity"] = velocity
    mem["last_seen"] = now
    mem["missing_since"] = None
    mem["is_horizontal"] = is_horizontal_bbox(x1, y1, x2, y2)

    features = extract_features(pose, x1, y1, x2, y2)
    mem["head_visible"] = features["head_visible"]
    mem["legs_visible"] = features["legs_visible"]

    if features["head_below_shoulders"]:
        if mem.get("head_below_shoulders_since") is None:
            mem["head_below_shoulders_since"] = now
    else:
        mem["head_below_shoulders_since"] = None

    if features["hand_above_head"]:
        if mem.get("hand_above_head_since") is None:
            mem["hand_above_head_since"] = now
    else:
        mem["hand_above_head_since"] = None

    return mem


def update_missing_ids(seen_ids, now):
    for tid, mem in track_memory.items():
        if tid not in seen_ids:
            if mem.get("missing_since") is None:
                mem["missing_since"] = now


def cleanup_old_tracks(now):
    to_delete = []
    for tid, mem in track_memory.items():
        if (now - mem.get("last_seen", now)) > MEMORY_TTL_SECS:
            to_delete.append(tid)
    for tid in to_delete:
        del track_memory[tid]


def detect_state(track_id, pose, x1, y1, x2, y2, now):
    mem = track_memory.get(track_id, {})
    prev_risk = float(mem.get("risk_score", 0.0))
    velocity = float(mem.get("velocity", 0.0))
    horizontal = bool(mem.get("is_horizontal", False))
    missing_since = mem.get("missing_since")
    head_visible = bool(mem.get("head_visible", False))
    legs_visible = bool(mem.get("legs_visible", False))
    hand_above_head_since = mem.get("hand_above_head_since")
    head_low_since = mem.get("head_below_shoulders_since")

    if missing_since is not None and (now - missing_since) >= WARNING_SECS:
        smoothed_risk = 1.0
        mem["risk_score"] = smoothed_risk
        return "danger", smoothed_risk

    if not pose:
        instant_risk = 0.35
        smoothed_risk = SMOOTH_ALPHA * prev_risk + (1.0 - SMOOTH_ALPHA) * instant_risk
        smoothed_risk = min(max(smoothed_risk, 0.0), 1.0)
        mem["risk_score"] = smoothed_risk
        if smoothed_risk >= DANGER_THR:
            return "danger", smoothed_risk
        elif smoothed_risk >= WARNING_THR:
            return "warning", smoothed_risk
        else:
            return "safe", smoothed_risk

    instant_risk = 0.0

    if horizontal:
        instant_risk += 0.30

    if velocity < LOW_VEL_THR:
        instant_risk += 0.30

    if head_visible and not legs_visible:
        instant_risk += 0.20

    if head_low_since is not None and (now - head_low_since) >= HEAD_LOW_SECS:
        instant_risk += 0.30

    if hand_above_head_since is not None and (now - hand_above_head_since) >= HAND_UP_SECS:
        instant_risk += 0.30

    instant_risk = min(max(instant_risk, 0.0), 1.0)

    clearly_safe = (
        head_visible
        and legs_visible
        and (not horizontal)
        and (velocity >= LOW_VEL_THR)
    )

    if clearly_safe:
        smoothed_risk = prev_risk * SAFE_DECAY
    else:
        smoothed_risk = SMOOTH_ALPHA * prev_risk + (1.0 - SMOOTH_ALPHA) * instant_risk

    smoothed_risk = min(max(smoothed_risk, 0.0), 1.0)
    mem["risk_score"] = smoothed_risk

    if smoothed_risk >= DANGER_THR:
        return "danger", smoothed_risk
    elif smoothed_risk >= WARNING_THR:
        return "warning", smoothed_risk
    else:
        return "safe", smoothed_risk


@app.get("/ping")
def ping():
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            frame_bytes = await ws.receive_bytes()

            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                await ws.send_text(json.dumps({
                    "ok": False,
                    "error": "decode_failed"
                }))
                continue

            results = model(img, conf=CONF_THR, verbose=False)

            detections = []
            det_confs = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf < CONF_THR:
                    continue
                detections.append([bx1, by1, bx2, by2, conf])
                det_confs.append(conf)

            if len(detections) > 0:
                detections = np.array(detections, dtype=np.float32)
            else:
                detections = np.empty((0, 5), dtype=np.float32)

            tracks = tracker.update(detections, img.shape[:2], img.shape[:2])

            now = time.time()
            persons = []
            seen_ids = set()
            count_seen = 0

            for t in tracks:
                if count_seen >= MAX_PERSONS:
                    break
                tx1, ty1, tx2, ty2 = map(int, t.tlbr)
                track_id = int(t.track_id)
                seen_ids.add(track_id)

                pose = pose_landmarks_for_person(img, tx1, ty1, tx2, ty2)
                update_track_memory(track_id, pose, tx1, ty1, tx2, ty2, now)
                state, risk = detect_state(track_id, pose, tx1, ty1, tx2, ty2, now)

                track_conf = float(getattr(t, "score", 0.0))

                persons.append({
                    "id": track_id,
                    "x1": int(tx1),
                    "y1": int(ty1),
                    "x2": int(tx2),
                    "y2": int(ty2),
                    "conf": track_conf,
                    "pose": pose,
                    "state": state,
                    "risk": round(float(risk), 3),
                })
                count_seen += 1

            update_missing_ids(seen_ids, now)
            cleanup_old_tracks(now)

            response = {
                "ok": True,
                "person_count": len(persons),
                "persons": persons,
                "server_ts_ms": int(time.time() * 1000)
            }
            await ws.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("Client disconnected", flush=True)
    except Exception as e:
        print("SERVER ERROR:", repr(e), flush=True)
        try:
            await ws.send_text(json.dumps({
                "ok": False,
                "error": str(e)
            }))
        except Exception:
            pass
