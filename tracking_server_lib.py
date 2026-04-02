from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import cv2
from ultralytics import YOLO
import json
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import asyncio

frame_bytes = await asyncio.wait_for(ws.receive_bytes(), timeout=5.0)

import sys
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
MAX_PERSONS = 12

POSE_MODEL_PATH = "/home/goikonom/diplw/pose_landmarker_lite.task"

base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# joints pou mas endiaferoun
JOINTS = [0, 11, 12, 15, 16, 23, 24, 25, 26]
VIS_THR = 0.3

# memory / heuristics
track_memory = {}
WARNING_SECS = 4.0
HORIZONTAL_RATIO = 1.15
HORIZONTAL_VEL_THR = 8.0
ARM_ANGLE_WARN = 150.0
MEMORY_TTL_SECS = 12.0


@app.get("/ping")
def ping():
    return {"ok": True}


def pose_landmarks_for_person(img_bgr, x1, y1, x2, y2):
    H, W = img_bgr.shape[:2]

    # clamp sta oria tou frame
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

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=roi_rgb
    )

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

        # apo crop coordinates -> full frame coordinates
        px = x1 + int(lm.x * w)
        py = y1 + int(lm.y * h)

        landmarks.append({
            "id": i,
            "x": int(px),
            "y": int(py),
            "v": float(lm.visibility)
        })

    return landmarks if landmarks else None


def get_lm(pose, lm_id):
    if not pose:
        return None
    for lm in pose:
        if lm.get("id") == lm_id:
            return lm
    return None


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
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    return (bbox_w / bbox_h) > HORIZONTAL_RATIO


def extract_pose_features(pose):
    nose = get_lm(pose, 0)
    l_sh = get_lm(pose, 11)
    r_sh = get_lm(pose, 12)
    l_wr = get_lm(pose, 15)
    r_wr = get_lm(pose, 16)

    nose_pt = point_xy(nose)
    l_sh_pt = point_xy(l_sh)
    r_sh_pt = point_xy(r_sh)
    l_wr_pt = point_xy(l_wr)
    r_wr_pt = point_xy(r_wr)

    left_arm_head_angle = angle_3pts(l_wr_pt, l_sh_pt, nose_pt)
    right_arm_head_angle = angle_3pts(r_wr_pt, r_sh_pt, nose_pt)

    head_below_shoulders = False
    if nose and l_sh and r_sh:
        shoulder_y = (l_sh["y"] + r_sh["y"]) / 2.0
        head_below_shoulders = nose["y"] > shoulder_y

    return {
        "head_below_shoulders": head_below_shoulders,
        "left_arm_head_angle": left_arm_head_angle,
        "right_arm_head_angle": right_arm_head_angle,
    }


def update_track_memory(track_id, pose, x1, y1, x2, y2, now):
    cx, cy = bbox_center(x1, y1, x2, y2)

    if track_id not in track_memory:
        track_memory[track_id] = {
            "last_seen": now,
            "last_bbox": (x1, y1, x2, y2),
            "last_center": (cx, cy),
            "velocity": 0.0,
            "is_horizontal": False,
            "missing_since": None,
            "head_below_shoulders_since": None,
            "arm_angle_left": None,
            "arm_angle_right": None,
        }

    mem = track_memory[track_id]
    vel = compute_velocity(mem.get("last_center"), (cx, cy))
    horizontal = is_horizontal_bbox(x1, y1, x2, y2)

    if pose:
        features = extract_pose_features(pose)
    else:
        features = {
            "head_below_shoulders": False,
            "left_arm_head_angle": None,
            "right_arm_head_angle": None,
        }

    mem["last_seen"] = now
    mem["last_bbox"] = (x1, y1, x2, y2)
    mem["last_center"] = (cx, cy)
    mem["velocity"] = vel if horizontal else 0.0
    mem["is_horizontal"] = horizontal
    mem["arm_angle_left"] = features["left_arm_head_angle"]
    mem["arm_angle_right"] = features["right_arm_head_angle"]
    mem["missing_since"] = None

    if features["head_below_shoulders"]:
        if mem["head_below_shoulders_since"] is None:
            mem["head_below_shoulders_since"] = now
    else:
        mem["head_below_shoulders_since"] = None

    return mem


def update_missing_tracks(seen_ids, now):
    for tid, mem in track_memory.items():
        if tid not in seen_ids:
            if mem["missing_since"] is None:
                mem["missing_since"] = now
        else:
            mem["missing_since"] = None


def get_missing_warning_ids(now):
    out = []
    for tid, mem in track_memory.items():
        if mem["missing_since"] is not None and (now - mem["missing_since"]) > WARNING_SECS:
            out.append(tid)
    return out


def cleanup_old_tracks(now):
    to_delete = []
    for tid, mem in track_memory.items():
        if (now - mem.get("last_seen", now)) > MEMORY_TTL_SECS:
            to_delete.append(tid)

    for tid in to_delete:
        del track_memory[tid]


def detect_state(pose, mem, now):
    if not pose:
        return "warning" if mem.get("is_horizontal") else "unknown"

    head_below_too_long = (
        mem["head_below_shoulders_since"] is not None and
        (now - mem["head_below_shoulders_since"]) > WARNING_SECS
    )

    if head_below_too_long:
        return "danger"

    if mem.get("is_horizontal", False) and mem.get("velocity", 0.0) < HORIZONTAL_VEL_THR:
        return "warning"

    for ang in (mem.get("arm_angle_left"), mem.get("arm_angle_right")):
        if ang is not None and ang > ARM_ANGLE_WARN:
            return "warning"

    return "safe"


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            frame_bytes = await ws.receive_bytes()
            now = time.time()

            # JPEG -> BGR
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                await ws.send_text(json.dumps({
                    "ok": False,
                    "error": "decode_failed"
                }))
                continue

            # YOLO inference
            results = model(img, conf=CONF_THR, verbose=False)

            detections = []

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue  # class 0 = person

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf < CONF_THR:
                    continue

                detections.append([x1, y1, x2, y2, conf])

            if len(detections) > 0:
                detections = np.array(detections, dtype=np.float32)
            else:
                detections = np.empty((0, 5), dtype=np.float32)

            tracks = tracker.update(detections, img.shape[:2], img.shape[:2])

            persons = []
            count_seen = 0
            seen_ids = set()

            for t in tracks:
                if count_seen >= MAX_PERSONS:
                    break

                x1, y1, x2, y2 = map(int, t.tlbr)
                track_id = int(t.track_id)
                seen_ids.add(track_id)

                pose = pose_landmarks_for_person(img, x1, y1, x2, y2)
                mem = update_track_memory(track_id, pose, x1, y1, x2, y2, now)
                state = detect_state(pose, mem, now)

                persons.append({
                    "id": track_id,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "conf": 1.0,
                    "pose": pose,
                    "state": state,
                    "velocity": float(mem["velocity"]),
                    "horizontal": bool(mem["is_horizontal"]),
                    "arm_angle_left": mem["arm_angle_left"],
                    "arm_angle_right": mem["arm_angle_right"],
                })

                count_seen += 1

            update_missing_tracks(seen_ids, now)
            missing_warning_ids = get_missing_warning_ids(now)
            cleanup_old_tracks(now)

            response = {
                "ok": True,
                "person_count": len(persons),
                "persons": persons,
                "missing_warning_ids": missing_warning_ids,
                "server_ts_ms": int(now * 1000)
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
