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
HORIZONTAL_RATIO = 1.15      # bbox_w / bbox_h πάνω από αυτό => περίπου οριζόντιος
HORIZONTAL_VEL_THR = 8.0     # px/frame περίπου, θα το ρυθμίσεις εμπειρικά

POSE_MODEL_PATH = "/home/goikonom/diplw/pose_landmarker_lite.task"

base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# joints pou mas endiaferoun
JOINTS = [0, 11, 12, 15, 16, 23, 24, 25, 26, 27, 28]
VIS_THR = 0.3
#motion_score = 0.1

track_memory = {}

MEMORY_TTL_SECS = 10.0

def point_xy(lm): #dinei tis syntetagmenes se pinaka floats gia na vrw meta gwnia!
    if lm is None:
        return None
    return (float(lm["x"]), float(lm["y"]))

def angle_3pts(a, b, c): #gia na vriskw gwnies xeriwn px
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

def extract_pose_features(pose):
    nose = get_lm(pose, 0)
    l_sh = get_lm(pose, 11)
    r_sh = get_lm(pose, 12)
    l_wr = get_lm(pose, 15)
    r_wr = get_lm(pose, 16)
    l_hip = get_lm(pose, 23)
    r_hip = get_lm(pose, 24)

    nose_pt = point_xy(nose)
    l_sh_pt = point_xy(l_sh)
    r_sh_pt = point_xy(r_sh)
    l_wr_pt = point_xy(l_wr)
    r_wr_pt = point_xy(r_wr)
    l_hip_pt = point_xy(l_hip)
    r_hip_pt = point_xy(r_hip)

    left_arm_head_angle = angle_3pts(l_wr_pt, l_sh_pt, nose_pt)
    right_arm_head_angle = angle_3pts(r_wr_pt, r_sh_pt, nose_pt)

    head_below_shoulders = False
    if nose and l_sh and r_sh:
        shoulder_y = (l_sh["y"] + r_sh["y"]) / 2.0
        head_below_shoulders = nose["y"] > shoulder_y

    hand_above_head = False
    if nose:
        if l_wr and l_wr["y"] < nose["y"]:
            hand_above_head = True
        if r_wr and r_wr["y"] < nose["y"]:
            hand_above_head = True

    legs_visible = any(get_lm(pose, i) is not None for i in [25, 26, 27, 28])
    head_visible = nose is not None

    return {
        "head_visible": head_visible,
        "legs_visible": legs_visible,
        "head_below_shoulders": head_below_shoulders,
        "hand_above_head": hand_above_head,
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

    features = extract_pose_features(pose) if pose else {
        "head_visible": False,
        "legs_visible": False,
        "head_below_shoulders": False,
        "hand_above_head": False,
        "left_arm_head_angle": None,
        "right_arm_head_angle": None,
    }
    
    mem["head_visible"] = features["head_visible"]
    mem["legs_visible"] = features["legs_visible"]
    mem["hand_above_head"] = features["hand_above_head"]
    mem["arm_angle_left"] = features["left_arm_head_angle"]
    mem["arm_angle_right"] = features["right_arm_head_angle"]

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
    
    if features["hand_above_head"]:
        if mem.get("hand_above_head_since") is None:
            mem["hand_above_head_since"] = now
    else:
        mem["hand_above_head_since"] = None

    return mem


def update_missing_ids(seen_ids, now):
    for tid, mem in track_memory.items():
        if tid not in seen_ids:
            if mem["missing_since"] is None:
                mem["missing_since"] = now
        else:
            mem["missing_since"] = None
            

def warn_missing_ids(now):
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




def detect_state(track_id, pose, x1, y1, x2, y2, now):
    mem = track_memory.get(track_id, {})

    velocity = float(mem.get("velocity", 0.0))
    horizontal = bool(mem.get("is_horizontal", False))
    missing_since = mem.get("missing_since")
    head_visible = bool(mem.get("head_visible", False))
    legs_visible = bool(mem.get("legs_visible", False))
    hand_above_head_since = mem.get("hand_above_head_since")
    head_low_since = mem.get("head_below_shoulders_since")

    if missing_since is not None and (now - missing_since) >= WARNING_SECS:
        return "danger"

    if head_visible and legs_visible and not horizontal:
        return "safe"

    if hand_above_head_since is not None and (now - hand_above_head_since) >= 1.5:
        return "danger"

    if horizontal and velocity < 3.0:
        return "danger"

    if head_low_since is not None and (now - head_low_since) >= 1.5:
        return "warning"

    if head_visible and not legs_visible and velocity < 2.0:
        return "warning"

    return "safe"


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            frame_bytes = await ws.receive_bytes()

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

            for t in tracks:
                if count_seen >= MAX_PERSONS:
                    break

                x1, y1, x2, y2 = map(int, t.tlbr)
                track_id = int(t.track_id)

                pose = pose_landmarks_for_person(img, x1, y1, x2, y2)
                
                now = time.time()
                mem = update_track_memory(track_id, pose, x1, y1, x2, y2, now)
                state = detect_state(track_id, pose, x1, y1, x2, y2, now)

                persons.append({
                    "id": track_id,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "conf": conf,
                    "pose": pose,
                    "state": state
                })

                count_seen += 1

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
