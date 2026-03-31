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

track_memory = {}

WARNING_SECS = 4.0
HORIZONTAL_RATIO = 1.15      # bbox_w / bbox_h πάνω από αυτό => περίπου οριζόντιος
HORIZONTAL_VEL_THR = 8.0     # px/frame περίπου, θα το ρυθμίσεις εμπειρικά

model = YOLO("yolov8n.pt")
CONF_THR = 0.35
MAX_PERSONS = 5

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
motion_score=1 #thelei tracking, na to ftiaksw!

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


def detect_state(pose, x1, y1, x2, y2):
    """
    Aplo heuristic gia demo:
    - danger: den fainontai ka8olou shoulders/hips h to nose einai xamila mesa sto bbox
    - warning: leipoun kapoia vasika joints
    - safe: alliws
    IDANIKA apla risk score px
    % head low
    % flailing
    % abnormal angle
    % motionless/ disappearence
    % emotion
    """

    if not pose:
        return "warning"

    nose = get_lm(pose, 0)
    l_sh = get_lm(pose, 11)
    r_sh = get_lm(pose, 12)
    l_hip = get_lm(pose, 23)
    r_hip = get_lm(pose, 24)

    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    horizontal = bbox_w > bbox_h
    
    
    
    #einai ektos nerou?
    head_visible = get_lm(pose, 0) is not None
    legs_visible = any(get_lm(pose, i) is not None for i in [25, 26, 27, 28])
    
    # full body & vertical -> probably out of water / safe
    if head_visible and legs_visible and not horizontal:
        return "safe"

    # full body & horizontal -> need motion check
    if head_visible and legs_visible and horizontal:
        if motion_score > 0.03:
            return "safe"      # likely swimming
        else:
            return "danger"    # floating / still / suspicious

    # head visible but lower body missing -> likely in water
    if head_visible and not legs_visible:
        if motion_score < 0.01:
            return "warning"
        return "safe"

    return "warning"

    # an leipoun kai oi 2 wμοi ή και oi 2 γοφοι -> πιο ύποπτο
    if (l_sh is None and r_sh is None) or (l_hip is None and r_hip is None):
        return "danger"

    # an leipei nose -> warning
    if nose is None:
        return "warning"

    # ypsilo suspicion an to kefali einai poly xamila sto bbox
    # (poly aprosfo, alla xrisimo san placeholder heuristic)
    rel_head_y = (nose["y"] - y1) / bbox_h
    if rel_head_y > 0.55:
        return "danger"

    # an leipei enas omos h enas gofos
    if l_sh is None or r_sh is None or l_hip is None or r_hip is None:
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
                state = detect_state(pose, x1, y1, x2, y2)

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
