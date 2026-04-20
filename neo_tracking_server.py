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

# ── Χρονικά κατώφλια ──────────────────────────────────────────────
WARNING_SECS          = 4.0   # πόσο να λείπει κάποιος για να θεωρηθεί danger
HAND_RAISE_SECS       = 1.5   # χέρι πάνω από κεφάλι για X δευτερόλεπτα => danger
HEAD_LOW_SECS         = 1.5   # κεφάλι κάτω από ώμους για X δευτερόλεπτα => warning
HORIZONTAL_SECS       = 1.0   # οριζόντιο bbox για X δευτερόλεπτα => danger

# ── Γεωμετρικά κατώφλια ───────────────────────────────────────────
HORIZONTAL_RATIO      = 1.15  # bbox_w / bbox_h > αυτό => πιθανώς οριζόντιος
HORIZONTAL_VEL_THR    = 8.0   # px/frame: κάτω από αυτό + horizontal => πτώση (όχι roll)

# ── Γωνίες βραχιόνων ──────────────────────────────────────────────
# Γωνία (wrist → shoulder → nose): μικρή γωνία = χέρι κοντά στο κεφάλι / σήμα κινδύνου
ARM_DISTRESS_ANGLE    = 45.0  # μοίρες: κάτω από αυτό ΚΑΙ οι δύο βραχίονες => danger
# Γωνία elbow-shoulder-hip (torso lean): μεγάλη γωνία = ο κορμός είναι ξαπλωτός
TORSO_LEAN_ANGLE      = 120.0 # μοίρες: πάνω από αυτό => ο κορμός τείνει οριζόντια

POSE_MODEL_PATH = "/home/goikonom/diplw/pose_landmarker_lite.task"

base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Joints που χρησιμοποιούμε:
# 0=nose, 11=l_shoulder, 12=r_shoulder, 13=l_elbow, 14=r_elbow
# 15=l_wrist, 16=r_wrist, 23=l_hip, 24=r_hip, 25=l_knee, 26=r_knee, 27=l_ankle, 28=r_ankle
JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
VIS_THR = 0.3

track_memory = {}
MEMORY_TTL_SECS = 10.0


# ══════════════════════════════════════════════════════════════════
# Βοηθητικές γεωμετρικές συναρτήσεις
# ══════════════════════════════════════════════════════════════════

def point_xy(lm):
    """Επιστρέφει (x, y) float tuple από landmark dict."""
    if lm is None:
        return None
    return (float(lm["x"]), float(lm["y"]))


def angle_3pts(a, b, c):
    """
    Γωνία στο σημείο b, σχηματιζόμενη από τα ευθύγραμμα τμήματα b→a και b→c.
    Επιστρέφει μοίρες [0, 180] ή None αν λείπει κάποιο σημείο.
    """
    if a is None or b is None or c is None:
        return None
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
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
    return math.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])


def is_horizontal_bbox(x1, y1, x2, y2):
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return (w / h) > HORIZONTAL_RATIO


# ══════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/ping")
def ping():
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════
# Pose detection
# ══════════════════════════════════════════════════════════════════

def pose_landmarks_for_person(img_bgr, x1, y1, x2, y2):
    """
    Τρέχει MediaPipe PoseLandmarker στο crop του ατόμου.
    Επιστρέφει λίστα από {"id", "x", "y", "v"} σε full-frame συντεταγμένες,
    ή None αν δεν βρεθεί pose.
    """
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
        landmarks.append({
            "id": i,
            "x": x1 + int(lm.x * w),
            "y": y1 + int(lm.y * h),
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


# ══════════════════════════════════════════════════════════════════
# Feature extraction από pose landmarks
# ══════════════════════════════════════════════════════════════════

def extract_pose_features(pose):
    """
    Εξάγει boolean και angular features από τα pose landmarks.

    Features:
    - head_visible        : το nose landmark είναι ορατό
    - legs_visible        : τουλάχιστον ένα από knee/ankle είναι ορατό
    - head_below_shoulders: το κεφάλι είναι χαμηλότερα από τους ώμους (πτώση)
    - hand_above_head     : ένα τουλάχιστον χέρι είναι πάνω από τη μύτη (σήμα κινδύνου)
    - arm_angle_left/right: γωνία wrist→shoulder→nose (μικρή = χέρι κοντά στο κεφάλι)
    - torso_angle_left/right: γωνία shoulder→hip→knee (μεγάλη = ξαπλωτός κορμός)
    - both_arms_low_angle : True αν ΚΑΙ οι δύο γωνίες βραχιόνων < ARM_DISTRESS_ANGLE
    - torso_horizontal    : True αν τουλάχιστον μία γωνία κορμού > TORSO_LEAN_ANGLE
    """
    nose  = get_lm(pose, 0)
    l_sh  = get_lm(pose, 11)
    r_sh  = get_lm(pose, 12)
    l_el  = get_lm(pose, 13)
    r_el  = get_lm(pose, 14)
    l_wr  = get_lm(pose, 15)
    r_wr  = get_lm(pose, 16)
    l_hip = get_lm(pose, 23)
    r_hip = get_lm(pose, 24)
    l_kn  = get_lm(pose, 25)
    r_kn  = get_lm(pose, 26)

    nose_pt  = point_xy(nose)
    l_sh_pt  = point_xy(l_sh)
    r_sh_pt  = point_xy(r_sh)
    l_wr_pt  = point_xy(l_wr)
    r_wr_pt  = point_xy(r_wr)
    l_hip_pt = point_xy(l_hip)
    r_hip_pt = point_xy(r_hip)
    l_kn_pt  = point_xy(l_kn)
    r_kn_pt  = point_xy(r_kn)

    # ── Γωνίες βραχιόνων (wrist → shoulder → nose) ────────────────
    # Μικρή γωνία = ο καρπός είναι ευθυγραμμισμένος με μύτη, π.χ. χέρι ψηλά
    left_arm_angle  = angle_3pts(l_wr_pt, l_sh_pt, nose_pt)
    right_arm_angle = angle_3pts(r_wr_pt, r_sh_pt, nose_pt)

    # ── Γωνίες κορμού (shoulder → hip → knee) ────────────────────
    # Μεγάλη γωνία ≈ ο κορμός είναι πιο οριζόντιος (π.χ. ξαπλωτός)
    left_torso_angle  = angle_3pts(l_sh_pt, l_hip_pt, l_kn_pt)
    right_torso_angle = angle_3pts(r_sh_pt, r_hip_pt, r_kn_pt)

    # ── Boolean features ───────────────────────────────────────────
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

    # Και οι δύο γωνίες βραχιόνων κάτω από κατώφλι => σήμα κινδύνου (χέρια κοντά στο κεφάλι)
    both_arms_low_angle = (
        left_arm_angle  is not None and left_arm_angle  < ARM_DISTRESS_ANGLE and
        right_arm_angle is not None and right_arm_angle < ARM_DISTRESS_ANGLE
    )

    # Τουλάχιστον μία γωνία κορμού πάνω από κατώφλι => κορμός οριζόντιος
    torso_horizontal = (
        (left_torso_angle  is not None and left_torso_angle  > TORSO_LEAN_ANGLE) or
        (right_torso_angle is not None and right_torso_angle > TORSO_LEAN_ANGLE)
    )

    return {
        "head_visible":         head_visible,
        "legs_visible":         legs_visible,
        "head_below_shoulders": head_below_shoulders,
        "hand_above_head":      hand_above_head,
        "arm_angle_left":       left_arm_angle,
        "arm_angle_right":      right_arm_angle,
        "torso_angle_left":     left_torso_angle,
        "torso_angle_right":    right_torso_angle,
        "both_arms_low_angle":  both_arms_low_angle,
        "torso_horizontal":     torso_horizontal,
    }


# ══════════════════════════════════════════════════════════════════
# Track memory management
# ══════════════════════════════════════════════════════════════════

def _new_track_entry(now, cx, cy, x1, y1, x2, y2):
    """Δημιουργεί νέο entry στο track_memory με όλα τα fields αρχικοποιημένα."""
    return {
        "last_seen":                now,
        "last_bbox":                (x1, y1, x2, y2),
        "last_center":              (cx, cy),
        "velocity":                 0.0,
        "is_horizontal":            False,
        "missing_since":            None,
        # Temporal trackers (αρχικοποιημένα ρητά)
        "head_below_shoulders_since": None,
        "hand_above_head_since":      None,
        "horizontal_since":           None,
        # Pose features (τελευταία γνωστή τιμή)
        "head_visible":             False,
        "legs_visible":             False,
        "hand_above_head":          False,
        "arm_angle_left":           None,
        "arm_angle_right":          None,
        "torso_angle_left":         None,
        "torso_angle_right":        None,
        "both_arms_low_angle":      False,
        "torso_horizontal":         False,
    }


def update_track_memory(track_id, pose, x1, y1, x2, y2, now):
    cx, cy = bbox_center(x1, y1, x2, y2)

    if track_id not in track_memory:
        track_memory[track_id] = _new_track_entry(now, cx, cy, x1, y1, x2, y2)

    mem = track_memory[track_id]
    vel        = compute_velocity(mem["last_center"], (cx, cy))
    horizontal = is_horizontal_bbox(x1, y1, x2, y2)

    # Pose features (fallback αν δεν υπάρχει pose)
    if pose:
        features = extract_pose_features(pose)
    else:
        features = {
            "head_visible":         False,
            "legs_visible":         False,
            "head_below_shoulders": False,
            "hand_above_head":      False,
            "arm_angle_left":       None,
            "arm_angle_right":      None,
            "torso_angle_left":     None,
            "torso_angle_right":    None,
            "both_arms_low_angle":  False,
            "torso_horizontal":     False,
        }

    # Ενημέρωση τελευταίων τιμών
    mem["last_seen"]        = now
    mem["last_bbox"]        = (x1, y1, x2, y2)
    mem["last_center"]      = (cx, cy)
    mem["velocity"]         = vel
    mem["is_horizontal"]    = horizontal
    mem["missing_since"]    = None

    mem["head_visible"]        = features["head_visible"]
    mem["legs_visible"]        = features["legs_visible"]
    mem["hand_above_head"]     = features["hand_above_head"]
    mem["arm_angle_left"]      = features["arm_angle_left"]
    mem["arm_angle_right"]     = features["arm_angle_right"]
    mem["torso_angle_left"]    = features["torso_angle_left"]
    mem["torso_angle_right"]   = features["torso_angle_right"]
    mem["both_arms_low_angle"] = features["both_arms_low_angle"]
    mem["torso_horizontal"]    = features["torso_horizontal"]

    # ── Temporal trackers (ξεκινούν χρονομέτρηση μόνο αν συνεχίζεται η συνθήκη) ──
    if features["head_below_shoulders"]:
        if mem["head_below_shoulders_since"] is None:
            mem["head_below_shoulders_since"] = now
    else:
        mem["head_below_shoulders_since"] = None

    if features["hand_above_head"]:
        if mem["hand_above_head_since"] is None:
            mem["hand_above_head_since"] = now
    else:
        mem["hand_above_head_since"] = None

    if horizontal:
        if mem["horizontal_since"] is None:
            mem["horizontal_since"] = now
    else:
        mem["horizontal_since"] = None

    return mem


def update_missing_ids(seen_ids, now):
    for tid, mem in track_memory.items():
        if tid not in seen_ids:
            if mem["missing_since"] is None:
                mem["missing_since"] = now
        else:
            mem["missing_since"] = None


def cleanup_old_tracks(now):
    to_delete = [
        tid for tid, mem in track_memory.items()
        if (now - mem.get("last_seen", now)) > MEMORY_TTL_SECS
    ]
    for tid in to_delete:
        del track_memory[tid]


# ══════════════════════════════════════════════════════════════════
# State detection  (safe / warning / danger)
# ══════════════════════════════════════════════════════════════════

def detect_state(track_id, pose, now):
    """
    Ιεραρχία κανόνων (από πιο σοβαρό σε λιγότερο):

    DANGER:
      1. Το άτομο λείπει από το frame για > WARNING_SECS
      2. Χέρια σε distress position (και οι δύο γωνίες βραχιόνων < ARM_DISTRESS_ANGLE)
         για > HAND_RAISE_SECS
      3. Οριζόντιο bbox + χαμηλή ταχύτητα + οριζόντιος κορμός (από γωνία)
         για > HORIZONTAL_SECS  [ισχυρή ένδειξη πτώσης]
      4. Οριζόντιο bbox + χαμηλή ταχύτητα (χωρίς γωνία κορμού)
         για > HORIZONTAL_SECS

    WARNING:
      5. Κεφάλι κάτω από ώμους για > HEAD_LOW_SECS
      6. Κεφάλι ορατό, πόδια όχι, χαμηλή ταχύτητα (μερική ορατότητα)
      7. Pose δεν εντοπίστηκε αλλά track εξακολουθεί να φαίνεται

    SAFE:
      8. Κεφάλι και πόδια ορατά, bbox κάθετο
    """
    mem = track_memory.get(track_id, {})

    velocity         = float(mem.get("velocity", 0.0))
    horizontal       = bool(mem.get("is_horizontal", False))
    missing_since    = mem.get("missing_since")
    head_visible     = bool(mem.get("head_visible", False))
    legs_visible     = bool(mem.get("legs_visible", False))
    hand_since       = mem.get("hand_above_head_since")
    head_low_since   = mem.get("head_below_shoulders_since")
    horizontal_since = mem.get("horizontal_since")
    both_arms_low    = bool(mem.get("both_arms_low_angle", False))
    torso_horiz      = bool(mem.get("torso_horizontal", False))

    # ── DANGER ────────────────────────────────────────────────────
    # 1. Εξαφάνιση από frame
    if missing_since is not None and (now - missing_since) >= WARNING_SECS:
        return "danger"

    # 2. Distress arm position (χέρια κλειστά γύρω από κεφάλι) για αρκετό χρόνο
    if both_arms_low and hand_since is not None and (now - hand_since) >= HAND_RAISE_SECS:
        return "danger"

    # 3. Οριζόντιο bbox + αργή κίνηση + γωνία κορμού επιβεβαιώνει οριζόντια θέση
    if (horizontal and torso_horiz and velocity < HORIZONTAL_VEL_THR
            and horizontal_since is not None
            and (now - horizontal_since) >= HORIZONTAL_SECS):
        return "danger"

    # 4. Οριζόντιο bbox + αργή κίνηση (χωρίς επιβεβαίωση γωνίας, λίγο πιο χαλαρό)
    if (horizontal and velocity < HORIZONTAL_VEL_THR
            and horizontal_since is not None
            and (now - horizontal_since) >= HORIZONTAL_SECS):
        return "danger"

    # ── WARNING ───────────────────────────────────────────────────
    # 5. Κεφάλι χαμηλά για αρκετό χρόνο
    if head_low_since is not None and (now - head_low_since) >= HEAD_LOW_SECS:
        return "warning"

    # 6. Μερική ορατότητα + ακινησία
    if head_visible and not legs_visible and velocity < 2.0:
        return "warning"

    # 7. Καθόλου pose αλλά track υπάρχει
    if not pose:
        return "warning"

    # ── SAFE ──────────────────────────────────────────────────────
    if head_visible and legs_visible and not horizontal:
        return "safe"

    return "safe"


# ══════════════════════════════════════════════════════════════════
# WebSocket endpoint
# ══════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            frame_bytes = await ws.receive_bytes()

            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                await ws.send_text(json.dumps({"ok": False, "error": "decode_failed"}))
                continue

            # ── YOLO inference ────────────────────────────────────
            results = model(img, conf=CONF_THR, verbose=False)
            detections = []

            for box in results[0].boxes:
                if int(box.cls[0]) != 0:   # class 0 = person
                    continue
                conf = float(box.conf[0])
                if conf < CONF_THR:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])

            if detections:
                detections = np.array(detections, dtype=np.float32)
            else:
                detections = np.empty((0, 5), dtype=np.float32)

            # ── ByteTrack ─────────────────────────────────────────
            tracks = tracker.update(detections, img.shape[:2], img.shape[:2])

            now = time.time()
            persons = []
            count_seen = 0

            for t in tracks:
                if count_seen >= MAX_PERSONS:
                    break

                x1, y1, x2, y2 = map(int, t.tlbr)
                track_id = int(t.track_id)

                pose = pose_landmarks_for_person(img, x1, y1, x2, y2)
                update_track_memory(track_id, pose, x1, y1, x2, y2, now)
                state = detect_state(track_id, pose, now)

                persons.append({
                    "id":    track_id,
                    "x1":   x1,
                    "y1":   y1,
                    "x2":   x2,
                    "y2":   y2,
                    "pose":  pose,
                    "state": state,
                })
                count_seen += 1

            seen_ids = {int(t.track_id) for t in tracks[:MAX_PERSONS]}
            update_missing_ids(seen_ids, now)
            cleanup_old_tracks(now)

            await ws.send_text(json.dumps({
                "ok":           True,
                "person_count": len(persons),
                "persons":      persons,
                "server_ts_ms": int(time.time() * 1000),
            }))

    except WebSocketDisconnect:
        print("Client disconnected", flush=True)

    except Exception as e:
        print("SERVER ERROR:", repr(e), flush=True)
        try:
            await ws.send_text(json.dumps({"ok": False, "error": str(e)}))
        except Exception:
            pass
