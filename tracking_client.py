import asyncio
import time
import json
import cv2
import websockets

WS_URL = "ws://147.102.74.170:8000/ws"
CAM_INDEX = 0
USE_VIDEO = True
MEDIA = "poolvid.mp4"
FPS = 5
TARGET_W, TARGET_H = 640, 480
JPEG_QUALITY = 75
WINDOW_NAME = "q to quit"

RECONNECT_DELAY = 2.0


def color_for_state(state):
    if state == "danger":
        return (0, 0, 255)
    elif state == "warning":
        return (0, 165, 255)
    else:
        return (0, 255, 0)


def draw_persons(frame, persons):
    for p in persons:
        x1 = int(p.get("x1", 0))
        y1 = int(p.get("y1", 0))
        x2 = int(p.get("x2", 0))
        y2 = int(p.get("y2", 0))
        track_id = p.get("id", -1)
        state = p.get("state", "safe")
        risk = p.get("risk", 0.0)
        pose = p.get("pose", None)
        color = color_for_state(state)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {track_id} {state} {risk:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        if pose:
            for lm in pose:
                x = int(lm["x"])
                y = int(lm["y"])
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)


def draw_overlay(frame, last_ok, last_person_count, last_error, max_risk, alert_state):
    status = "OK" if last_ok else "ERR"
    cv2.putText(
        frame,
        f"persons: {last_person_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"status: {status}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"max risk: {max_risk:.2f}",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color_for_state(alert_state),
        2,
        cv2.LINE_AA,
    )
    if not last_ok and last_error:
        cv2.putText(
            frame,
            f"error: {last_error}",
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


async def session(cap, encode_params, frame_interval):
    last_ok = False
    last_person_count = 0
    last_error = ""
    last_alert_ts = 0.0

    async with websockets.connect(
        WS_URL, max_size=10 * 1024 * 1024, proxy=None
    ) as ws:
        print(f"Connected to {WS_URL}")
        next_t = time.time()

        while True:
            now = time.time()
            if now < next_t:
                await asyncio.sleep(next_t - now)
            next_t += frame_interval

            ok, frame = cap.read()
            if not ok or frame is None:
                if USE_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Failed to read frame from camera")
                    break

            frame = cv2.resize(
                frame,
                (TARGET_W, TARGET_H),
                interpolation=cv2.INTER_AREA,
            )

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            jpg_bytes = buf.tobytes()

            await ws.send(jpg_bytes)
            reply = await ws.recv()
            data = json.loads(reply)

            last_ok = bool(data.get("ok", False))
            last_person_count = data.get("person_count", 0)
            last_error = data.get("error", "")
            persons = data.get("persons", []) if last_ok else []

            max_risk = 0.0
            alert_state = "safe"
            for p in persons:
                r = float(p.get("risk", 0.0))
                if r > max_risk:
                    max_risk = r
                if p.get("state") == "danger":
                    alert_state = "danger"
                elif p.get("state") == "warning" and alert_state != "danger":
                    alert_state = "warning"

            if alert_state in ("warning", "danger"):
                if (now - last_alert_ts) >= 1.0:
                    print("\a", end="", flush=True)
                    last_alert_ts = now

            draw_persons(frame, persons)
            draw_overlay(
                frame, last_ok, last_person_count, last_error, max_risk, alert_state
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False

    return True


async def main():
    if USE_VIDEO:
        cap = cv2.VideoCapture(MEDIA)
    else:
        cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open source")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    frame_interval = 1.0 / FPS

    try:
        while True:
            try:
                should_reconnect = await session(cap, encode_params, frame_interval)
                if not should_reconnect:
                    break
            except (
                websockets.ConnectionClosed,
                ConnectionRefusedError,
                OSError,
            ) as e:
                print(f"Connection lost: {e}. Reconnecting in {RECONNECT_DELAY}s...")
                await asyncio.sleep(RECONNECT_DELAY)
            except Exception as e:
                print(f"Unexpected error: {e!r}. Reconnecting in {RECONNECT_DELAY}s...")
                await asyncio.sleep(RECONNECT_DELAY)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
