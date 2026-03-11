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


def color_for_state(state):
    if state == "danger":
        return (0, 0, 255)      # red
    elif state == "warning":
        return (255, 0, 0)      # blue
    else:
        return (0, 255, 0)      # green


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
    next_t = time.time()

    last_ok = False
    last_person_count = 0
    last_error = ""

    async with websockets.connect(WS_URL, max_size=10 * 1024 * 1024) as ws:
        while True:
            # FPS pacing
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
                interpolation=cv2.INTER_AREA
            )

            # JPEG encode
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            jpg_bytes = buf.tobytes()

            # send
            await ws.send(jpg_bytes)

            # receive
            reply = await ws.recv()
            data = json.loads(reply)

            # debug pio light
            print("ok:", data.get("ok"), "persons:", data.get("person_count"))

            last_ok = bool(data.get("ok", False))
            last_person_count = data.get("person_count", 0)
            last_error = data.get("error", "")

            persons = data.get("persons", []) if last_ok else []

            # -------------------------
            # Draw detections
            # -------------------------
            for p in persons:
                x1 = int(p.get("x1", 0))
                y1 = int(p.get("y1", 0))
                x2 = int(p.get("x2", 0))
                y2 = int(p.get("y2", 0))
                conf = p.get("conf", None)
                pose = p.get("pose", None)
                state = p.get("state", "safe")

                color = color_for_state(state)
                if color =="(0, 0, 255)":
                	print('\a') #hxhtiko                	

                # bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # label
                if conf is not None:
                    label = f"{state} {conf:.2f}"
                else:
                    label = state

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

                # pose landmarks
                if pose:
                    for lm in pose:
                        x = int(lm["x"])
                        y = int(lm["y"])
                        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

            # -------------------------
            # Overlay status
            # -------------------------
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

            if not last_ok and last_error:
                cv2.putText(
                    frame,
                    f"error: {last_error}",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
