"""
Emotion_Detection_System - Real-time Emotion Detection Engine
=============================================
Captures webcam feed, detects faces, classifies emotions using DeepFace,
logs every inference to SQLite, and fires alerts on repeated negative emotions.

Compatible with NVIDIA Jetson (Xavier NX / Nano) via cuDNN-enabled OpenCV.
"""

import os

import cv2
import datetime
import sqlite3
import time
import threading
import argparse
from collections import deque
from deepface import DeepFace

# ─────────────────────────────────────────────
# CONFIG — all tunable values in one place
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "alert_threshold": 5,        # how many consecutive negative detections trigger alert
    "alert_window_sec": 10,      # time window (seconds) to count negatives within
    "frame_skip": 2,             # process every Nth frame (1 = every frame, 2 = every other)
    "resize_width": 640,         # resize frame before inference (smaller = faster)
    "db_path": "Emotion_Detection_System_log.db",
    "negative_emotions": {"sad", "angry", "fear", "disgust"},
}

# Shared state — dashboard.py reads from the DB, but we also keep a live dict
# so other modules can read it without hitting the DB each time
shared_state = {
    "current_emotion": "N/A",
    "current_confidence": 0.0,
    "alert_active": False,
    "fps": 0.0,
}
state_lock = threading.Lock()


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def init_db(db_path: str) -> sqlite3.Connection:
    """Create the detections table if it doesn't exist."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            emotion   TEXT    NOT NULL,
            confidence REAL   NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_detection(conn: sqlite3.Connection, emotion: str, confidence: float):
    """Write one inference event to the database."""
    ts = datetime.datetime.now().isoformat(timespec="milliseconds")
    confidence_val = float(round(float(confidence), 4))
    conn.execute(
        "INSERT INTO detections (timestamp, emotion, confidence) VALUES (?, ?, ?)",
        (ts, str(emotion), confidence_val),
    )
    conn.commit()

# ─────────────────────────────────────────────
# ALERT SYSTEM
# ─────────────────────────────────────────────

class AlertManager:
    """
    Tracks recent detections in a sliding time window.
    Fires an alert when negative emotions exceed the threshold.

    Hook points for future integrations:
        - on_alert_email()   → plug in smtplib / sendgrid
        - on_alert_mqtt()    → plug in paho-mqtt
    """

    def __init__(self, threshold: int, window_sec: float, negative_emotions: set):
        self.threshold = threshold
        self.window_sec = window_sec
        self.negative_emotions = negative_emotions
        self._timestamps: deque = deque()  # timestamps of negative detections
        self._alert_fired = False           # prevents repeated console spam
        self._last_alert_emotion = None

    def update(self, emotion: str) -> bool:
        """
        Call after every detection.
        Returns True if alert should fire, False otherwise.
        """
        now = time.time()

        # Only count genuinely negative emotions
        if emotion.lower() in self.negative_emotions:
            self._timestamps.append(now)

        # Drop timestamps outside the window
        while self._timestamps and (now - self._timestamps[0]) > self.window_sec:
            self._timestamps.popleft()

        alert = len(self._timestamps) >= self.threshold

        # Fire alert once per trigger — reset when it drops below threshold
        if alert and not self._alert_fired:
            self._alert_fired = True
            self._last_alert_emotion = emotion
            self._trigger_alert(emotion)
        elif not alert:
            self._alert_fired = False  # reset so it can fire again next time

        return alert

    def _trigger_alert(self, emotion: str):
        """Console alert — extend this method for other channels."""
        print(f"[ALERT] Negative emotion '{emotion}' detected {self.threshold}+ times "
              f"in {self.window_sec}s window!")
        self._on_alert_email_hook(emotion)
        self._on_alert_mqtt_hook(emotion)

    # ── Future hooks (implement these to add notification channels) ──
    def _on_alert_email_hook(self, emotion: str):
        """Send email alert via Gmail."""
        import smtplib
        from email.mime.text import MIMEText
        from dotenv import load_dotenv
        load_dotenv()

        SENDER_EMAIL    = os.environ.get("SENDER_EMAIL")
        SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
        RECEIVER_EMAIL  = os.environ.get("RECEIVER_EMAIL")

        if not all([SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL]):
            print("[EMAIL] Missing email config in .env file")
            return

        subject = f"Emotion_Detection_System Alert — Negative Emotion Detected"
        body    = (
            f"Alert triggered!\n\n"
            f"Emotion   : {emotion}\n"
            f"Threshold : {self.threshold} detections in {self.window_sec}s\n"
            f"Time      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Check the Emotion_Detection_System dashboard for details."
        )

        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"]    = SENDER_EMAIL
            msg["To"]      = RECEIVER_EMAIL

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

            print(f"[EMAIL] Alert sent to {RECEIVER_EMAIL}")

        except Exception as e:
            print(f"[EMAIL] Failed to send: {e}")


    def _on_alert_mqtt_hook(self, emotion: str):
        """TODO: plug in MQTT publish here (paho-mqtt)."""
        pass

    def update_config(self, threshold: int, window_sec: float):
        """Hot-reload config without restarting — called by dashboard sliders."""
        self.threshold = threshold
        self.window_sec = window_sec


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def run(config: dict = None):
    if config is None:
        config = DEFAULT_CONFIG.copy()

    conn = init_db(config["db_path"])
    alert_mgr = AlertManager(
        threshold=config["alert_threshold"],
        window_sec=config["alert_window_sec"],
        negative_emotions=config["negative_emotions"],
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check USB connection.")

    # Try to use GPU-accelerated backend if available (Jetson / CUDA)
    # Falls back to CPU silently if CUDA is not available
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    frame_count = 0
    fps_timer = time.time()
    last_emotion = "N/A"
    last_confidence = 0.0
    alert_active = False

    print("Emotion_Detection_System running — press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1

        # ── Frame skipping for performance ──
        if frame_count % config["frame_skip"] != 0:
            # Still draw last known result on skipped frames
            _draw_overlay(frame, last_emotion, last_confidence, alert_active)
            cv2.imshow("Emotion_Detection_System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ── Resize for faster inference ──
        h, w = frame.shape[:2]
        scale = config["resize_width"] / w
        small = cv2.resize(frame, (config["resize_width"], int(h * scale)))

        # ── DeepFace inference ──
        try:
            results = DeepFace.analyze(
                small,
                actions=["emotion"],
                enforce_detection=False,   # don't crash if no face found
                silent=True,
            )
            emotions = results[0]["emotion"]                    # dict of emotion→score
            dominant = results[0]["dominant_emotion"]
            confidence = emotions[dominant] / 100.0             # normalise to 0–1

            last_emotion = dominant
            last_confidence = confidence

            # ── Log to DB ──
            log_detection(conn, dominant, confidence)

            # ── Alert check ──
            alert_active = alert_mgr.update(dominant)

            # ── Update shared state for dashboard ──
            with state_lock:
                shared_state["current_emotion"] = dominant
                shared_state["current_confidence"] = confidence
                shared_state["alert_active"] = alert_active

        except Exception as e:
            # No face detected or inference error — keep last known values
            pass

        # ── FPS calculation ──
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()
            with state_lock:
                shared_state["fps"] = round(fps, 1)

        # ── Draw overlay ──
        _draw_overlay(frame, last_emotion, last_confidence, alert_active)
        cv2.imshow("Emotion_Detection_System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    print("Emotion_Detection_System stopped.")


def _draw_overlay(frame, emotion: str, confidence: float, alert: bool):
    """Draw emotion label, confidence bar, and alert banner onto frame."""
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Emotion + confidence text
    label = f"{emotion.upper()}  {confidence * 100:.1f}%"
    cv2.putText(frame, label, (15, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 180), 2)

    # Confidence bar
    bar_w = int(w * 0.4 * confidence)
    cv2.rectangle(frame, (15, 50), (15 + bar_w, 56), (0, 255, 180), -1)

    # Alert banner
    if alert:
        cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, "⚠  ALERT: Sustained negative emotion detected",
                    (15, h - 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion_Detection_System Emotion Detection")
    parser.add_argument("--threshold", type=int, default=DEFAULT_CONFIG["alert_threshold"],
                        help="Negative detections needed to trigger alert")
    parser.add_argument("--window", type=float, default=DEFAULT_CONFIG["alert_window_sec"],
                        help="Time window in seconds for alert counting")
    parser.add_argument("--skip", type=int, default=DEFAULT_CONFIG["frame_skip"],
                        help="Process every Nth frame (higher = faster but less smooth)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["alert_threshold"] = args.threshold
    config["alert_window_sec"] = args.window
    config["frame_skip"] = args.skip

    run(config)