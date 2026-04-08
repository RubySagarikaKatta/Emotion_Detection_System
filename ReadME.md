# 🧠 Neofocus — Real-time Emotion Detection System

A production-ready emotion detection pipeline built with **DeepFace**, **OpenCV**, and **Streamlit**.  
Designed for USB webcams, with optimisations targeting **NVIDIA Jetson** (Xavier NX / Nano).

---

## Features

| Feature | Details |
|---|---|
| 🎭 Emotion detection | DeepFace — 7 emotions: happy, sad, angry, fear, disgust, surprise, neutral |
| 📊 Confidence overlay | Live percentage on the webcam feed |
| 🗄️ SQLite logging | Every inference timestamped and stored |
| ⚠️ Alert system | Fires when negative emotions repeat within a configurable time window |
| 🔌 Alert hooks | Stub functions ready for email / MQTT plug-in |
| 📈 Live dashboard | Streamlit — emotion timeline, distribution chart, recent detections table |
| 🎛️ Hot config | Adjust alert thresholds via dashboard sliders — no restart needed |
| ⚡ Jetson-ready | Frame skipping, resize optimisation, cuDNN-compatible OpenCV |

---

## Project Structure

```
neofocus/
├── app.py            # Webcam capture + DeepFace inference + logging + alerts
├── dashboard.py      # Streamlit dashboard
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Quick Start (Standard PC / Mac)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/neofocus.git
cd neofocus
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the detection engine
```bash
python app.py
```
Press **`q`** in the OpenCV window to quit.

### 5. Run the dashboard (separate terminal)
```bash
streamlit run dashboard.py
```
Open **http://localhost:8501** in your browser.

---

## Configuration

You can pass arguments directly to `app.py`:

```bash
python app.py --threshold 5 --window 10 --skip 2
```

| Argument | Default | Description |
|---|---|---|
| `--threshold` | `5` | Negative detections needed to trigger alert |
| `--window` | `10` | Time window in seconds to count negatives within |
| `--skip` | `2` | Process every Nth frame (higher = faster, less smooth) |

All alert thresholds can also be adjusted **live** via the dashboard sidebar sliders.

---

## NVIDIA Jetson Installation

> Tested target: Jetson Xavier NX / Nano with JetPack 5.x

### 1. Install JetPack (includes CUDA, cuDNN, TensorRT)
Follow NVIDIA's official [JetPack SDK guide](https://developer.nvidia.com/embedded/jetpack).

### 2. Install cuDNN-enabled OpenCV
JetPack ships with a CUDA-enabled OpenCV. Do **not** install via pip — it will overwrite it.  
Verify with:
```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i cuda
```
You should see `CUDA: YES`.

### 3. Install Python dependencies (excluding OpenCV)
```bash
pip install deepface streamlit plotly pandas tensorflow tf-keras
```

### 4. Performance tuning on Jetson
- Set `--skip 2` or higher if FPS drops below 15
- Use `--resize 480` (edit `DEFAULT_CONFIG["resize_width"]`) for faster inference
- Enable max power mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`

---

## Alert Hook Integration

`app.py` contains two stub methods in the `AlertManager` class, ready for you to implement:

```python
def _on_alert_email_hook(self, emotion: str):
    """TODO: plug in email notification here (smtplib / SendGrid)."""
    pass

def _on_alert_mqtt_hook(self, emotion: str):
    """TODO: plug in MQTT publish here (paho-mqtt)."""
    pass
```

---

## Dashboard Preview

The Streamlit dashboard (localhost:8501) shows:
- **Current emotion** and confidence — updates every 2 seconds
- **Alert banner** when the threshold is exceeded
- **Emotion timeline** scatter chart (colour-coded by emotion)
- **Distribution bar chart** across the session
- **Recent detections table** (last 15 entries)
- **Sidebar sliders** for live threshold adjustment

---

## How It Works

```
Webcam frame
    │
    ▼
Resize + frame skip  (performance optimisation)
    │
    ▼
DeepFace.analyze()   (face detection + emotion classification)
    │
    ├─► SQLite log   (timestamp, emotion, confidence)
    │
    ├─► AlertManager (sliding window counter)
    │       └─► on_alert hooks (email / MQTT stubs)
    │
    └─► OpenCV overlay (label + confidence bar + alert banner)
```

---

## Tech Stack

- **[DeepFace](https://github.com/serengil/deepface)** — emotion classification
- **[OpenCV](https://opencv.org/)** — webcam capture and frame rendering
- **[Streamlit](https://streamlit.io/)** — dashboard UI
- **[Plotly](https://plotly.com/python/)** — interactive charts
- **[SQLite](https://www.sqlite.org/)** — lightweight local database

---

## License

MIT