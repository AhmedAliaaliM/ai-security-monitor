# AI Security Monitor

AI-powered security camera pipeline for real-time **person, fire/smoke, motion, and face-recognition monitoring**.

The system combines computer vision models with configurable detection zones, face authorization, low-light enhancement, scheduling, and a Streamlit dashboard to provide an end-to-end intelligent security monitoring system.

## Features

* **Motion detection** with configurable detection zones and ignore zones
* **Person detection** using YOLOv8n trained on the COCO dataset
* **Fire/smoke detection** using a dedicated YOLOv8 model
* **Face recognition** for authorized and unauthorized-person detection
* **Person-confidence gating** before running the face-recognition pipeline
* **Low-light enhancement** using Zero-DCE++ before selected AI checks
* **AI-mode scheduling** to control when the full AI pipeline is active
* **Authorized-entry scheduling** based on configurable days and hours
* **Pet filtering** to reduce unnecessary motion triggers
* **Repetitive-motion suppression** to reduce alerts from objects such as moving branches or flags
* **Configurable detection and ignore zones**
* **Live Streamlit dashboard** for monitoring and configuration
* **CLI mode** for running the pipeline without the dashboard
* **Live configuration updates** through the dashboard
* **Privacy-conscious design** with camera credentials and face data excluded from Git

## System Overview

The pipeline processes camera frames through several specialized computer-vision components:

```text
Camera / RTSP / Webcam
          │
          ▼
   Frame Capture
          │
          ▼
   Motion Detection
          │
          ├──────────────► No motion → Continue monitoring
          │
          ▼
   Zone / Pet / Motion Filtering
          │
          ▼
    Person Detection
          │
          ├──────────────► Person detected
          │                       │
          │                       ▼
          │                Face Recognition
          │                       │
          │                       ▼
          │                Authorization Check
          │
          ▼
 Fire / Smoke Detection
          │
          ▼
 Low-Light Enhancement
          │
          ▼
 Alerts / Dashboard / Logs
```

Fire/smoke detection operates independently from motion-triggered person processing, allowing hazard detection to run according to its own configured interval.

## Tech Stack

| Component             | Technology          |
| --------------------- | ------------------- |
| Language              | Python              |
| Deep Learning         | PyTorch             |
| Object Detection      | Ultralytics YOLOv8  |
| Person Detection      | YOLOv8n / COCO      |
| Fire Detection        | Custom YOLOv8 model |
| Face Recognition      | InsightFace         |
| Image Processing      | OpenCV              |
| Low-Light Enhancement | Zero-DCE++          |
| Dashboard             | Streamlit           |
| Configuration         | JSON                |
| Version Control       | Git / GitHub        |

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AhmedAliaaliM/ai-security-monitor.git
cd ai-security-monitor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the appropriate PyTorch build for your CUDA environment before running the pipeline.

### 3. Configure the camera

The application uses `my_camera.json` for camera and pipeline configuration.

You can generate a starter configuration with:

```bash
python main.py --init-config my_camera.json
```

Or use the dashboard to configure the camera and pipeline settings.

For a webcam:

```json
{
  "source": 0
}
```

For an RTSP camera:

```json
{
  "source": "rtsp://username:password@camera-ip:554/stream"
}
```

**Do not commit camera URLs containing credentials.**

## Run the Dashboard

The recommended way to use the system is through the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The dashboard provides access to:

* Live camera feed
* Detection results
* Motion zones
* Ignore zones
* Camera configuration
* AI scheduling
* Face enrollment
* Detection thresholds
* Alert configuration
* Pipeline status

See [`DASHBOARD_README.md`](DASHBOARD_README.md) for the detailed dashboard documentation.

## Run the CLI

The pipeline can also be run directly from the command line:

```bash
python main.py --config my_camera.json --preview
```

The `--preview` option displays the camera feed with detection information.

For headless execution:

```bash
python main.py --config my_camera.json
```

## Face Enrollment

Authorized faces can be enrolled using:

```bash
python enroll_face.py
```

The resulting face information is stored locally and should not be committed to GitHub.

The system uses the enrolled face embeddings to determine whether a detected person matches an authorized identity.

## Configuration

The main configuration file is:

```text
my_camera.json
```

This file is intentionally gitignored because it may contain camera URLs, credentials, and other local settings.

Important configuration options include:

| Setting                          | Purpose                                          |
| -------------------------------- | ------------------------------------------------ |
| `source`                         | Webcam index, video source, or RTSP URL          |
| `detection_zones`                | Areas where motion detection is enabled          |
| `ignore_zones`                   | Areas excluded from motion detection             |
| `sensitivity`                    | Motion detection sensitivity                     |
| `motion_area_threshold`          | Minimum movement area required for a trigger     |
| `hazard_confidence_threshold`    | Confidence threshold for fire/smoke detection    |
| `face_match_threshold`           | Similarity threshold for face recognition        |
| `person_confidence_threshold`    | Minimum person confidence before face processing |
| `schedule.ai_mode`               | Schedule controlling full AI processing          |
| `auth_schedule`                  | Days/hours when recognized faces are authorized  |
| `cooldown_seconds`               | Minimum time between recording/alert events      |
| `max_clip_seconds`               | Maximum recorded clip duration                   |
| `pre_buffer_seconds`             | Amount of video retained before a trigger        |
| `repetitive_motion_window`       | Time window for repetitive-motion analysis       |
| `repetitive_motion_max_triggers` | Maximum repeated triggers before suppression     |

The configuration schema and default values are defined in:

[`config/camera_config.py`](config/camera_config.py)

## Detection Zones

Detection zones allow the system to focus on specific regions of the camera frame.

Example:

```json
{
  "detection_zones": [
    [
      [440, 20],
      [620, 20],
      [620, 200],
      [440, 200]
    ]
  ]
}
```

This can be used to monitor a specific area such as:

* Doorways
* Windows
* Hallways
* Restricted areas
* Entry points

Ignore zones can be used to exclude areas that frequently generate false motion detections.

## Project Structure

```text
ai-security-monitor/
│
├── main.py                       # CLI entry point
├── app.py                        # Streamlit dashboard
├── engine.py                     # Dashboard/background pipeline engine
├── enroll_face.py                # Face enrollment CLI
│
├── config/
│   └── camera_config.py          # Configuration schema and defaults
│
├── core/
│   ├── capture.py                # Camera capture and motion triggering
│   ├── motion_router.py          # Motion, zone and filtering logic
│   ├── person_detector.py        # YOLOv8n person detection
│   ├── hazard_pipeline.py        # Fire/smoke detection
│   ├── identity_pipeline.py      # Face detection and recognition
│   ├── enhancement_pipeline.py   # Zero-DCE++ enhancement
│   └── schedule.py               # Scheduling logic
│
├── models/
│   ├── fire_yolov8n.pt           # Fire/smoke detection model
│   └── ...                        # Other model weights
│
├── .streamlit/
│   └── config.toml               # Streamlit configuration
│
├── requirements.txt              # Python dependencies
├── DASHBOARD_README.md            # Detailed dashboard documentation
└── README.md                     # Project documentation
```

## Local vs Cloud Deployment

This project is primarily designed for **local camera processing**.

The application opens the configured camera source using OpenCV. When using:

```python
cv2.VideoCapture(0)
```

the camera is the webcam attached to the computer running the application.

Therefore:

### Local PC

Running the application on the computer connected to the camera works normally:

```text
Camera
   │
   ▼
Local PC
   │
   ├── YOLO
   ├── Face Recognition
   ├── Fire Detection
   ├── Enhancement
   └── Streamlit Dashboard
```

### Cloud Hosting

Simply deploying the repository to a cloud platform does **not** automatically provide access to your physical webcam.

A cloud server generally cannot access:

```text
Your PC webcam
      │
      X
Cloud Server
```

For remote camera monitoring, an accessible camera stream such as RTSP, or another secure video-ingestion architecture, is required.

See [`DASHBOARD_README.md`](DASHBOARD_README.md) for deployment and remote-access considerations.

## Privacy

The project is designed to keep sensitive local information out of the repository.

The following files should remain local:

```text
my_camera.json
known_faces.json
```

These may contain:

* Camera URLs
* Camera credentials
* Face embeddings
* Authorized identities
* Local configuration

They should therefore be included in `.gitignore`.

Example:

```gitignore
my_camera.json
known_faces.json
*.env
__pycache__/
*.pyc
```

Model weights can also be excluded from Git if they are large. In that case, document where users can obtain them and how to place them in the `models/` directory.

## Limitations

* Local webcam access requires the camera to be accessible from the machine running the application.
* Detection accuracy depends on camera quality, lighting, camera angle, and model quality.
* Face recognition performance can decrease with occlusion, extreme angles, poor lighting, or low-resolution faces.
* Fire/smoke detection is model-dependent and should not be treated as a certified fire-safety system.
* Remote deployment requires an appropriate camera-stream architecture.
* GPU acceleration is recommended for higher-resolution or multi-camera workloads.

## Future Improvements

Potential future improvements include:

* Multi-camera support
* Event database and historical analytics
* Mobile notifications
* WebRTC-based remote streaming
* Docker deployment
* REST API for external integrations
* Object tracking
* Improved fire/smoke datasets
* Model quantization for edge devices
* Edge deployment on NVIDIA Jetson / Raspberry Pi-class hardware
* Automated evaluation and benchmark reporting

## Documentation

* [`DASHBOARD_README.md`](DASHBOARD_README.md) — Dashboard usage, configuration, and deployment details
* [`config/camera_config.py`](config/camera_config.py) — Configuration schema and defaults

## License

Add the project's license here if one has been selected.

---

**AI Security Monitor** — intelligent computer vision for real-time security monitoring.
