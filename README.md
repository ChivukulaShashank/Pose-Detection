# 📌 Human Pose Detection & Posture Classification System

This project uses **Ultralytics YOLOv11 Pose Estimation** to detect human keypoints in real-time via webcam and classify posture as **Standing**, **Sitting**, or **Sleeping**, with skeletal overlay and CSV logging. The system can also trigger a sound using a local server when specific postures are detected.

---

## 🧠 Features

- 🔍 Real-time **pose detection** using `yolo11n-pose.pt`
- 📐 Calculates **leg joint angles** to determine posture
- 🪑 Classifies posture as:
  - `Standing` (avg angle > 120°)
  - `Sitting` (avg angle ≤ 120°)
  - `Sleeping` (based on body width vs height)
- 🔊 Sends trigger to speaker via **HTTP POST** when posture changes
- 🎥 Saves annotated video output as `output_with_posture_camera.mp4`
- 📄 Logs pose data (with timestamp, angles, posture, keypoints) into `pose_data.csv`
- 🧠 Skeleton rendering and joint marking on video frame

---

## 🛠️ Requirements

- Python 3.8+
- A webcam
- Install dependencies:

```bash
pip install ultralytics opencv-python numpy requests
```

> ⚠️ Note: This assumes you already have `yolo11n-pose.pt`.

---

## 🚀 How to Run

```bash
git clone https://github.com/ChivukulaShashank/Pose-Detection.git
cd Pose-Detection
python posture_detection.py
```

Press `q` to exit the webcam window.

---

## 🔊 Speaker Trigger (Optional)

This script optionally sends an HTTP POST to a speaker server:
- Endpoint: `http://192.168.0.44:80/play`
- Payload:
```json
{ "play": "1.mp3", "times": 1 }
```

You can disable this by commenting out the `sound_trigger()` call.

---

## 📁 Output Files

- `pose_data.csv` – posture logs
- `output_with_posture_camera.mp4` – processed video

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
