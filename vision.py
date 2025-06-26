from ultralytics import YOLO
import cv2
import numpy as np
import csv
import time
import requests

# Load the pretrained YOLOv11 pose model
model = YOLO("yolo11n-pose.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get input FPS (fallback to 30 if unknown)
input_fps = cap.get(cv2.CAP_PROP_FPS)
if input_fps == 0:
    input_fps = 30

desired_fps = 1
frame_skip = max(int(input_fps / desired_fps), 1)
print(f"Input FPS: {input_fps}, Processing every {frame_skip} frame(s) to achieve ~{desired_fps} FPS.")

# Get frame dimensions
frame_width = 640
frame_height = 360

# Initialize video writer
out = cv2.VideoWriter(
    "output_with_posture_camera.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    desired_fps,
    (frame_width, frame_height)
)

# Define skeleton connections
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 13), (12, 14),
    (13, 15), (14, 16)
]

leg_keypoint_indices = [13, 14, 15, 16]

# Sound trigger function
def sound_trigger(play="1.mp3", times=1):
    try:
        url = 'http://192.168.0.44:80/play'
        payload = {"play": play, "times": times}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Success:", response.text)
        else:
            print(f"Failed! Status Code: {response.status_code}")
    except Exception as e:
        print("Error:", e)

# CSV writer setup
csv_file = open("pose_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Person", "Posture", "Angle_Left", "Angle_Right", "Keypoints"])

# Angle calculator
def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180 / np.pi
    return angle

frame_count = 0
last_sitting_trigger = 0
last_standing_trigger = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Error: Could not read frame.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    results = model(frame)

    if results[0].keypoints is None or results[0].boxes is None:
        continue

    keypoints = results[0].keypoints.xy.cpu().numpy()
    confidences = results[0].keypoints.conf
    confidences = confidences.cpu().numpy() if confidences is not None else np.ones_like(keypoints[..., 0])
    boxes = results[0].boxes.xyxy.cpu().numpy()

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for i, (box, person_keypoints, person_confidences) in enumerate(zip(boxes, keypoints, confidences)):
        x_min, y_min, x_max, y_max = map(int, box)
        width = x_max - x_min
        height = y_max - y_min

        leg_detected = True
        for idx in leg_keypoint_indices:
            if idx >= len(person_keypoints):
                leg_detected = False
                break
            x, y = person_keypoints[idx]
            conf = person_confidences[idx] if idx < len(person_confidences) else 0.0
            if x <= 0 or y <= 0 or conf < 0.5:
                leg_detected = False
                break

        angle_left = angle_right = None

        if not leg_detected:
            posture = "Full body not detected"
        elif width > 1.5 * height:
            posture = "Sleeping"
        else:
            angles = []
            if all(person_keypoints[k][0] > 0 for k in [11, 13, 15]):
                angle_left = calculate_angle(person_keypoints[11], person_keypoints[13], person_keypoints[15])
                angles.append(angle_left)
            if all(person_keypoints[k][0] > 0 for k in [12, 14, 16]):
                angle_right = calculate_angle(person_keypoints[12], person_keypoints[14], person_keypoints[16])
                angles.append(angle_right)

            avg_angle = np.mean(angles) if angles else 180.0
            posture = "Standing" if avg_angle > 120 else "Sitting"

            # Trigger speaker at specific intervals
            current_time = time.time()
            if posture == "Sitting" and current_time - last_sitting_trigger > 2:
                sound_trigger()
                last_sitting_trigger = current_time
            elif posture == "Standing" and current_time - last_standing_trigger > 5:
                sound_trigger()
                last_standing_trigger = current_time

        print(f"Person {i+1} Box: W={width}px, H={height}px, Posture={posture}")
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(frame, f"W:{width}px H:{height}px {posture}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for point in person_keypoints:
            x, y = int(point[0]), int(point[1])
            if x > 0 and y > 0:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        for start, end in skeleton:
            if start < len(person_keypoints) and end < len(person_keypoints):
                start_point = tuple(map(int, person_keypoints[start]))
                end_point = tuple(map(int, person_keypoints[end]))
                if all(coord > 0 for coord in start_point + end_point):
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

        # Save to CSV
        csv_writer.writerow([
            timestamp, i+1, posture,
            f"{angle_left:.2f}" if angle_left else "NA",
            f"{angle_right:.2f}" if angle_right else "NA",
            person_keypoints.tolist()
        ])

    out.write(frame)
    cv2.imshow("Pose Estimation with Posture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
