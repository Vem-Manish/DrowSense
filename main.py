import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
from PIL import Image
import numpy as np
import time
import os
import winsound

# --- CONFIGURATION ---
EYE_MODEL_PATH = "mobilenet_v3_best.pth"
YAWN_MODEL_PATH = "yawn_model_2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SOUND FILES ---
ALERT_SOUND = "alert.wav"
YAWN_SOUND = "yawn.wav"
HEAD_SOUND = "danger.wav"

# --- SOUND STATE (ADDED) ---
is_sound_playing = False
current_sound = None

# --- SOUND FUNCTIONS (UPDATED) ---
def play_loop(sound_file):
    global is_sound_playing, current_sound
    if not is_sound_playing or current_sound != sound_file:
        winsound.PlaySound(None, winsound.SND_PURGE)
        winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        is_sound_playing = True
        current_sound = sound_file


def stop_sound():
    global is_sound_playing, current_sound
    winsound.PlaySound(None, winsound.SND_PURGE)
    is_sound_playing = False
    current_sound = None

# --- THRESHOLDS ---
EYE_THRESHOLD = 0.40
YAWN_THRESHOLD = 0.99

TARGET_FPS = 10

# --- HEAD POSE CONFIG ---
DROOP_THRESHOLD = -20.0
TIME_THRESHOLD = 1.5

model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

# --- MODEL LOADER ---
def load_mobilenet_v3(path):
    print(f"[INFO] Loading model from {path}...")
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        exit()

    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 1)

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_yawn_model_advanced(path):
    print(f"[INFO] Loading Advanced Yawn Model from {path}...")
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        exit()

    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1)
    )

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --- PREPROCESSING ---
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- MEDIAPIPE SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_IDXS = [33, 133, 159, 145]
RIGHT_EYE_IDXS = [362, 263, 386, 374]
MOUTH_IDXS = [13, 14, 61, 291]

# --- HEAD POSE FUNCTION ---
def get_head_pose(landmarks, frame_w, frame_h):
    image_points = np.array([
        (landmarks[1].x * frame_w, landmarks[1].y * frame_h),
        (landmarks[152].x * frame_w, landmarks[152].y * frame_h),
        (landmarks[33].x * frame_w, landmarks[33].y * frame_h),
        (landmarks[263].x * frame_w, landmarks[263].y * frame_h),
        (landmarks[61].x * frame_w, landmarks[61].y * frame_h),
        (landmarks[291].x * frame_w, landmarks[291].y * frame_h)
    ], dtype=np.float64)

    focal_length = frame_w
    camera_matrix = np.array([
        [focal_length, 0, frame_w / 2],
        [0, focal_length, frame_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    _, rot_vec, trans_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, np.zeros((4, 1))
    )

    rmat, _ = cv2.Rodrigues(rot_vec)
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(
        np.hstack((rmat, trans_vec))
    )

    pitch = angles.flatten()[0]
    pitch = (180 - pitch) if pitch > 0 else (-180 - pitch)

    return pitch


# --- CROP FUNCTIONS ---
def get_eye_crop(frame, landmarks, indices):
    h, w, _ = frame.shape
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

    x_min = min([c[0] for c in coords])
    x_max = max([c[0] for c in coords])
    y_min = min([c[1] for c in coords])
    y_max = max([c[1] for c in coords])

    eye_width = x_max - x_min
    pad_x = int(eye_width * 0.40)
    pad_y = int(eye_width * 0.60)

    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)

    crop_w = x_max - x_min
    crop_h = y_max - y_min
    if crop_w > crop_h:
        diff = crop_w - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_max + diff // 2)
    else:
        diff = crop_h - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_max + diff // 2)

    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def get_mouth_crop(frame, landmarks):
    h, w, _ = frame.shape
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH_IDXS]

    x_min = min([c[0] for c in coords])
    x_max = max([c[0] for c in coords])
    y_min = min([c[1] for c in coords])
    y_max = max([c[1] for c in coords])

    feature_width = x_max - x_min
    feature_height = y_max - y_min

    pad_x = int(feature_width * 0.40)
    pad_y_top = int(feature_height * 0.20)
    pad_y_bot = int(feature_height * 0.60)

    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y_top)
    y_max = min(h, y_max + pad_y_bot)

    crop_w = x_max - x_min
    crop_h = y_max - y_min
    if crop_w > crop_h:
        diff = crop_w - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_max + diff // 2)
    else:
        diff = crop_h - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_max + diff // 2)

    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def predict(model, img_bgr):
    if img_bgr.size == 0:
        return 0.0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = preprocess_transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    return prob


# --- MAIN LOOP ---
def main():
    global last_alert_time

    eye_model = load_mobilenet_v3(EYE_MODEL_PATH)
    yawn_model = load_yawn_model_advanced(YAWN_MODEL_PATH)

    cap = cv2.VideoCapture(0)
    delay_ms = int(1000 / TARGET_FPS)

    droop_start_time = None

    print(f"[INFO] System Ready @ {TARGET_FPS} FPS.")
    print(f"[INFO] Eye < {EYE_THRESHOLD} | Yawn > {YAWN_THRESHOLD}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        main_alert_text = "Status: Active"
        main_alert_color = (0, 255, 0)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # --- HEAD POSE ---
                pitch = get_head_pose(landmarks, frame.shape[1], frame.shape[0])
                head_drop_active = False

                if pitch < DROOP_THRESHOLD:
                    if droop_start_time is None:
                        droop_start_time = time.time()

                    elapsed = time.time() - droop_start_time

                    if elapsed >= TIME_THRESHOLD:
                        head_drop_active = True
                        main_alert_text = "DANGER: HEAD DROP!"
                        main_alert_color = (0, 0, 255)
                    else:
                        main_alert_text = f"Droop Detected: {elapsed:.1f}s"
                        main_alert_color = (0, 165, 255)
                else:
                    droop_start_time = None

                cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # --- 1. EYES ---
                l_crop, l_box = get_eye_crop(frame, landmarks, LEFT_EYE_IDXS)
                r_crop, r_box = get_eye_crop(frame, landmarks, RIGHT_EYE_IDXS)

                score_eye_l = predict(eye_model, l_crop)
                score_eye_r = predict(eye_model, r_crop)

                is_drowsy = (score_eye_l < EYE_THRESHOLD) and (score_eye_r < EYE_THRESHOLD)

                c_l = (0, 0, 255) if score_eye_l < EYE_THRESHOLD else (0, 255, 0)
                c_r = (0, 0, 255) if score_eye_r < EYE_THRESHOLD else (0, 255, 0)

                cv2.rectangle(frame, (l_box[0], l_box[1]), (l_box[2], l_box[3]), c_l, 2)
                cv2.rectangle(frame, (r_box[0], r_box[1]), (r_box[2], r_box[3]), c_r, 2)

                # --- 2. YAWN ---
                mouth_crop, mouth_box = get_mouth_crop(frame, landmarks)

                score_yawn = predict(yawn_model, mouth_crop)
                is_yawning = score_yawn > YAWN_THRESHOLD

                m_color = (0, 255, 255) if is_yawning else (200, 200, 200)
                cv2.rectangle(frame, (mouth_box[0], mouth_box[1]), (mouth_box[2], mouth_box[3]), m_color, 2)

                cv2.putText(frame, f"L:{score_eye_l:.2f}", (l_box[0], l_box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_l, 1)
                cv2.putText(frame, f"R:{score_eye_r:.2f}", (r_box[0], r_box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_r, 1)
                cv2.putText(frame, f"Y:{score_yawn:.2f}", (mouth_box[0], mouth_box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, m_color, 1)

                # --- PRIORITY (UPDATED ONLY THIS PART) ---
                if head_drop_active:
                    play_loop(HEAD_SOUND)

                elif is_yawning:
                    main_alert_text = f"YAWN DETECTED!"
                    main_alert_color = (0, 255, 255)
                    play_loop(YAWN_SOUND)

                elif is_drowsy:
                    main_alert_text = "DROWSINESS DETECTED!"
                    main_alert_color = (0, 0, 255)
                    play_loop(ALERT_SOUND)

                else:
                    stop_sound()

        text_size = cv2.getTextSize(main_alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, main_alert_text, (text_x, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, main_alert_color, 3)

        cv2.imshow('Driver Monitoring System', frame)

        if cv2.waitKey(delay_ms) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
