import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
from PIL import Image
import numpy as np
import time
import base64
import json
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# =========================================================
# CONFIG & GLOBALS
# =========================================================

EYE_MODEL_PATH = "mobilenet_v3_best.pth"
YAWN_MODEL_PATH = "yawn_model_2.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Server running on: {DEVICE}")

EYE_THRESHOLD = 0.40
YAWN_THRESHOLD = 0.90
DROOP_THRESHOLD_RELATIVE = -20.0
TIME_THRESHOLD = 1.5
CALIBRATION_FRAMES = 30
JPEG_QUALITY = 85

live_activations = {}


def get_activation(name):
    def hook(model, input, output):
        live_activations[name] = output.detach().cpu().numpy()[0]
    return hook


# =========================================================
# MODELS & MEDIAPIPE INITIALIZATION
# =========================================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

LEFT_EYE_IDXS  = [33, 133, 159, 145]
RIGHT_EYE_IDXS = [362, 263, 386, 374]
MOUTH_IDXS     = [13, 14, 61, 291]

model_points = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_eye_model(path):
    if not os.path.exists(path):
        print(f"[ERROR] Eye model not found at: {path}", file=sys.stderr)
        sys.exit(1)
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Eye model loaded from {path}")
    return model


def load_yawn_model(path):
    if not os.path.exists(path):
        print(f"[ERROR] Yawn model not found at: {path}", file=sys.stderr)
        sys.exit(1)
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Sequential(
        nn.Linear(model.classifier[3].in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1)
    )
    model.classifier[3][1].register_forward_hook(get_activation('yawn_hidden'))
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Yawn model loaded from {path}")
    return model


# Load models at startup — exit with clear error if files are missing
eye_model  = load_eye_model(EYE_MODEL_PATH)
yawn_model = load_yawn_model(YAWN_MODEL_PATH)


# =========================================================
# HELPERS
# =========================================================

def smooth(old, new, alpha=0.8):
    return alpha * old + (1 - alpha) * new


def predict(model, crop):
    if crop.size == 0:
        return 0.0
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
    return torch.sigmoid(out).item()


def crop_region(frame, landmarks, idxs, pad_rx=0.40, pad_ry=0.60):
    h, w, _ = frame.shape
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idxs]
    x0, x1 = min(c[0] for c in coords), max(c[0] for c in coords)
    y0, y1 = min(c[1] for c in coords), max(c[1] for c in coords)
    ew = x1 - x0
    px, py = int(ew * pad_rx), int(ew * pad_ry)
    x0, x1 = max(0, x0 - px), min(w, x1 + px)
    y0, y1 = max(0, y0 - py), min(h, y1 + py)
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


def get_head_pose(landmarks, fw, fh):
    img_pts = np.array([
        (landmarks[1].x   * fw, landmarks[1].y   * fh),
        (landmarks[152].x * fw, landmarks[152].y * fh),
        (landmarks[33].x  * fw, landmarks[33].y  * fh),
        (landmarks[263].x * fw, landmarks[263].y * fh),
        (landmarks[61].x  * fw, landmarks[61].y  * fh),
        (landmarks[291].x * fw, landmarks[291].y * fh)
    ], dtype=np.float64)
    cam_mat = np.array([[fw, 0, fw / 2], [0, fw, fh / 2], [0, 0, 1]], dtype=np.float64)
    _, rot_vec, trans_vec = cv2.solvePnP(model_points, img_pts, cam_mat, np.zeros((4, 1)))
    rmat, _ = cv2.Rodrigues(rot_vec)
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, trans_vec)))
    pitch = angles.flatten()[0]
    return (180 - pitch) if pitch > 0 else (-180 - pitch)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="DrowSense Web")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# FIX: Serve index.html explicitly at root — the StaticFiles mount
# is removed because mounting StaticFiles on "/" shadows this route.
# Static assets (wav files) are served via the /static mount below.
@app.get("/")
async def root():
    return FileResponse("index.html")


# Serve static files (danger.wav, yawn.wav, etc.) under /static
app.mount("/static", StaticFiles(directory="."), name="static")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Per-connection state — fully isolated per user
    state = {
        "calibrating":       True,
        "calibration_count": 0,
        "pitch_history":     [],
        "baseline_pitch":    0.0,
        "prev_eye_l":        0.0,
        "prev_eye_r":        0.0,
        "prev_yawn":         0.0,
        "prev_pitch":        0.0,
        "droop_start":       None
    }

    try:
        while True:
            # Receive Base64 image from browser
            data = await ws.receive_text()
            if not data.startswith("data:image"):
                continue

            # Decode image
            img_data = base64.b64decode(data.split(",")[1])
            np_arr   = np.frombuffer(img_data, np.uint8)
            frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            payload = {
                "status": "active", "eye_l": 0.0, "eye_r": 0.0,
                "yawn": 0.0, "pitch": 0.0, "face_detected": False,
                "nn_acts": [], "calib_progress": 0
            }
            fh, fw, _ = frame.shape

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                payload["face_detected"] = True

                pitch = smooth(state["prev_pitch"], get_head_pose(lm, fw, fh))
                state["prev_pitch"] = pitch
                payload["pitch"] = round(float(pitch), 1)

                left_crop,  left_box  = crop_region(frame, lm, LEFT_EYE_IDXS)
                right_crop, right_box = crop_region(frame, lm, RIGHT_EYE_IDXS)
                mouth_crop, mouth_box = crop_region(frame, lm, MOUTH_IDXS, pad_rx=0.40, pad_ry=0.60)

                sl = smooth(state["prev_eye_l"], predict(eye_model,  left_crop))
                sr = smooth(state["prev_eye_r"], predict(eye_model,  right_crop))
                sy = smooth(state["prev_yawn"],  predict(yawn_model, mouth_crop))

                state["prev_eye_l"], state["prev_eye_r"], state["prev_yawn"] = sl, sr, sy
                payload["eye_l"] = round(float(sl), 3)
                payload["eye_r"] = round(float(sr), 3)
                payload["yawn"]  = round(float(sy), 3)

                # Neural network activation extraction
                raw_tensors        = live_activations.get('yawn_hidden', np.zeros(128))
                compressed_tensors = np.max(raw_tensors.reshape(-1, 4), axis=1)
                max_val            = np.max(compressed_tensors) if np.max(compressed_tensors) > 0 else 1
                payload["nn_acts"] = [round(float(x), 3) for x in (np.clip(compressed_tensors, 0, None) / max_val)]

                # Calibration & drowsiness logic
                if state["calibrating"]:
                    payload["status"]        = "calibrating"
                    payload["calib_progress"] = int((state["calibration_count"] / CALIBRATION_FRAMES) * 100)
                    state["pitch_history"].append(pitch)
                    state["calibration_count"] += 1

                    if state["calibration_count"] >= CALIBRATION_FRAMES:
                        state["baseline_pitch"] = sum(state["pitch_history"]) / len(state["pitch_history"])
                        state["calibrating"]    = False
                else:
                    is_drowsy  = (sl < EYE_THRESHOLD and sr < EYE_THRESHOLD)
                    is_yawning = (sy > YAWN_THRESHOLD)
                    head_drop  = False

                    if pitch < (state["baseline_pitch"] + DROOP_THRESHOLD_RELATIVE):
                        if state["droop_start"] is None:
                            state["droop_start"] = time.time()
                        if (time.time() - state["droop_start"]) >= TIME_THRESHOLD:
                            head_drop = True
                    else:
                        state["droop_start"] = None

                    if head_drop:
                        payload["status"] = "head_drop"
                    elif is_yawning:
                        payload["status"] = "yawning"
                    elif is_drowsy:
                        payload["status"] = "drowsy"

                # Draw corner brackets on detected regions
                green, orange, red = (0, 255, 120), (0, 180, 255), (0, 0, 255)
                ec = red if (sl < EYE_THRESHOLD and sr < EYE_THRESHOLD) else green
                mc = orange if sy > YAWN_THRESHOLD else green

                for box, color in [(left_box, ec), (right_box, ec), (mouth_box, mc)]:
                    x0, y0, x1, y1 = box
                    l, t = 18, 2
                    cv2.line(frame, (x0, y0), (x0 + l, y0), color, t)
                    cv2.line(frame, (x0, y0), (x0, y0 + l), color, t)
                    cv2.line(frame, (x1, y0), (x1 - l, y0), color, t)
                    cv2.line(frame, (x1, y0), (x1, y0 + l), color, t)
                    cv2.line(frame, (x0, y1), (x0 + l, y1), color, t)
                    cv2.line(frame, (x0, y1), (x0, y1 - l), color, t)
                    cv2.line(frame, (x1, y1), (x1 - l, y1), color, t)
                    cv2.line(frame, (x1, y1), (x1, y1 - l), color, t)

            else:
                payload["status"]      = "no_face"
                state["droop_start"]   = None

            # Encode processed frame and send back to browser
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            payload["frame"]   = base64.b64encode(buffer).decode('utf-8')
            payload["running"] = True

            await ws.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
