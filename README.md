# DrowSense 🚗💤
### Real-time Driver Drowsiness Detection 

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/manishvem/DrowSense)

---

## What is this?

DrowSense is a real-time drowsiness detection system that uses your webcam to monitor whether a driver is falling asleep at the wheel. It detects three danger signals simultaneously:

- 😴 **Eye closure** — are the driver's eyes drooping shut?
- 🥱 **Yawning** — is the driver showing signs of fatigue through yawning?
- 📉 **Head drop** — is the driver's head nodding forward?

When any of these are detected, the dashboard triggers an alert. 

---

## How it works

```
Webcam Feed (Browser)
        │
        ▼
  Base64 Frame → WebSocket → FastAPI Server
                                    │
                                    ▼
                          MediaPipe Face Mesh
                          (468 facial landmarks)
                                    │
                        ┌───────────┼───────────┐
                        ▼           ▼           ▼
                   Left Eye     Right Eye     Mouth
                   Crop         Crop          Crop
                        │           │           │
                        └─── MobileNetV3 ───────┘
                             Eye Model    Yawn Model
                                    │
                                    ▼
                          Drowsiness Logic +
                          Head Pose (PnP)
                                    │
                                    ▼
                    Annotated Frame + Telemetry
                    ← WebSocket back to Browser
```

The browser captures a frame, sends it to the server via WebSocket, the server runs all the ML inference, draws the detection boxes, and sends the annotated frame + telemetry data back — all in real time.

---

## The Models

Two separate MobileNetV3-Large models handle detection, each fine-tuned for its specific task.

### Eye State Model (`mobilenet_v3_best.pth`)
Trained to classify cropped eye regions as **open** or **closed**.

- Architecture: MobileNetV3-Large (ImageNet pretrained) with a custom binary classifier head
- Training: Transfer learning with frozen backbone, 10 epochs
- Input: 224×224 cropped eye region

**Training Results:**

<img width="1500" height="500" alt="training_metrics" src="https://github.com/user-attachments/assets/5bb48672-5203-4a30-a175-c3d26d1e1aa8" />



| Metric | Score |
|---|---|
| Validation Accuracy | 97% |
| Convergence | Epoch 8 |

---

### Yawn Detection Model (`yawn_model_2.pth`)
Trained to classify cropped mouth regions as **yawning** or **not yawning**.

- Architecture: MobileNetV3-Large with a custom deep head (Linear → ReLU → Dropout → Linear)
- Training: 20 epochs, class-weighted loss to handle dataset imbalance, partial fine-tuning of last 3 feature layers
- Loss function: BCEWithLogitsLoss with `pos_weight` calculated from class ratio
- Input: 224×224 cropped mouth region

**Training Results:**

<img width="1500" height="500" alt="training_graphs_yawn" src="https://github.com/user-attachments/assets/25129ecb-f778-4989-b5a9-cf6bed055a59" />



| Metric | Score |
|---|---|
| Validation Accuracy | 98% |
| Validation Precision | ~0.99 |
| Validation Recall | ~0.99 |

The yawn model's hidden layer activations are also extracted live and visualized as a 3D neural network in the dashboard UI.

---

## Drowsiness Logic

A detection event is triggered based on three independent checks:

| Signal | Condition | Threshold |
|---|---|---|
| **Drowsy** | Both eyes below confidence threshold | Eye score < 0.40 |
| **Yawning** | Mouth open with high confidence | Yawn score > 0.90 |
| **Head Drop** | Head pitch drops below calibrated baseline | >20° drop sustained for 1.5s |

On startup, the system runs a **30-frame calibration** to establish the driver's neutral head position before monitoring begins.

Predictions are also **temporally smoothed** (exponential moving average, α=0.8) to prevent flickering alerts from single noisy frames.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | PyTorch, MobileNetV3-Large |
| Face Tracking | MediaPipe Face Mesh (468 landmarks) |
| Head Pose | OpenCV solvePnP |
| Backend | FastAPI + WebSockets |
| Frontend | Vanilla JS, Three.js (3D NN visualization) |
| Deployment | Docker, Hugging Face Spaces |

---

## Running Locally

**Prerequisites:** Python 3.10, the two `.pth` model files

```bash
# Clone the repo
git clone https://github.com/Vem-Manish/DrowSense.git
cd DrowSense

# Install dependencies (CPU-only torch)
pip install -r requirements.txt

# Start the server
python server.py
```

Then open `http://localhost:7860` in your browser and click **Initialize**.

> **Note:** The `.pth` model files are tracked via Git LFS. Make sure you have Git LFS installed (`git lfs install`) before cloning, or download them separately from the repo.

---

## Repository Structure

```
DrowSense/
├── server.py              # FastAPI server — WebSocket handling, inference pipeline
├── index.html             # Frontend dashboard (single file, no framework)
├── mobilenet_v3.py        # Training script — Eye state model
├── train_yawn.py          # Training script — Yawn detection model
├── mobilenet_v3_best.pth  # Trained eye model weights
├── yawn_model_2.pth       # Trained yawn model weights
├── requirements.txt       # Python dependencies
└── Dockerfile             # Container config for deployment
```

---

## Limitations & Known Issues

- Works best in **good lighting** — low light degrades MediaPipe landmark accuracy
- Calibration assumes the driver sits **facing forward** at startup
- Currently supports **single driver** only (one face at a time)
- Inference runs on CPU — there's a small latency (~100–200ms per frame) depending on server load

---

## Author

**Manish Vem** — built as part of an ML project exploring real-world deployment of computer vision models.

[![HuggingFace](https://img.shields.io/badge/Try%20it%20Live-DrowSense-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/manishvem/DrowSense)
