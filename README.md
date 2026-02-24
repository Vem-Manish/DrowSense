# 🚗 DrowSense – Driver Drowsiness Detection System

DrowSense is a real-time driver drowsiness detection system that uses **Deep Learning and Computer Vision** to identify fatigue based on:

- 👁 Eye closure  
- 😮 Yawning  
- 🧠 Head pose (drooping detection)  

The system uses **MobileNetV3 models** along with real-time video processing to provide **instant alerts** for driver safety.

---

## 🔥 Features

- Real-time webcam-based detection  
- Eye state classification (Open / Closed)  
- Yawn detection  
- Head pose estimation  
- Priority-based alert system  
- Lightweight and efficient (runs on CPU)

---

## 📁 Project Structure

```bash
DrowSense/
│── main.py                # Main real-time detection script
│── train_eye.py           # Eye model training
│── train_yawn.py          # Yawn model training
│── mobilenet_v3_best.pth  # Eye model weights
│── yawn_model_2.pth       # Yawn model weights
│── README.md
