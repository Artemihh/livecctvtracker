# 🎥 Live CCTV Person Tracking & Re‑Identification System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

A **real‑time CCTV-based person detection, tracking, and re‑identification system**
built using **YOLO, DeepSORT, and OSNet**, with a **FastAPI backend** and a
modern **React + Vite frontend**.

The system allows uploading a **probe image** from the frontend and searches
for that person in a **live CCTV camera feed**.

---

## ✨ Features

- 📷 Live CCTV feed using OpenCV (MJPEG stream)
- 🧍 Person detection with YOLO
- 🔁 Multi-person tracking using DeepSORT
- 🧠 Person Re‑Identification (ReID) using OSNet
- 📸 Upload probe image from frontend
- 🔍 Real-time matching in live feed
- ⚡ FastAPI backend
- 🎨 React + Vite frontend
- 🐳 Docker & Docker Compose support

---

## 🧠 System Architecture

Camera (OpenCV)
↓
YOLO Detector
↓
DeepSORT Tracker
↓
OSNet Re‑Identification
↓
Match Result → Frontend

css
Copy code

### Probe Image Flow

Frontend Upload
↓
FastAPI Backend
↓
Probe Embedding Saved
↓
Worker Matches in Live Feed
↓
Match Result Returned

yaml
Copy code

---

## 🗂 Project Structure

live-cctv-tracker/
├── backend/ # FastAPI backend
├── cv_pipeline/ # Camera, tracking & ReID worker
├── frontend/ # React frontend
├── screenshots/ # UI & result images
├── demo/ # Demo GIF
└── README.md

yaml
Copy code

---

## 📸 Screenshots

Place screenshots inside the `screenshots/` folder.

screenshots/
├── frontend_ui.png
├── live_feed.png
└── match_result.png

yaml
Copy code

Example usage in this README:

![Frontend UI](screenshots/frontend_ui.png)
![Live Feed](screenshots/live_feed.png)
![Match Result](screenshots/match_result.png)

---

## 🎬 Demo GIF

Place a demo GIF inside the `demo/` folder.

demo/
└── demo.gif

yaml
Copy code

Example:

![Demo](demo/demo.gif)

---

## ⚙️ Installation (Manual Setup)

### 1️⃣ Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app
Backend runs at:

cpp
Copy code
http://127.0.0.1:8000
2️⃣ CV Pipeline (Worker)
bash
Copy code
cd cv_pipeline
python ingest_worker.py
Live MJPEG feed:

arduino
Copy code
http://127.0.0.1:5001/feed
3️⃣ Frontend (React)
bash
Copy code
cd frontend
npm install
npm run dev
Frontend runs at:

arduino
Copy code
http://localhost:5173
🐳 Docker Support
Run the entire system using Docker Compose
bash
Copy code
docker compose up --build
Example docker-compose.yml
yaml
Copy code
version: "3.9"

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"

  worker:
    build: ./cv_pipeline
    devices:
      - "/dev/video0:/dev/video0"

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
⚠️ Camera device mapping may differ on Windows.

🧪 How It Works
Camera captures live video

YOLO detects persons

DeepSORT assigns unique track IDs

OSNet extracts ReID embeddings

User uploads a probe image

System searches for the person in real-time

Match result is returned to the frontend

🎯 Use Cases
🏫 Campus & university surveillance

🏢 Office security systems

🛒 Retail analytics

🚪 Smart access control

🎥 Intelligent CCTV solutions

⚠️ Notes
Model weights (YOLO / OSNet) are NOT included in this repository

Download the weights separately before running

Tested on Windows + CUDA, works on CPU as well

🛠 Tech Stack
Backend: FastAPI, Python

Computer Vision: OpenCV, YOLO, DeepSORT, OSNet

Frontend: React, Vite

DevOps: Docker, Docker Compose

👨‍💻 Author
Indro and Ishaan
Computer Vision & AI Enthusiast
