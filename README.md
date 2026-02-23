# SurgiPath

**AI-powered surgical training coach that uses computer vision to analyze instrument handling and hand technique in real time.**

SurgiPath watches medical students practice through a webcam, detects surgical instruments with a custom-trained YOLO model, tracks hand pose with MediaPipe, and coaches them on grip, stability, and procedural correctness — all running locally without cloud dependencies.

## Demo

<!-- Replace with a GIF, screenshot, or link to your demo video -->

*Coming soon — add a GIF or screenshot of the app in action here.*

## Features

- **Real-time instrument detection** — Custom YOLOv8n trained on 3,070 images recognizes 8 surgical tools (97.1% mAP50)
- **Hand technique analysis** — MediaPipe tracks 21 keypoints per hand to evaluate grip type, wrist stability, instrument angle, jerk-based motion smoothness, and economy of movement
- **Rule-based coaching engine** — JSON-defined rules per surgical phase with evidence gating and debounce to minimize false alerts
- **Voice coaching** — Text-to-speech alerts for technique issues (edge-tts with offline fallback)
- **WebRTC video pipeline** — YOLO + MediaPipe inference runs on a callback thread at 15-30 FPS, decoupled from UI rendering
- **Three-phase workflow** — Setup (tool checklist + calibration) → Practice (real-time coaching) → Report (mastery score + error log)
- **Demo mode** — Full walkthrough without a camera using synthetic detections
- **Optional AI procedure generation** — Gemini API can generate training plans from text input (not required to run)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit with custom CSS |
| Live video | WebRTC (streamlit-webrtc) + OpenCV fallback |
| Instrument detection | YOLOv8n (Ultralytics), custom-trained |
| Hand tracking | MediaPipe HandLandmarker (21 keypoints × 2 hands) |
| Coaching engine | JSON rule definitions + evidence-gated evaluation |
| Voice alerts | edge-tts / gTTS / pyttsx3 (3-tier fallback) |
| Data validation | Pydantic |
| AI procedure gen | Google Gemini API *(optional, not required)* |

## Installation

### Prerequisites

- Python 3.10+
- Webcam (optional — demo mode works without one)

### Steps

```bash
git clone https://github.com/anle0429/Panacea.git
cd Panacea
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Model files

Download and place in the `models/` directory:

| File | Size | Source |
|------|------|--------|
| `best.pt` | ~5 MB | Custom YOLOv8n (from training run) |
| `hand_landmarker.task` | ~8 MB | [MediaPipe](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) |

### Gemini API key (optional)

Only needed for AI procedure generation. Create `.env`:

```
GOOGLE_API_KEY=your-key
```

Everything else runs fully offline.

## Usage

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Quick demo (2 minutes)

1. Turn on **Demo Mode** → tool checklist auto-passes
2. **Begin Lab Session** → camera opens
3. Show your hands → technique monitor displays grip, stability, angle in real time
4. **End Lab Session** → mastery score + error report

## Project Structure

```
SurgiPath/
├── app.py                  Orchestrator (sidebar, video routing, tab routing)
├── brain.py                Optional Gemini AI integration
├── requirements.txt
│
├── ui/                     UI modules
│   ├── setup.py            Setup tab: calibration, tool checklist
│   ├── practice.py         Practice tab: coaching, technique monitor
│   ├── report.py           Report tab: mastery score, error details
│   ├── video.py            Video pipeline: WebRTC, display, demo
│   ├── components.py       Reusable HTML components
│   ├── helpers.py          Prompt formatting, performance stats
│   └── tts.py              Text-to-speech (3-tier fallback)
│
├── src/                    Backend modules
│   ├── detector.py         YOLO inference and drawing
│   ├── hands.py            MediaPipe hand detection + technique analysis
│   ├── rules.py            Phase-specific rule evaluation with debounce
│   ├── evidence.py         Detection stability tracking
│   ├── state.py            PRE_OP / INTRA_OP / POST_OP state machine
│   ├── constants.py        Config defaults and session state keys
│   ├── logger.py           Event logging
│   └── utils.py            Recipe loader
│
├── models/                 ML models (not committed)
├── recipes/                Training scenario definitions (JSON)
├── styles/                 Custom CSS theme
└── tools/                  Grip classifier training utilities
```

## Model Performance

| Metric | Value |
|--------|-------|
| mAP50 | 97.1% |
| mAP50-95 | 92.0% |
| Precision | 97.2% |
| Recall | 92.7% |
| Training images | 3,070 |
| Validation images | 344 |
| Classes | 8 (scalpel, forceps, needle_holder, scissors, clamp, suction, gauze, syringe) |
| Epochs | 50 |

The YOLO model was fine-tuned from a pretrained surgical tool detection model by [shaniangel/surgical_tool_detection](https://github.com/shaniangel/surgical_tool_detection) and trained on a merged dataset from multiple sources (see [Dataset Citations](#dataset-citations)).

## Dataset Citations

**Surgical Tools Dataset** (Roboflow)
> Mahmudul Hasan Ratul. *Surgical Tools Dataset*. Roboflow Universe, 2024.
> https://universe.roboflow.com/mahmudul-hasan-ratul/surgical-tools-v1ozl

**Surgical Dataset** (Roboflow)
> Deva. *Surgical Dataset*. Roboflow Universe, 2022.
> https://universe.roboflow.com/deva-9h2aw/surgical-jhjsm

**JIGSAWS — JHU-ISI Gesture and Skill Assessment Working Set**
> Yixin Gao, S. Swaroop Vedula, Carol E. Reiley, Narges Ahmidi, et al. *The JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS): A Surgical Activity Dataset for Human Motion Modeling.* M2CAI — MICCAI Workshop, 2014.
> https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/

> Narges Ahmidi, Lingling Tao, Shahin Sefati, Yixin Gao, et al. *A Dataset and Benchmarks for Segmentation and Recognition of Gestures in Robotic Surgery.* IEEE Transactions on Biomedical Engineering, 2017.

**Pretrained YOLO weights**
> shaniangel. *Surgical Tool Object Detection System.* GitHub, 2024.
> https://github.com/shaniangel/surgical_tool_detection

## License

MIT License.

## Acknowledgments

- **Kushal Shrestha** — collaborator (https://github.com/Kushal-Shr)
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8n
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — hand landmark detection
- [Streamlit](https://streamlit.io/) — web app framework
- [shaniangel/surgical_tool_detection](https://github.com/shaniangel/surgical_tool_detection) — pretrained surgical tool YOLO model
- [Roboflow](https://roboflow.com/) — dataset hosting and annotation tools
