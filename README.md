# GestureWrite ✍️

> Touch-free virtual air canvas and real-time handwriting recognition using computer vision, gesture tracking, and machine learning.

GestureWrite enables natural air-writing and drawing in front of a standard webcam. By tracking hand landmarks in real time, users can write, sketch, erase, select colors, and automatically transcribe air-written characters into text copied directly to the clipboard.

---

## 🚀 Features

- **Real-Time Hand Tracking**: Powered by MediaPipe Hand Landmarker with gesture hysteresis and 3D knuckle/fingertip analysis.
- **Air-Canvas Engine**: Mask-based virtual drawing layer with smooth Bézier/stroke interpolation and stroke-level undo.
- **Intuitive Gesture Controls**:
  - 👆 **Index finger extended**: Draw / write on canvas.
  - ✌️ **Index + Middle fingers extended**: Hover mode / select colors / browse HUD tools.
  - ✋ **Open hand / Fist gestures**: Eraser and quick controls.
- **Hybrid Recognition Engine**:
  - Custom trained ML classifier (SVM / scikit-learn) with letter segmentation.
  - Windows Native OCR (`winocr`) integration for character transcription.
  - Automatic clipboard synchronization for transcribed text.
- **Dataset Collection & Training Pipeline**: Built-in interactive dataset collector and training scripts for custom alphabet models.
- **Adaptive HUD Overlay**: Responsive UI supporting camera resolutions from 640x480 up to 4K.

---

## 🛠️ Project Structure

```text
├── canvas_manager.py         # Canvas rendering, stroke buffers, and blending
├── hand_tracker.py           # MediaPipe hand detection and gesture state machine
├── ui_overlay.py             # HUD widgets, color palettes, and visual feedback
├── ocr_engine.py             # Windows OCR integration
├── main.py                   # Main application entry point
├── collect_dataset.py        # Interactive air-writing dataset collector
├── run_air_writing.bat       # Quick launch script for air-writing
├── collect_dataset.bat       # Quick launch script for dataset collection
├── requirements.txt          # Python dependencies
├── models/
│   └── hand_landmarker.task  # MediaPipe Hand Landmarker model
├── recognition/
│   ├── dataset_manager.py    # Dataset loader and preprocessor
│   ├── feature_extractor.py  # Spatial and stroke feature extraction
│   ├── letter_segmenter.py   # Temporal pause and spatial clustering
│   ├── recognizer.py         # Character classification pipeline
│   ├── train_baseline.py     # SVM model training script
│   └── model/
│       └── character_svm.joblib # Pretrained character recognition model
└── data/
    └── dataset/              # Collected character sample datasets
```

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/25f1002767/GestureWrite.git
cd GestureWrite
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Quick Launch
On Windows, simply run the batch script:
```cmd
run_air_writing.bat
```
Or execute via Python:
```bash
python main.py
```

### Keyboard Shortcuts
- `C`: Clear canvas
- `U`: Undo last stroke
- `S`: Save snapshot of current canvas
- `R`: Trigger character recognition
- `Q` or `ESC`: Exit application

---

## 📚 Dataset Collection & Model Training

To record your own handwriting samples and retrain the classifier:

1. **Collect Samples**:
   ```cmd
   collect_dataset.bat
   ```
   Follow the on-screen prompts to write sample letters in the air.

2. **Train Model**:
   ```bash
   python recognition/train_baseline.py
   ```
   This generates an updated `character_svm.joblib` model under `recognition/model/`.

---

## 📄 License
This project is open-source under the MIT License.
