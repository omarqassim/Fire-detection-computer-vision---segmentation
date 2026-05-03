# YOLOv11 Segmentation Deployment Scaffold

A production-ready project scaffold for deploying a YOLOv11 Segmentation model locally or on Streamlit Cloud.

> **Note:** This template is specifically built for YOLO segmentation models (using `task="segment"`). It will extract, process, and overlay segmentation masks alongside bounding boxes.

## 📂 Folder Structure

```
yolo-seg-deployment/
│
├── app.py                 # Local testing CLI script
├── streamlit_app.py       # Main Streamlit web application
├── requirements.txt       # Python dependencies
├── packages.txt           # OS-level dependencies (for Streamlit Cloud/Linux)
├── README.md              # This file
│
├── model/                 # Put your trained model here
│   ├── best.pt            # Your YOLOv11 segmentation weights
│   └── labels.txt         # Your class labels (one per line)
│
├── utils/                 # Core functionality
│   ├── segmentor.py       # YOLO inference logic
│   └── visualization.py   # OpenCV drawing functions
│
└── assets/
    └── demo.png           # Fallback image for testing
```

## 🚀 Setup & Installation

1. **Install Dependencies:**
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Setup:**
   - Place your trained YOLOv11 segmentation model inside the `model/` directory and name it `best.pt`.
   - Ensure `model/labels.txt` contains your class names, one per line. If it is missing, the app will try to extract the names from the `.pt` file directly.

## 💻 How to Run Locally

### 1. Local CLI Test
To test the inference engine locally without a GUI, run:
```bash
python app.py
```
This will process `assets/demo.png`, print the detection summary to the console, and save the result as `output.jpg`.

### 2. Streamlit Web App
To launch the interactive web application, run:
```bash
streamlit run streamlit_app.py
```
This will open a clean UI in your browser where you can upload images, adjust confidence thresholds, and toggle visualizations.

## ☁️ Streamlit Cloud Deployment

This project is perfectly optimized for Streamlit Cloud.
1. Push this folder to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app" and point it to your repository.
4. Set the "Main file path" to `streamlit_app.py`.
5. Deploy!

> **Why `packages.txt`?**
> The `packages.txt` file ensures `libgl1` is installed on the underlying Linux environment in Streamlit Cloud, which is required by `opencv-python-headless`.

## 🛠️ Features
- Modular design with separated inference and visualization logic.
- Graceful handling of missing masks.
- Dynamic color assignment per class based on HSV space.
- Production-grade UI with loading states and structured tables.
