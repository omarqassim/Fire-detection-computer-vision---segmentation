# 🔥 Fire Detection & Segmentation System (YOLOv8 Nano)

A web-based application for detecting and segmenting fire instances in images using YOLOv8 Nano Segmentation model. The system provides real-time visualization including masks, bounding boxes, and confidence scores.

---

## 🚀 Features

- 🔍 Fire Detection using YOLOv8 Nano
- 🎯 Instance Segmentation (pixel-level masks)
- 📦 Bounding Boxes + Labels
- 📊 Adjustable Confidence Threshold
- 🖼️ Upload or Use Demo Image
- 💾 Download Segmented Output
- ⚡ Fast and lightweight (Nano model)

---

## 🖥️ Demo UI Overview

- Original Image → shows input image  
- Segmented Result → shows detected fire with:
  - Masks
  - Bounding boxes
  - Confidence score  

- Settings Panel:
  - Confidence threshold slider
  - Toggle visualization (Masks / Boxes / Labels)

---

## 🧠 Model Used

- YOLOv8 Nano Segmentation (yolov8n-seg)
- Lightweight & optimized for real-time inference
- Detects fire regions and segments them accurately

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/fire-detection-segmentation.git

# Go to project directory
cd fire-detection-segmentation

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
python app.py
```

Then open your browser:

```
http://localhost:8501
```

---

## 📁 Project Structure

```
├── app.py
├── model/
├── utils/
├── assets/
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- YOLOv8 (Ultralytics)  
- OpenCV  

---

## 📊 Example Output

- Fire detected with confidence (e.g., 36.9%)
- Segmented region highlighted with colored mask
- Bounding box around detected fire

---

## ⚡ Future Improvements

- 🎥 Real-time video detection
- 🔔 Fire alert system
- ☁️ Cloud deployment
- 📱 Mobile-friendly UI

---

## 👨‍💻 Author

Omar Qassim  
Full Stack Developer | AI & Cybersecurity Enthusiast

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
