import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io

from utils.segmentor import YOLOSegmentor
from utils.visualization import draw_segmentation, generate_colors

st.set_page_config(
    page_title="YOLOv8 Nano Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.5rem;
        color: #fff;
        background-color: #3B82F6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading YOLOv8 Nano Segmentation Model...")
def load_model():
    model_path = os.path.join("model", "best.pt")
    if not os.path.exists(model_path):
        return None
    return YOLOSegmentor(model_path)

with st.sidebar:
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Reverse.png", width=200)
    st.markdown("### Settings")
    
    conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=0.95, value=0.25, step=0.05)
    
    st.markdown("### Visualization")
    show_masks = st.checkbox("Show Masks", value=True)
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    show_labels = st.checkbox("Show Labels", value=True)
    
    st.markdown("### Legend")
    legend_container = st.empty()

st.markdown('<p class="main-header">🎯YOLOv8 Nano Object Segmentation</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image to perform high-precision instance segmentation.</p>', unsafe_allow_html=True)

segmentor = load_model()

if segmentor is None:
    st.error("⚠️ Model not found! Please ensure 'model/best.pt' exists.")
    st.stop()

unique_classes = len(segmentor.labels)
if unique_classes > 0:
    colors = generate_colors(max(segmentor.labels.keys()) + 1)
    legend_html = "<ul>"
    for class_id, class_name in segmentor.labels.items():
        color_rgb = f"rgb({colors[class_id][2]}, {colors[class_id][1]}, {colors[class_id][0]})"
        legend_html += f'<li><span style="display:inline-block; width:12px; height:12px; background-color:{color_rgb}; border-radius:50%; margin-right:8px;"></span>{class_name}</li>'
    legend_html += "</ul>"
    legend_container.markdown(legend_html, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])

# --- Image Selection Logic ---
image_to_process = None

if uploaded_file is not None:
    try:
        image_to_process = Image.open(uploaded_file)
        st.session_state["use_demo"] = False
    except Exception as e:
        st.error(f"Error loading image: {e}")
else:
    demo_path = os.path.join("assets", "demo.png")
    if os.path.exists(demo_path):
        st.info("No image uploaded. Click the button below to use the demo image.")
        if st.button("🖼️ Use Demo Image", use_container_width=True):
            st.session_state["use_demo"] = True

    if st.session_state.get("use_demo"):
        image_to_process = Image.open(demo_path)

# --- Run Segmentation ---
if image_to_process is not None:
    img_np = np.array(image_to_process.convert('RGB'))
    
    if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
        
        with st.spinner("Performing segmentation inference..."):
            detections = segmentor.predict(img_np, conf_threshold=conf_threshold)
            
        if len(detections) > 0:
            st.markdown(f'<div class="badge">Instances Detected: {len(detections)}</div>', unsafe_allow_html=True)
            
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            annotated_bgr = draw_segmentation(
                img_bgr, 
                detections, 
                show_masks=show_masks,
                show_boxes=show_boxes,
                show_labels=show_labels
            )
            
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Original Image")
                st.image(img_np, use_container_width=True)
            with col2:
                st.markdown("### Segmented Result")
                st.image(annotated_rgb, use_container_width=True)
                
            result_img = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            result_img.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Download Segmented Image",
                data=byte_im,
                file_name="segmented_output.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
                
            st.markdown("### Detection Summary")
            
            table_data = []
            for d in detections:
                box_str = f"[{int(d['box'][0])}, {int(d['box'][1])}, {int(d['box'][2])}, {int(d['box'][3])}]"
                table_data.append({
                    "Class": d['class_name'],
                    "Confidence": f"{d['confidence']:.2%}",
                    "Bounding Box": box_str
                })
                
            st.table(table_data)
            
        else:
            st.warning("No instances detected. Try lowering the confidence threshold.")
            st.markdown("### Original Image")
            st.image(img_np, use_container_width=True)
