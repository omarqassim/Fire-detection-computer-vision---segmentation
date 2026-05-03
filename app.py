import cv2
from PIL import Image
from utils.segmentor import YOLOSegmentor
from utils.visualization import draw_segmentation
import os

def main():
    # Ensure assets and model exist
    model_path = os.path.join("model", "best.pt")
    image_path = os.path.join("assets", "demo.png")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print("Loading YOLO Segmentation Model...")
    segmentor = YOLOSegmentor(model_path)
    
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    
    # Run inference
    detections = segmentor.predict(image, conf_threshold=0.25)
    
    # Print results
    print(f"\n--- Detection Summary ---")
    print(f"Total instances detected: {len(detections)}")
    for i, d in enumerate(detections):
        box = [round(x, 2) for x in d['box']]
        print(f"Instance {i+1}: {d['class_name']} ({d['confidence']:.2f}) - Box: {box}")
    print("-------------------------\n")
    
    # Visualize
    img_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    annotated_img = draw_segmentation(img_np, detections)
    
    # Save output
    output_path = "output.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    main()
