import os
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

class YOLOSegmentor:
    def __init__(self, model_path):
        """
        Initialize the YOLO Segmentation model.
        Args:
            model_path (str): Path to the best.pt model.
        """
        # Automatically detect device (CPU/GPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model using YOLO with task="segment"
        self.model = YOLO(model_path, task="segment")
        self.model.to(self.device)
        
        # Load labels dynamically
        labels_path = os.path.join(os.path.dirname(model_path), "labels.txt")
        self.labels = {}
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    class_name = line.strip()
                    if class_name:
                        self.labels[idx] = class_name
        else:
            # Fallback to model's internal names if labels.txt doesn't exist
            self.labels = self.model.names

    def predict(self, image, conf_threshold=0.25):
        """
        Run inference on an image and return structured detections.
        
        Args:
            image (PIL.Image or np.ndarray): Input image.
            conf_threshold (float): Confidence threshold for detections.
            
        Returns:
            list: List of dictionaries containing detection information.
        """
        # Convert PIL to numpy if necessary
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        orig_h, orig_w = image.shape[:2]
        
        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        # Process results
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy()
            
            for i in range(len(boxes)):
                class_id = class_ids[i]
                confidence = float(confidences[i])
                box = boxes[i].tolist()
                
                # Retrieve mask and resize to match original image size
                mask = masks[i]
                mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized * 255).astype(np.uint8)
                
                class_name = self.labels.get(class_id, f"Class {class_id}")
                
                detections.append({
                    "box": box,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "mask": mask_binary
                })
                
        return detections
