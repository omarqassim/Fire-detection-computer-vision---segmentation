import cv2
import numpy as np

def generate_colors(num_classes):
    """
    Generate distinct colors for each class using HSV color space.
    """
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        # S=200, V=255 for bright distinct colors
        hsv_color = np.uint8([[[hue, 200, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(bgr_color.tolist())
    return colors

def draw_segmentation(image, detections, alpha=0.45, show_masks=True, show_boxes=True, show_labels=True):
    """
    Draw segmentation masks, bounding boxes, and labels on the image.
    
    Args:
        image (np.ndarray): Original image (RGB or BGR).
        detections (list): List of dictionaries from YOLOSegmentor.predict().
        alpha (float): Transparency of the masks.
        show_masks (bool): Whether to show masks.
        show_boxes (bool): Whether to show bounding boxes.
        show_labels (bool): Whether to show labels.
        
    Returns:
        np.ndarray: Annotated image.
    """
    # Create a copy of the image to draw on
    annotated_image = image.copy()
    overlay = image.copy()
    
    # Determine number of unique classes for color generation
    unique_classes = set([d['class_id'] for d in detections])
    max_class_id = max(unique_classes) if unique_classes else 0
    colors = generate_colors(max_class_id + 1)
    
    # Adaptive thickness based on image size
    h, w = image.shape[:2]
    thickness = max(1, int(min(w, h) / 500))
    font_scale = max(0.4, min(w, h) / 1000)
    
    for det in detections:
        color = colors[det['class_id']]
        
        # 1. Draw Masks
        if show_masks:
            mask = det['mask']
            # Find where mask is white
            mask_indices = mask == 255
            # Apply color to the overlay
            overlay[mask_indices] = color
            
        # 2. Draw Bounding Boxes
        if show_boxes:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
        # 3. Draw Labels
        if show_labels:
            x1, y1 = map(int, det['box'][:2])
            label = f"{det['class_name']}: {det['confidence'] * 100:.1f}%"
            
            # Text size for background rectangle
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1))
            
            # Draw background rectangle
            cv2.rectangle(annotated_image, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
            
            # Draw text (white with dark outline for readability)
            text_x = x1
            text_y = y1 - 5
            
            # Outline
            cv2.putText(annotated_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            # Inner text
            cv2.putText(annotated_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

    # Blend the overlay with the original image for transparent masks
    if show_masks:
        cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)
        
    return annotated_image
