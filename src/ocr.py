import cv2
import pytesseract
import numpy as np
import os
import re
import logging
from preprocess import preprocess_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.name == 'nt':  # for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def decode_predictions(scores, geometry, min_confidence):
    num_rows, num_cols = scores.shape[2:4]
    boxes = []
    confidences = []
    
    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0_data = geometry[0, 0, y]
        x1_data = geometry[0, 1, y]
        x2_data = geometry[0, 2, y]
        x3_data = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]
        
        for x in range(num_cols):
            if scores_data[x] < min_confidence:
                continue
            
            offset_x, offset_y = x * 4.0, y * 4.0
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            angle = angles_data[x]
            
            cos = np.cos(angle)
            sin = np.sin(angle)
            x1 = int(offset_x + (cos * x1_data[x]) + (sin * x2_data[x]))
            y1 = int(offset_y - (sin * x1_data[x]) + (cos * x2_data[x]))
            
            boxes.append([x1, y1, int(w), int(h)])
            confidences.append(float(scores_data[x]))
    
    return boxes, confidences

def detect_text_regions(image):
    orig_height, orig_width = image.shape[:2]
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    new_width, new_height = (320, 320)
    ratio_width = orig_width / float(new_width)
    ratio_height = orig_height / float(new_height)
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height), 
                                 (123.68, 116.78, 103.94), True, False)
    
    net = cv2.dnn.readNet("models/frozen_east_text_detection.pb")
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    
    boxes, confidences = decode_predictions(scores, geometry, min_confidence=0.5)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    text_regions = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            x = int(x * ratio_width)
            y = int(y * ratio_height)
            w = int(w * ratio_width)
            h = int(h * ratio_height)
            text_regions.append(image[y:y+h, x:x+w])
    
    return text_regions

def apply_ocr(image, config=''):
    try:
        return pytesseract.image_to_string(image, config=config)
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return ""

def clean_text(text):
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase and strip whitespace
    return text.lower().strip()

def extract_product_info(text):
    info = {
        'brand': None,
        'product_name': None,
        'quantity': None,
        'expiry_date': None
    }
    
    lines = text.split('\n')
    
    # Improved extraction logic
    for line in lines:
        clean_line = clean_text(line)
        
        if not info['brand'] and len(clean_line) > 2:
            info['brand'] = clean_line
        
        if re.search(r'\b\d+\s*(g|ml|l|kg)\b', clean_line, re.IGNORECASE):
            info['quantity'] = clean_line
            
        if re.search(r'\b\d{2}[/-]\d{2}[/-]\d{2,4}\b', line):
            info['expiry_date'] = line
            
    # Refine product name extraction
    remaining_lines = [l for l in lines if clean_text(l) not in info.values() and len(l) > 2]
    if remaining_lines:
        info['product_name'] = max(remaining_lines, key=len)
    
    return info


def extract_text(image_path):
    try:
        preprocessed_img = preprocess_image(image_path)
        if preprocessed_img is None:
            raise ValueError("Preprocessing failed")
        
        text_regions = detect_text_regions(preprocessed_img)
        
        all_text = ""
        for region in text_regions:
            text1 = apply_ocr(region, config='--psm 4')
            text2 = apply_ocr(region, config='--psm 11')
            all_text += text1 + "\n" + text2 + "\n"
        
        product_info = extract_product_info(all_text)
        return product_info
    except Exception as e:
        logger.error(f"An error occurred during OCR: {str(e)}")
        return None

def main():
    image_path = "C:/Users/aj\Desktop/SmartLabelVision/tests/data/IMG_20220318_153035.jpg"
    product_info = extract_text(image_path)
    if product_info:
        logger.info("Extracted Product Information:")
        for key, value in product_info.items():
            logger.info(f"{key.capitalize()}: {value}")
    else:
        logger.error("Failed to extract information from the image.")

if __name__ == "__main__":
    main()