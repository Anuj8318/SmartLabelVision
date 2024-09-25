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

# Configure Tesseract for Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to decode predictions from EAST model output
def decode_predictions(scores, geometry, min_confidence=0.5):
    num_rows, num_cols = scores.shape[2:4]
    boxes = []
    confidences = []
    
    for y in range(num_rows):
        for x in range(num_cols):
            score = scores[0, 0, y, x]
            if score < min_confidence:
                continue
            
            offset_x, offset_y = x * 4.0, y * 4.0
            
            # Extract geometry data for bounding box
            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            angle = geometry[0, 4, y, x]
            
            # Calculate the coordinates using rotation
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_center = offset_x + cos_a * geometry[0, 1, y, x] + sin_a * geometry[0, 2, y, x]
            y_center = offset_y - sin_a * geometry[0, 1, y, x] + cos_a * geometry[0, 2, y, x]
            
            # Append box and confidence score
            boxes.append([int(x_center - w / 2), int(y_center - h / 2), int(w), int(h)])
            confidences.append(float(score))
    
    return boxes, confidences

# Function to detect text regions using the EAST model
def detect_text_regions(image):
    orig_h, orig_w = image.shape[:2]
    
    if len(image.shape) == 2:  # Convert grayscale to 3-channel if needed
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Resize the image for the model
    resized_image = cv2.resize(image, (320, 320))
    ratio_w, ratio_h = orig_w / 320.0, orig_h / 320.0
    
    # Prepare the image blob for the EAST model
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (320, 320), (123.68, 116.78, 103.94), True, False)
    
    # Load the EAST model and perform a forward pass
    net = cv2.dnn.readNet("models/frozen_east_text_detection.pb")
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    
    # Decode the predictions and apply non-max suppression
    boxes, confidences = decode_predictions(scores, geometry)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Extract text regions from the original image
    text_regions = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            x, y = int(x * ratio_w), int(y * ratio_h)
            w, h = int(w * ratio_w), int(h * ratio_h)
            text_regions.append(image[y:y+h, x:x+w])
    
    return text_regions

# Apply OCR to the detected text regions
def apply_ocr(image, config='--psm 6'):
    try:
        return pytesseract.image_to_string(image, lang='eng+hin')
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return ""
# Clean up extracted text by removing unwanted characters
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

# New helper functions for information extraction
def extract_brand(text):
    known_brands = ['quaker', 'nestle', 'kelloggs', 'pepsico', 'unilever']
    words = text.split()
    for word in words:
        if word in known_brands:
            return word.capitalize()
    return None

def extract_product_name(text):
    words = text.split()
    product_words = []
    for word in words:
        if word in english_words and len(word) > 2:
            product_words.append(word)
        if len(product_words) == 2:
            break
    return ' '.join(product_words).capitalize() if product_words else None

def extract_quantity(text):
    quantity_pattern = r'\b(\d+(?:\.\d+)?)\s*(g|kg|ml|l|oz|lb)\b'
    match = re.search(quantity_pattern, text, re.IGNORECASE)
    if match:
        return match.group()
    return None

def extract_expiry_date(text):
    date_formats = [
        r'\b(0?[1-9]|[12][0-9]|3[01])[/.-](0?[1-9]|1[012])[/.-](19|20)\d\d\b',
        r'\b(0?[1-9]|1[012])[/.-](0?[1-9]|[12][0-9]|3[01])[/.-](19|20)\d\d\b',
        r'\b(19|20)\d\d[/.-](0?[1-9]|1[012])[/.-](0?[1-9]|[12][0-9]|3[01])\b',
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (0?[1-9]|[12][0-9]|3[01]),? (19|20)\d\d\b'
    ]
    
    for date_format in date_formats:
        match = re.search(date_format, text, re.IGNORECASE)
        if match:
            try:
                date_str = match.group()
                date_obj = datetime.strptime(date_str, "%d/%m/%Y")
                return date_obj.strftime("%d/%m/%Y")
            except ValueError:
                continue
    return None

# Updated extract_product_info function
def extract_product_info(text):
    cleaned_text = clean_text(text)
    
    info = {
        'brand': extract_brand(cleaned_text),
        'product_name': extract_product_name(cleaned_text),
        'quantity': extract_quantity(cleaned_text),
        'expiry_date': extract_expiry_date(cleaned_text)
    }
    
    return info

# Main text extraction pipeline
def extract_text(image_path):
    try:
        binary, color_processed = preprocess_image(image_path)
        logger.info(f"Preprocessing completed. Binary shape: {binary.shape}, Color shape: {color_processed.shape}")
        
        text_regions = detect_text_regions(color_processed)
        logger.info(f"Detected {len(text_regions)} text regions")
        
        all_text = ""
        for i, region in enumerate(text_regions):
            region_text = apply_ocr(region, config=r'--oem 3 --psm 11')
            logger.info(f"OCR result for region {i}: {region_text}")
            all_text += region_text + "\n"
        
        product_info = extract_product_info(all_text)
        logger.info(f"Extracted product info: {product_info}")
        
        return product_info
    except Exception as e:
        logger.error(f"Error in extract_text: {str(e)}")
        return None

# Main function to run the pipeline
def main():
    image_path = "C:/Users/aj/Desktop/SmartLabelVision/tests/data/nescafe-classic-instant-coffee-45-g-product-images-o490004155-p490004155-0-202309041728.webp"
    try:
        product_info = extract_text(image_path)
        if product_info:
            logger.info("Extracted Product Information:")
            for key, value in product_info.items():
                logger.info(f"{key.capitalize()}: {value}")
        else:
            logger.error("Failed to extract information from the image.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
