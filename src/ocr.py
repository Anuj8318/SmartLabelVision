import cv2
import pytesseract
import numpy as np
from PIL import Image
import os
import re
from preprocess import preprocess_image

if os.name == 'nt':  # for Windows
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'

def detect_text_regions(image):
    # Use EAST text detector or a similar algorithm to detect text regions
    # This is a placeholder function - you'll need to implement actual text detection
    # For now, we'll just return the whole image
    return [image]

def apply_ocr(image, config=''):
    try:
        return pytesseract.image_to_string(image, config=config)
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return ""

def clean_text(text):
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase and strip whitespace
    return text.lower().strip()

def extract_product_info(text):
    # This is a simple example - you'll need to adapt this based on your specific needs
    info = {
        'brand': None,
        'product_name': None,
        'quantity': None,
        'expiry_date': None
    }
    
    lines = text.split('\n')
    for line in lines:
        clean_line = clean_text(line)
        if not info['brand'] and len(clean_line) > 2:
            info['brand'] = clean_line  # Assume first non-empty line is brand
        elif 'g' in clean_line or 'ml' in clean_line or 'l' in clean_line:
            info['quantity'] = clean_line  # Assume this line contains quantity
        elif re.search(r'\d{2}[/-]\d{2}[/-]\d{2,4}', line):
            info['expiry_date'] = line  # Assume this line contains expiry date
    
    # Assume the longest remaining line is the product name
    remaining_lines = [l for l in lines if clean_text(l) not in info.values() and len(l) > 2]
    if remaining_lines:
        info['product_name'] = max(remaining_lines, key=len)
    
    return info

def extract_text(image_path):
    try:
        # Preprocess the image
        preprocessed_img = preprocess_image(image_path, target_size=(1000, 1000), apply_gray=True, apply_blur=False)
        preprocessed_img = (preprocessed_img * 255).astype(np.uint8)
        
        # Detect text regions
        text_regions = detect_text_regions(preprocessed_img)
        
        all_text = ""
        for region in text_regions:
            # Apply different preprocessing techniques
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Perform multiple OCR passes
            text1 = apply_ocr(binary, config='--psm 6')
            text2 = apply_ocr(denoised, config='--psm 11')
            
            all_text += text1 + "\n" + text2 + "\n"
        
        # Extract product information
        product_info = extract_product_info(all_text)
        
        return product_info
    except Exception as e:
        print(f"An error occurred during OCR: {str(e)}")
        return None

def main():
    image_path = 'processed_image.jpg'  # Update this with the path to your test image
    product_info = extract_text(image_path)
    if product_info:
        print("Extracted Product Information:")
        for key, value in product_info.items():
            print(f"{key.capitalize()}: {value}")
    else:
        print("Failed to extract information from the image.")

if __name__ == "__main__":
    main()