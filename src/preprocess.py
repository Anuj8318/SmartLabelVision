import cv2
import numpy as np
from skimage import exposure
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_path):
    """Load an image from file."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None

def resize_image(image, target_size=(1000, 1000)):
    """Resize the image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    aspect = w / h
    if w > h:
        new_w = target_size[0]
        new_h = int(new_w / aspect)
    else:
        new_h = target_size[1]
        new_w = int(new_h * aspect)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def enhance_contrast(image):
    """Enhance contrast using CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def denoise_image(image):
    """Apply non-local means denoising."""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def sharpen_image(image):
    """Sharpen the image using unsharp masking."""
    gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0, image)

def adjust_gamma(image, gamma=1.0):
    """Adjust the image gamma."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def create_text_mask(image):
    """Create a mask to isolate text-like regions using Sobel and Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    gradient = np.uint8(gradient / gradient.max() * 255)

    # Combine with Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    combined_edges = cv2.addWeighted(gradient, 0.5, edges, 0.5, 0)
    
    # Threshold and morphological operations
    _, thresh = cv2.threshold(combined_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    text_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return text_mask

def remove_background(image):
    """Remove background using GrabCut algorithm."""
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (20, 20, image.shape[1]-20, image.shape[0]-20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    return image * mask2[:,:,np.newaxis]

def preprocess_image(image_path, target_size=(1000, 1000)):
    """Preprocess the image for optimal OCR performance."""
    logger.info(f"Loading image from {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        logger.error(f"Failed to load image from {image_path}. Check if the file exists and the format is supported.")
        return None, None
    
    logger.info(f"Image loaded successfully. Image shape: {image.shape}")

    try:
        # Resize the image
        image = resize_image(image, target_size)
        
        # Remove background
        image = remove_background(image)
        
        # Enhance contrast and denoise
        enhanced = enhance_contrast(image)
        denoised = denoise_image(enhanced)
        
        # Sharpen and adjust gamma
        sharpened = sharpen_image(denoised)
        gamma_corrected = adjust_gamma(sharpened, 1.2)
        
        # Create text mask
        text_mask = create_text_mask(gamma_corrected)
        
        # Apply text mask to the original image
        text_regions = cv2.bitwise_and(gamma_corrected, gamma_corrected, mask=text_mask)
        
        # Final binarization
        gray = cv2.cvtColor(text_regions, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        logger.info("Preprocessing completed successfully.")
        return binary, gamma_corrected  # Return both binary and processed color image

    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        return None, None

def main():
    image_path = "C:/Users/aj/Desktop/SmartLabelVision/tests/data/IMG_20220318_180632.jpg"
    
    # Capture both return values from preprocess_image
    binary, color_processed = preprocess_image(image_path)
    
    if binary is not None and color_processed is not None:
        cv2.imwrite("preprocessed_binary.jpg", binary)
        cv2.imwrite("preprocessed_color.jpg", color_processed)
        logger.info("Preprocessed images saved successfully.")
    else:
        logger.error("Preprocessing failed.")

if __name__ == "__main__":
    main()