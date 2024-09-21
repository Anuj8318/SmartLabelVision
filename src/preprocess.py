import cv2
import numpy as np
from skimage import exposure
import logging

# Set up logging
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

def preprocess_image(image_path, target_size=(1000, 1000), gamma=1.5, clip_limit=2.0):
    """Load, resize, and preprocess an image for OCR with configurable parameters."""
    image = load_image(image_path)
    if image is None:
        return None

    try:
        # Step 1: Resize
        image = resize_image(image, target_size)

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Denoise and Enhance Contrast in a Single Step
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(denoised)

        # Step 4: Gamma Correction
        gamma_corrected = exposure.adjust_gamma(contrast_enhanced, gamma)

        # Step 5: Sharpening
        sharpened = cv2.GaussianBlur(gamma_corrected, (0, 0), 3)
        sharpened = cv2.addWeighted(gamma_corrected, 1.5, sharpened, -0.5, 0)

        # Step 6: Binarization
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Step 7: Deskew
        coords = np.column_stack(np.where(binary > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = binary.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return deskewed

    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        return None

if __name__ == "__main__":
    test_image_path = "C:/Users/aj\Desktop/SmartLabelVision/tests/data/IMG_20220318_153035.jpg"
    processed_image = preprocess_image(test_image_path, target_size=(1000, 1000), gamma=1.5, clip_limit=2.0)
    
    if processed_image is not None:
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("processed_image.jpg", processed_image)
        logger.info("Processed image saved as 'processed_image.jpg'")
    else:
        logger.error("Failed to process the image")
