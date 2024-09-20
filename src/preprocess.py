import cv2
import numpy as np
from skimage import exposure

def load_image(image_path):
    """Load an image from file."""
    return cv2.imread(image_path)

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
    return cv2.resize(image, (new_w, new_h))

def to_grayscale(image):
    """Convert the image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise_image(image):
    """Apply adaptive noise reduction."""
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def adjust_gamma(image, gamma=1.0):
    """Adjust the gamma of the image."""
    return exposure.adjust_gamma(image, gamma)

def enhance_contrast(image):
    """Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def sharpen_image(image):
    """Sharpen the image using unsharp masking."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

def binarize_image(image):
    """Apply adaptive thresholding to binarize the image."""
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def deskew_image(image):
    """Deskew the image."""
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(image_path, target_size=(1000, 1000)):
    """Load, resize, and preprocess an image for OCR."""
    # Load and resize image
    image = load_image(image_path)
    image = resize_image(image, target_size)
    
    # Convert to grayscale
    gray = to_grayscale(image)
    
    # Denoise the image
    denoised = denoise_image(gray)
    
    # Adjust gamma for brightness correction
    gamma_corrected = adjust_gamma(denoised, 1.5)
    
    # Enhance contrast
    contrast_enhanced = enhance_contrast(gamma_corrected)
    
    # Sharpen the image
    sharpened = sharpen_image(contrast_enhanced)
    
    # Apply adaptive binarization
    binary = binarize_image(sharpened)
    
    # Deskew the image
    deskewed = deskew_image(binary)
    
    return deskewed

if __name__ == "__main__":
    # Test the preprocessing functions
    test_image_path = "C:\\Users\\aj\\Desktop\\SmartLabelVision\\tests\\data\\IMG_20220318_180628.jpg"  # Replace with an actual image path
    processed_image = preprocess_image(test_image_path)
    
    # Display the original and processed images
    original_image = load_image(test_image_path)
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the processed image
    cv2.imwrite("processed_image.jpg", processed_image)
    print("Processed image saved as 'processed_image.jpg'")
