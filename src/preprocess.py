import cv2
import numpy as np

def load_image(image_path):
    """Load an image from file."""
    return cv2.imread(image_path)

def resize_image(image, target_size=(224, 224)):
    """Resize the image to the target size."""
    return cv2.resize(image, target_size)

def to_grayscale(image):
    """Convert the image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, ksize=(5,5)):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(image, ksize, 0)

def normalize_image(image):
    """Normalize pixel values to range [0, 1]."""
    return image.astype(np.float32) / 255.0

def preprocess_image(image_path, target_size=(224, 224), apply_gray=True, apply_blur=True):
    """Load, resize, and preprocess an image."""
    image = load_image(image_path)
    image = resize_image(image, target_size)
    if apply_gray:
        image = to_grayscale(image)
    if apply_blur:
        image = apply_gaussian_blur(image)
    image = normalize_image(image)
    return image

if __name__ == "__main__":
    # Test the preprocessing functions
    test_image_path = "C:\\Users\\aj\\Desktop\\SmartLabelVision\\tests\\image.webp"  # Replace with an actual image path
    processed_image = preprocess_image(test_image_path)
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Processed image data type: {processed_image.dtype}")
    print(f"Processed image value range: [{processed_image.min()}, {processed_image.max()}]")
    
    # Display the original and processed images
    original_image = load_image(test_image_path)
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()