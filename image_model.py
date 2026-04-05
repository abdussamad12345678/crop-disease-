"""
image_model.py

Simple image-based crop health prediction module.
Currently uses pixel intensity heuristic.
Can be upgraded to deep learning model (CNN) later.
"""

from PIL import Image
import numpy as np


class ImageDiseasePredictor:
    """
    Image-based disease prediction using basic pixel analysis.
    """

    def __init__(self, threshold: float = 100.0):
        """
        Initialize predictor.

        Args:
            threshold (float): Pixel intensity threshold for classification
        """
        self.threshold = threshold

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Convert image to normalized numpy array.

        Args:
            image (PIL.Image): Input image

        Returns:
            np.ndarray: Processed image array
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")

        img = image.convert("RGB")  # Ensure 3 channels
        img_array = np.array(img, dtype=np.float32)

        return img_array

    def predict(self, image: Image.Image) -> dict:
        """
        Predict disease status from image.

        Args:
            image (PIL.Image): Input image

        Returns:
            dict: Prediction result with label and confidence
        """
        img_array = self.preprocess(image)

        avg_pixel = float(img_array.mean())

        if avg_pixel < self.threshold:
            label = "Disease Detected"
            confidence = round(1 - (avg_pixel / self.threshold), 2)
        else:
            label = "Healthy"
            confidence = round(avg_pixel / 255, 2)

        return {
            "label": label,
            "confidence": confidence,
            "avg_pixel_value": round(avg_pixel, 2)
        }


# -------------------------------
# Convenience function (for quick use)
# -------------------------------
def predict_image(image: Image.Image) -> str:
    """
    Quick prediction function for backward compatibility.

    Args:
        image (PIL.Image): Input image

    Returns:
        str: Prediction label
    """
    predictor = ImageDiseasePredictor()
    result = predictor.predict(image)
    return result["label"]
