from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np

# Load both models once
binary_model = load_model('./binary_classifier.keras')      # Classifies brain vs non-brain
tumor_model = load_model('./braintumor.keras')              # Classifies tumor type


def predict_image(img_path):
    """
    Step 1: Check if it's a brain MRI.
    Step 2: If yes, classify tumor type.
    """

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Predict brain vs non-brain
    binary_result = binary_model.predict(img_array)
    if binary_result[0][0] > 0.5:
      return {
            "classification": "non_brain",
            "message": "The uploaded image is not a brain MRI. Tumor classification skipped."
        }
    else:
        img = cv2.imread(img_path)
        img_s = cv2.resize(img.copy(), (150, 150))
        img_array = np.array(img_s).reshape(1, 150, 150, 3)
        tumor_result = tumor_model.predict(img_array)
        tumor_index = tumor_result.argmax()

        tumor_labels = {
            0: "Glioma",
            1: "Melignoma",
            2: "No Tumor",
            3: "Pituitary"
        }

        return {
            "classification": "brain_mri",
            "tumor_prediction": tumor_labels[tumor_index]
        }