import tensorflow as tf
import numpy as np
import cv2 # We use OpenCV here just to load and preprocess the image
import matplotlib.pyplot as plt

# --- PRE-FLIGHT CHECKLIST (Fill this out) ---

# 1. Path to your saved Keras model
MODEL_PATH = 'image_classification_model.h5'

# 2. Path to the single image you want to test
#    IMPORTANT: Replace this with the actual path to YOUR test image
TEST_IMAGE_PATH = 'my_test_rock.jpg' # <--- CHANGE THIS

# 3. The class labels in the correct alphabetical order
CLASS_LABELS = ['paper', 'rock', 'scissors']

# 4. The image size your model expects
IMG_SIZE = (150, 150)

# --- END OF CHECKLIST ---


def predict_image(model, image_path):
    """Loads an image, preprocesses it, and returns the model's prediction."""
    
    # 1. Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None, None

    # --- PREPROCESSING: This MUST match your training pipeline ---

    # a. Convert color from BGR (OpenCV default) to RGB (TensorFlow expects)
    #    This is a very common bug, so we do it just in case.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # b. Resize the image to the target size
    resized_image = cv2.resize(rgb_image, IMG_SIZE)

    # c. IMPORTANT: Handle Normalization
    #    If your model has a Rescaling layer, you don't need to divide by 255.
    #    If it DOES NOT, you MUST uncomment the line below.
    # resized_image = resized_image / 255.0

    # d. Expand dimensions to create a "batch" of 1 image
    #    The model expects (batch_size, height, width, channels)
    img_batch = np.expand_dims(resized_image, axis=0)
    
    # --- Make Prediction ---
    prediction = model.predict(img_batch)
    
    # --- Interpret Prediction ---
    predicted_class_index = np.argmax(prediction[0])
    predicted_label = CLASS_LABELS[predicted_class_index]
    confidence = np.max(prediction[0])
    
    # Display the image with its prediction
    plt.imshow(rgb_image)
    plt.title(f'Prediction: {predicted_label} ({confidence:.2f})')
    plt.axis('off')
    plt.show()
    
    return predicted_label, confidence


# --- Main execution ---
if __name__ == "__main__":
    # Load the trained model
    print(f"Loading model from: {MODEL_PATH}")
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    
    # Make a prediction on the test image
    label, conf = predict_image(loaded_model, TEST_IMAGE_PATH)
    
    if label:
        print("\n--- Prediction Result ---")
        print(f"The model predicted: {label}")
        print(f"Confidence: {conf:.2%}")