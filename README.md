# OpenCv-RPS-Game
A real-time hand gesture detector for Rock, Paper, Scissors, built with Python, OpenCV, MediaPipe, and TensorFlow/Keras.

## Real-Time Rock, Paper, Scissors Detector
This is a computer vision project that uses a machine learning model to detect and classify hand gestures for "Rock," "Paper," and "Scissors" in real-time using a webcam feed.

The primary goal of this project was to build and explore an end-to-end computer vision pipeline, from data collection and model training to real-time inference.

## Tech Stack
* **Python**
* **OpenCV**
* **MediaPipe** (for hand detection and segmentation)
* **TensorFlow / Keras**
* **NumPy**

## Current Status: Proof-of-Concept
This project is currently a functional proof-of-concept. The core pipeline is in place and can successfully classify gestures, but the model's accuracy is heavily dependent on the user and environment, especially when using the model trained on raw images.

## Analysis & Key Limitations
This project served as a valuable learning experience in applied machine learning. The primary limitations identified relate directly to the training data:

* **Limited, Homogeneous Dataset:** The model was exclusively trained on images of my own hand. As a result, it may not generalize well to other users with different hand shapes, sizes, or skin tones.
* **Lack of Background Variation:** The initial training data was captured using limited and simple backgrounds. This meant the model likely learned to associate my specific environment (e.g., my room, my lighting) with the gestures, rather than only the hand itself, leading to poor performance in new environments. Data augmentation improved accuracy, but lack of background variation enabled some inaccuracy to persist.

* **Segmentation Experiment:** To specifically address the background variation issue, a second pipeline was implemented using **MediaPipe** for **hand detection and segmentation**. This pipeline first isolates the hand using `HandLandmarker` and `ImageSegmenter`, creating an image of the hand on a black background. This segmented image is then fed to a model trained exclusively on similarly segmented data (`run_rps_game_with_segmentation.py`). In real-time testing, this approach **yielded better and more consistent performance** compared to the model trained on raw images with varied backgrounds, demonstrating the effectiveness of segmentation for background invariance in this setup.

## Future Improvements
While this proof-of-concept is complete, I recognize the clear steps needed to make this a robust and generalized application. The current limitations highlight the importance of dataset diversity.

To improve this project, the next steps would be:

* **Dataset Expansion:** The highest priority is to gather a large dataset (both raw and segmented) from multiple individuals in various lighting conditions and backgrounds.
* **Model Tuning:** Experiment with different model architectures (like MobileNetV2 for efficiency on edge devices) and hyperparameter tuning to improve accuracy and real-time performance for both segmented and non-segmented approaches.

## Trained Model

The trained model files (`image_classification_model.h5`, `image_classification_model2.h5`, etc.) are not included in this repository due to their potential large file size (often over 100MB). GitHub repositories are intended for source code, not large binary data files.

The models can be retrained using the provided data processing and `train_model.py` scripts. For the segmented version you will need to change the DATA_DIR constants and comment out augmentation and one of the convolutional layers and max pooling layers.

How to Run

Clone the repository:

Bash

git clone https://github.com/zachbs/OpenCv-RPS-Game.git

Install dependencies:

Bash

pip install -r requirements.txt
Run the application:

Bash

python run_rps_game.py

or

python run_rps_game_with_segmentation.py
