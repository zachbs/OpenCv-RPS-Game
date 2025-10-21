# OpenCv-RPS-Game
A real-time hand gesture detector for Rock, Paper, Scissors, built with Python, OpenCV, and TensorFlow/Keras.
Real-Time Rock, Paper, Scissors Detector
This is a computer vision project that uses a machine learning model to detect and classify hand gestures for "Rock," "Paper," and "Scissors" in real-time using a webcam feed.

The primary goal of this project was to build and explore an end-to-end computer vision pipeline, from data collection and model training to real-time inference.

Tech Stack
Python

OpenCV

TensorFlow / Keras (or PyTorch)

NumPy

Current Status: Proof-of-Concept
This project is currently a functional proof-of-concept. The core pipeline is in place and can successfully classify gestures, but the model's accuracy is heavily dependent on the user and environment.

Analysis & Key Limitations
This project served as a valuable learning experience in applied machine learning. The primary limitations of the current model are directly related to the training data:

Limited, Homogeneous Dataset: The model was exclusively trained on images of my own hand. As a result, it does not generalize well to other users with different hand shapes, sizes, or skin tones.

Lack of Background Variation: The training data was captured using limited and simple backgrounds. This means the model has likely learned to associate my specific environment (e.g., my room, my lighting) with the gestures, rather than only the hand itself. It will not perform well in new or complex environments. I added data augmentation elements which improved accuracy, but lack of background variation enabled some inaccuracy to persist.

Future Improvements
While this proof-of-concept is complete, I recognize the clear steps needed to make this a robust and generalized application. The current limitations highlight the importance of a diverse and comprehensive dataset.

To improve this project, the next steps would be:

Dataset Expansion: The highest priority is to gather a large dataset from multiple individuals in various lighting conditions.

Background Invariance: Explore preprocessing techniques like hand segmentation or background subtraction to isolate the hand gesture and force the model to learn only the relevant features.

Model Tuning: Experiment with different model architectures (like MobileNet for efficiency) and hyperparameter tuning to improve accuracy and real-time performance.

## Trained Model

The trained model (`image_classification_model.h5`) is not included in this repository due to its large file size (over 200MB). GitHub repositories are intended for source code, not large binary data files.

The model can be retrained by running the the `Train_Model.py` scripts.

How to Run
(Optional, but good to have)

Clone the repository:

Bash

git clone [your-repo-link]
Install dependencies:

Bash

pip install -r requirements.txt
Run the application:

Bash

python [your_main_script.py]
