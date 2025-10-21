import cv2
import os
import time

# --- Constants ---
# Directory to save the images
DATA_DIR = 'data'
# Number of samples to collect for each gesture
NUM_SAMPLES = 400
# List of gestures
CLASSES = ['rock', 'paper', 'scissors']

# --- Create Directories ---
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for class_name in CLASSES:
    class_dir = os.path.join(DATA_DIR, class_name)
    # The exist_ok=True parameter prevents an error if the directory already exists
    os.makedirs(class_dir, exist_ok=True)

print("Directories created. Starting data collection...")

# --- Data Collection Loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

for class_name in CLASSES:
    print(f"\nCollecting images for: {class_name}")
    print("Get ready! Place your hand in the green box.")
    
    # Give the user a moment to get ready
    time.sleep(3)
    
    # Prompt user to start capturing for the current class
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) # Flip for a mirror effect
        
        # Draw the ROI and instructions
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        cv2.putText(frame, f"Press 's' to start collecting for {class_name}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Start collecting samples
    count = 200
    while count < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        
	# Extract the Region of Interest (ROI)
        roi = frame[100:400, 100:400]

        # Display the count and ROI
        cv2.putText(frame, f'Capturing: {count}/{NUM_SAMPLES}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        cv2.imshow('Data Collection', frame)

        # Capture key 'c'
        if cv2.waitKey(25) & 0xFF == ord('c'):
            
            
            # Define the save path
            save_path = os.path.join(DATA_DIR, class_name, f'{class_name}_{count}.jpg')
            
            # Save the image
            cv2.imwrite(save_path, roi)
            print(f"Saved {save_path}")
            
            count += 1

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("\nData collection complete!")