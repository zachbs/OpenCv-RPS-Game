import mediapipe as mp
import cv2
import numpy as np
import os
import glob
import math

# --- Configuration ---
INPUT_DIR = 'data'
OUTPUT_DIR = 'processed_data'
HAND_MODEL_PATH = 'hand_landmarker.task'
# IMPORTANT: Make sure this model corresponds to the categories you provided
SEGMENT_MODEL_PATH = 'selfie_multiclass_256x256.tflite' # Or the actual model filename you downloaded

# Bounding Box Settings
PADDING_FACTOR = 0.1

# Confidence Threshold for Segmentation
# Pixels with confidence > this value for 'body-skin' will be kept
SEGMENTATION_THRESHOLD = 0.5 # Adjust if needed (0.0 to 1.0)
TARGET_CATEGORY_INDEX = 2 # 0=bg, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

segment_options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=SEGMENT_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    output_confidence_masks=True # Request confidence masks
)

# --- Helper Function: Calculate Bounding Box ---
def calculate_bounding_box(image_shape, landmarks):
    h, w = image_shape[:2]
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    box_w_px = (x_max - x_min) * w
    box_h_px = (y_max - y_min) * h
    x_min_px = int(x_min * w)
    y_min_px = int(y_min * h)
    padding_w = int(box_w_px * PADDING_FACTOR)
    padding_h = int(box_h_px * PADDING_FACTOR)
    x_min_pad = max(0, x_min_px - padding_w)
    y_min_pad = max(0, y_min_px - padding_h)
    x_max_pad = min(w, x_min_px + int(box_w_px) + padding_w)
    y_max_pad = min(h, y_min_px + int(box_h_px) + padding_h)
    return x_min_pad, y_min_pad, x_max_pad, y_max_pad

# --- Main Processing Loop ---
with HandLandmarker.create_from_options(hand_options) as landmarker, \
     ImageSegmenter.create_from_options(segment_options) as segmenter:

    print(f"Scanning for JPG images in '{INPUT_DIR}'...")
    image_paths = glob.glob(f"{INPUT_DIR}/**/*.jpg", recursive=True)
    print(f"Found {len(image_paths)} images to process.")

    processed_count = 0
    skipped_no_hand = 0
    skipped_segment_fail = 0

    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"!! Failed to load image {image_path}, skipping.")
                continue
            mp_image = mp.Image.create_from_file(image_path)

            # --- Step 1: Detect Hand Landmarks ---
            detection_result = landmarker.detect(mp_image)
            if not detection_result.hand_landmarks:
                print(f"-- No hand detected in {image_path}, skipping.")
                skipped_no_hand += 1
                continue

            # --- Step 2: Calculate Bounding Box & Crop ---
            landmarks = detection_result.hand_landmarks[0]
            x_min, y_min, x_max, y_max = calculate_bounding_box(frame.shape, landmarks)
            if x_min >= x_max or y_min >= y_max:
                 print(f"-- Invalid bounding box calculated for {image_path}, skipping.")
                 continue
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            if cropped_frame.size == 0:
                 print(f"-- Cropped frame is empty for {image_path}, skipping.")
                 continue

            # --- Step 3: Segment the Cropped Hand Region ---
            cropped_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            segmentation_result = segmenter.segment(cropped_mp_image)

            # --- MODIFIED MASK HANDLING ---
            if not hasattr(segmentation_result, 'confidence_masks') or not segmentation_result.confidence_masks:
                print(f"-- Segmentation failed or produced no confidence masks for {image_path}, skipping.")
                skipped_segment_fail += 1
                continue

            # Check if the target category index is valid for the masks list
            if TARGET_CATEGORY_INDEX >= len(segmentation_result.confidence_masks):
                 print(f"-- Target category index {TARGET_CATEGORY_INDEX} is out of bounds "
                       f"for available masks (count: {len(segmentation_result.confidence_masks)}) in {image_path}, skipping.")
                 skipped_segment_fail += 1
                 continue

            # Get the confidence mask specifically for the 'body-skin' category (index 2)
            confidence_mask = segmentation_result.confidence_masks[TARGET_CATEGORY_INDEX].numpy_view()

            # Create the condition based on the threshold
            condition = confidence_mask > SEGMENTATION_THRESHOLD
            # ---------------------------

            # --- Step 4: Create the Final Cutout ---
            if condition.shape != cropped_frame.shape[:2]:
                 print(f"-- Mask shape {condition.shape} doesn't match cropped frame {cropped_frame.shape[:2]} for {image_path}, attempting resize.")
                 condition = cv2.resize(condition.astype(np.uint8), (cropped_frame.shape[1], cropped_frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

            hand_cutout = np.zeros(cropped_frame.shape, dtype=np.uint8)
            condition_3c = np.stack((condition,) * 3, axis=-1)
            hand_cutout[condition_3c] = cropped_frame[condition_3c]

            # --- Step 5: Save the Output ---
            relative_path = os.path.relpath(image_path, INPUT_DIR)
            new_save_path = os.path.join(OUTPUT_DIR, relative_path)
            os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
            cv2.imwrite(new_save_path, hand_cutout)
            print(f"-> Saved segmented hand to: {new_save_path}")
            processed_count += 1

        except Exception as e:
            print(f"!! An unexpected error occurred processing {image_path}: {e}")
            # import traceback
            # traceback.print_exc()

    print("\n--- Processing Summary ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (no hand detected): {skipped_no_hand}")
    print(f"Skipped (segmentation failed/no mask/wrong index): {skipped_segment_fail}")
    print(f"Total images scanned: {len(image_paths)}")