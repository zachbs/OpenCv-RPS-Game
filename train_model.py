import tensorflow as tf
import os
import matplotlib.pyplot as plt

# --- Constants ---
DATA_DIR = 'processed_data'
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32 # Number of images to process at a time

# check if gpu is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("TensorFlow did NOT find any GPUs.")

# --- Load and Prepare Data ---

# This powerful function does most of the work for us.
# It reads images from the subdirectories in DATA_DIR, uses the directory
# names as labels, and resizes the images.
# We also split the data: 80% for training, 20% for validation.

print("Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123, # Seed for reproducibility
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

print("\nLoading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# --- Verify the Data ---
class_names = train_ds.class_names
print(f"\nClasses found: {class_names}")

# This shows the shape of our data.
# The 'None' in the batch size means it can be flexible.
# (150, 150, 3) means 150x150 pixels with 3 color channels (RGB).
for image_batch, labels_batch in train_ds.take(1):
    print("\nShape of one batch of images:", image_batch.shape)
    print("Shape of one batch of labels:", labels_batch.shape)
    break

# --- Configure for Performance ---
# These are standard optimization steps.
# .cache() keeps images in memory after they're loaded off disk.
# .prefetch() overlaps data preprocessing and model execution.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\nDatasets are now ready for training!")

# Your model building and training code (model.compile, model.fit) would go here...
# For example:
# model = ... (your Sequential CNN model definition)
# model.compile(...)
# model.fit(train_ds, validation_data=val_ds, epochs=10)
Epochs = 50
data_augmentation = tf.keras.Sequential([
    # Randomly rotate the image by up to 20 degrees
    tf.keras.layers.RandomRotation(0.2), 
    # Randomly zoom in on the image by up to 20%
    tf.keras.layers.RandomZoom(0.2),
    # Randomly shift the image horizontally and vertically by up to 20% of its size
    tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
    # Randomly flip the image horizontally (great for left/right hand variety)
    tf.keras.layers.RandomFlip("horizontal"), 
])
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    data_augmentation,
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])
# Manually create the Adam optimizer with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer, # Use your custom optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.fit(train_ds, validation_data=val_ds, epochs=Epochs)


# --- Graphing the results ---
history = model.history
# 1. Get the accuracy and loss for each epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(Epochs)

# 2. Create the plots
plt.figure(figsize=(12, 6)) # Create a figure to contain the plots

# Plot for accuracy
plt.subplot(1, 2, 1) # 1 row, 2 columns, this is the first plot
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Plot for loss
plt.subplot(1, 2, 2) # 1 row, 2 columns, this is the second plot
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 3. Show the plots
plt.suptitle('Model Training History')
plt.show()

# Save the model
model.save('image_classification_model2.h5')

