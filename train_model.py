import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("Checking dataset structure...")

# ✅ CORRECT PATHS
train_path = "dataset/train"
validation_path = "dataset/validation"

print(f"Train path exists: {os.path.exists(train_path)}")
print(f"Validation path exists: {os.path.exists(validation_path)}")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Training generator
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary"
)

# Validation generator
validation_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary"
)

print(f"\n✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {validation_generator.samples}\n")

# Build model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("✅ Model compiled!")

# Train
print("\n🚀 Training started...\n")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save
model.save("model.keras")
print("\n✅ Model saved as model.keras!")

print(f"\n📊 Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"📊 Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")