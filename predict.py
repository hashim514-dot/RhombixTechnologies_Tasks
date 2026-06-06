import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

print("Loading model...")
model = tf.keras.models.load_model('model.keras')  # ← Just loads, doesn't train!
print("✅ Model loaded!\n")

def predict_image(img_path):
    """Predict if image is cat or dog"""
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    prediction = model.predict(img_array, verbose=0)
    
    if prediction[0][0] > 0.5:
        result = "DOG"
        confidence = prediction[0][0] * 100
    else:
        result = "CAT"
        confidence = (1 - prediction[0][0]) * 100
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{result} - {confidence:.2f}%", fontsize=14, fontweight='bold')
    plt.show()
    
    print(f"File: {img_path}")
    print(f"Result: {result}")
    print(f"Confidence: {confidence:.2f}%\n")

# Test cat images
print("=" * 60)
print("TESTING CAT IMAGES")
print("=" * 60)

cat_folder = "dataset/test/cats"
cat_list = os.listdir(cat_folder)[:3]

for cat_img in cat_list:
    cat_path = os.path.join(cat_folder, cat_img)
    predict_image(cat_path)

# Test dog images
print("\n" + "=" * 60)
print("TESTING DOG IMAGES")
print("=" * 60)

dog_folder = "dataset/test/dogs"
dog_list = os.listdir(dog_folder)[:3]

for dog_img in dog_list:
    dog_path = os.path.join(dog_folder, dog_img)
    predict_image(dog_path)

print("✅ Done!")
