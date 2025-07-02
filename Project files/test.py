import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# ✅ Parameters
img_size = 224
batch_size = 32
test_path = 'dataset4/test'  # Folder containing test subfolders by class

# ✅ Load the trained model
model = load_model('model/best_model_mobilenetv2.h5')
print("✅ Model loaded successfully!")

# ✅ Preprocess test dataset (for batch evaluation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ✅ Predict on batch test set
predictions = model.predict(test_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# ✅ Report
print("\n📊 Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print("🧮 Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

# ✅ ------- Single Image Prediction (Specify Path) -------
single_image_path = 'OIP.jpeg'  # 🔁 Change to your test image path

try:
    img = load_img(single_image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize like training

    pred = model.predict(img_array)
    pred_class_index = np.argmax(pred, axis=1)[0]
    pred_label = class_labels[pred_class_index]
    confidence = pred[0][pred_class_index]

    print("\n🔎 Single Image Prediction:")
    print(f"🖼️ Image: {single_image_path}")
    print(f"🔍 Predicted Class: {pred_label}")
    print(f"📈 Confidence: {confidence:.4f}")

except Exception as e:
    print(f"⚠️ Error reading or processing image: {e}")