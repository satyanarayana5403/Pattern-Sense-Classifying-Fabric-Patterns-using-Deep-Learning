import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

# ✅ Parameters
img_size = 224
batch_size = 32
initial_epochs = 20
fine_tune_epochs = 10

# ✅ Dataset Paths (single dataset structure)
train_dir = 'dataset03/train'
val_dir = 'dataset03/val'
test_dir = 'dataset03/test'

# ✅ Data Generators with ResNet preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ✅ Get number of classes
num_classes = train_data.num_classes
print(f"Detected {num_classes} classes:", train_data.class_indices)

# ✅ Load ResNet50 Base Model
base_model = ResNet50(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze base layers

# ✅ Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ✅ Compile Phase 1
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    ModelCheckpoint('best_model_resnet50.keras', save_best_only=True)
]

# ✅ Train Phase 1
model.fit(
    train_data,
    validation_data=val_data,
    epochs=initial_epochs,
    callbacks=callbacks
)

# ✅ Fine-Tune Phase 2
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=fine_tune_epochs,
    callbacks=callbacks
)

# ✅ Save Final Model
model.save('Model_CNN.h5')

# ✅ Predict on Test Set
pred_probs = model.predict(test_data)
pred_classes = np.argmax(pred_probs, axis=1)

# ✅ Map predicted class indices to class names
class_labels = list(test_data.class_indices.keys())
predicted_names = [class_labels[i] for i in pred_classes]

# ✅ Print Predictions
for i in range(10):
    print(f"Image: {test_data.filenames[i]} --> Predicted: {predicted_names[i]}")
