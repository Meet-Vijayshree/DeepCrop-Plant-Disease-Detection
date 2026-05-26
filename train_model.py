# train_model.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json
import os

# ---------------------------------------------------------
# ✅ 1. Dataset setup
# ---------------------------------------------------------
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',               # 📂 your dataset path
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ---------------------------------------------------------
# ✅ 2. Model architecture
# ---------------------------------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(train_generator.num_classes, activation='softmax')
])

# ---------------------------------------------------------
# ✅ 3. Compile + Train
# ---------------------------------------------------------
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# ---------------------------------------------------------
# ✅ 4. Save model
# ---------------------------------------------------------
os.makedirs('model', exist_ok=True)
model.save('model/deepcrop_cnn.h5')
print("✅ Model saved at model/deepcrop_cnn.h5")

# ---------------------------------------------------------+-
# ✅ 5. Save class_indices.json (important for Flask)
# ---------------------------------------------------------
with open("model/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f, indent=2)
print("✅ Saved model/class_indices.json successfully!")
