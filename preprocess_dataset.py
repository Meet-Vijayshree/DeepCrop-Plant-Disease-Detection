import os
import from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ DIRECT PATHS (MOST IMPORTANT)
train_dir = "C:/Users/Shambhavi/Desktop/DeepCrop/DeepCrop_Final_Dataset/train"
val_dir   = "C:/Users/Shambhavi/Desktop/DeepCrop/DeepCrop_Final_Dataset/valid"

# ✅ Output directory
output_dir = "C:/Users/Shambhavi/Desktop/DeepCrop/preprocessed_dataset"

# ✅ Image generator (NO validation_split here)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

# ✅ Training data
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    save_to_dir=output_dir + "/train",
    save_prefix='aug',
    save_format='jpg'
)

# ✅ Validation data
val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    save_to_dir=output_dir + "/val",
    save_prefix='aug',
    save_format='jpg'
)

print("✅ Dataset preprocessing completed successfully!")
