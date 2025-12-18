import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- 1. Data Preparation ---
# Using the user's dataset location
base_dir = r'X:\FTDS\P2\GC7\rps-cv-images'

# Check if data exists, if not, logic to extract would go here (similar to original notebook)
if not os.path.exists(base_dir):
    print("Dataset directory not found. Please ensure data is extracted in 'tmp/rps-cv-images/'")
    # In a real scenario, we'd trigger extraction here if needed.

# --- 2. Enhanced Data Augmentation ---
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.4  # Original split
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224), # EfficientNetB0 standard size
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# --- 3. Build EfficientNetB0 Model ---
def build_model(num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

model, base_model = build_model(num_classes=3)

# --- 4. Phase 1: Training the Top Layers ---
print("--- Phase 1: Training the classification head ---")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

history_phase1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=callbacks
)

# --- 5. Phase 2: Fine-Tuning ---
print("--- Phase 2: Fine-tuning the top layers of EfficientNet ---")
# Unfreeze the base model
base_model.trainable = True

# We only unfreeze the top layers to avoid destroying pretrained weights
# EfficientNetB0 has ~237 layers. Unfreeze the last 30.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a much lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('rock_paper_scissors_improved_model.h5', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max')

history_phase2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=callbacks + [checkpoint]
)

print("Training finished. Improved model saved as 'rock_paper_scissors_improved_model.h5'")
