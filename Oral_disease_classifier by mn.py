# -*- coding: utf-8 -*-
"""
Created on Tue May 13 00:51:19 2025

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Oral Disease Classifier using MobileNetV2
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Parameters
batch_size = 32
img_height = 224
img_width = 224
epochs = 30
seed = 123
data_dir = r"F:\oral diseases"

# Image preprocessing function with decorator to silence AutoGraph warning
@tf.autograph.experimental.do_not_convert
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    return image, label

# Data loading and preparation
def load_and_preprocess_data(directory):
    # Initial dataset load to get class names
    init_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(img_height, img_width),
        batch_size=None,
        shuffle=False,
        seed=seed
    )
    class_names = init_ds.class_names
    num_classes = len(class_names)
    
    # Get file paths and labels
    file_paths = init_ds.file_paths
    labels = np.array([label for _, label in init_ds])
    
    # Stratified split (60% train, 20% val, 20% test)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.25, stratify=train_labels, random_state=seed
    )
    
    # Create TensorFlow datasets
    def create_dataset(paths, labels):
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train_ds = create_dataset(train_paths, train_labels)
    val_ds = create_dataset(val_paths, val_labels)
    test_ds = create_dataset(test_paths, test_labels)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = dict(enumerate(class_weights))
    
    return train_ds, val_ds, test_ds, class_names, class_weights

# Load data
train_ds, val_ds, test_ds, class_names, class_weights = load_and_preprocess_data(data_dir)
num_classes = len(class_names)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
])

# Model building
def build_model():
    base_model = MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )
    
    # Fine-tuning: Unfreeze last 50 layers
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    
    # Top layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)

model = build_model()

# Fixed learning rate setup (removed the schedule from optimizer)
initial_learning_rate = 0.001
optimizer = optimizers.Adam(learning_rate=initial_learning_rate)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')]
)

# Callbacks with ReduceLROnPlateau for learning rate adjustment
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights
)



# Evaluation with random test samples - FIXED VERSION
def evaluate_model(model, test_ds, class_names, num_samples=18):
    # Get properly shaped random samples (fixes the ValueError)
    test_images, test_labels = next(iter(
        test_ds.unbatch().shuffle(1000).batch(num_samples)  # Unbatch first to get individual samples
    ))
    
    # Predictions
    test_preds = model.predict(test_images)
    test_pred_classes = np.argmax(test_preds, axis=1)
    
    # Classification report for full test set
    full_test_images, full_test_labels = [], []
    for images, labels in test_ds:
        full_test_images.append(images)
        full_test_labels.append(labels)
    
    full_test_images = tf.concat(full_test_images, axis=0)
    full_test_labels = tf.concat(full_test_labels, axis=0)
    
    full_preds = model.predict(full_test_images)
    full_pred_classes = np.argmax(full_preds, axis=1)
    
    print("\nFull Test Set Classification Report:")
    print(classification_report(full_test_labels, full_pred_classes, 
                              target_names=class_names, digits=4))
    
    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(full_test_labels, full_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, 
               yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Visualize random predictions with confidence
    plt.figure(figsize=(18, 12))
    for i in range(min(num_samples, len(test_images))):
        ax = plt.subplot(3, 6, i+1)
        plt.imshow(test_images[i].numpy().astype("uint8"))
        true_label = class_names[test_labels[i]]
        pred_label = class_names[test_pred_classes[i]]
        confidence = np.max(test_preds[i])
        color = "green" if true_label == pred_label else "red"
        
        plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}", 
                 color=color, fontsize=8)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()



evaluate_model(model, test_ds, class_names)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(14, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.plot(history.history['top3_accuracy'], label='Train Top-3 Accuracy')
    plt.plot(history.history['val_top3_accuracy'], label='Val Top-3 Accuracy')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Save final model
model.save('oral_disease_classifier.keras')
