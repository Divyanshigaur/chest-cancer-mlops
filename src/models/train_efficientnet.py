import tensorflow as tf
from tensorflow.keras import layers, models
import yaml
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# ✅ Fix path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.preprocessing import get_preprocessing, get_augmentation

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

os.makedirs(params["artifacts"]["model_dir"], exist_ok=True)

# Paths
train_dir = params["data"]["train_dir"]
val_dir = params["data"]["val_dir"]
test_dir = params["data"]["test_dir"]

# Config
IMG_SIZE = (params["model"]["input_size"], params["model"]["input_size"])
BATCH_SIZE = params["training"]["batch_size"]
EPOCHS = params["training"]["epochs"]
NUM_CLASSES = params["model"]["num_classes"]

# Load data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

# ✅ Preprocessing
augmentation_layer = get_augmentation()

# Train → augmentation + correct preprocessing
train_data = train_data.map(lambda x, y: (augmentation_layer(x), y))
train_data = train_data.map(lambda x, y: (preprocess_input(x), y))

# Val/Test → ONLY preprocessing
val_data = val_data.map(lambda x, y: (preprocess_input(x), y))
test_data = test_data.map(lambda x, y: (preprocess_input(x), y))

# 🔥 EfficientNet Base
base_model = EfficientNetB0(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Step 1 freeze

# Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile (initial)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# MLflow
mlflow.set_experiment("chest-cancer")

with mlflow.start_run(run_name="efficientnet_model"):

    mlflow.log_param("model", "EfficientNetB0")
    mlflow.log_param("fine_tuning", True)
    mlflow.log_artifact("src/models/train_efficientnet.py")
    mlflow.log_artifact("params.yaml")

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    # 🚀 Phase 1: Train frozen
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5,
        callbacks=[early_stopping]
    )

    # 🔥 Phase 2: Fine-tuning
    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Recompile with LOW LR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5,
        callbacks=[early_stopping]
    )

    # Metrics
    mlflow.log_metric("train_accuracy", history_fine.history['accuracy'][-1])
    mlflow.log_metric("val_accuracy", history_fine.history['val_accuracy'][-1])

    # Test
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test Accuracy: {test_acc}")

    y_true = np.concatenate([y for x, y in test_data], axis=0)
    y_pred = np.argmax(model.predict(test_data), axis=1)

    report = classification_report(y_true, y_pred, output_dict=True)

    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("EfficientNet Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = "efficientnet_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # Save model
    model_path = f"{params['artifacts']['model_dir']}/efficientnet.h5"
    model.save(model_path)

    mlflow.tensorflow.log_model(model, "model")

print("✅ EfficientNet + Fine-tuning complete!")