import sys
import os
import yaml
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

# ✅ Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# ✅ Import preprocessing
from src.data.preprocessing import get_preprocessing, get_augmentation


# ================= LOAD PARAMS =================
with open("params.yaml") as f:
    params = yaml.safe_load(f)

os.makedirs(params["artifacts"]["model_dir"], exist_ok=True)

# Paths
train_dir = params["data"]["train_dir"]
val_dir = params["data"]["val_dir"]
test_dir = params["data"]["test_dir"]

IMG_SIZE = (params["model"]["input_size"], params["model"]["input_size"])
BATCH_SIZE = params["training"]["batch_size"]
EPOCHS = params["training"]["epochs"]
NUM_CLASSES = params["model"]["num_classes"]


# ================= LOAD DATA =================
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ================= PREPROCESS =================
normalization_layer = get_preprocessing()
augmentation_layer = get_augmentation()

train_data = train_data.map(lambda x, y: (augmentation_layer(normalization_layer(x)), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))


# ================= MODEL =================
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["training"]["learning_rate"]),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ================= MLFLOW =================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("chest-cancer")

with mlflow.start_run(run_name="cnn_model"):

    # Params
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", params["training"]["learning_rate"])
    mlflow.log_param("model_name", params["model"]["name"])

    # Artifacts (SAFE paths)
    mlflow.log_artifact("src/models/train_cnn.py")
    mlflow.log_artifact("params.yaml")

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    # ================= TRAIN =================
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )

    mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])

    # ================= EVALUATE =================
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test Accuracy: {test_acc}")

    y_true = np.concatenate([y for x, y in test_data], axis=0)
    y_pred = np.argmax(model.predict(test_data), axis=1)

    report = classification_report(y_true, y_pred, output_dict=True)

    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # ================= SAVE MODEL =================
    model.save("models/cnn.h5")
    mlflow.tensorflow.log_model(model, "model")


print("✅ Training + MLflow complete!")