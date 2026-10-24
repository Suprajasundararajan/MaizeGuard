import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm.keras import TqdmCallback

SEED = 42
CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
IMG_SIZE = (256, 256)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def collect_files(data_dir, allow_missing_healthy=False):
    rows = []
    missing = []
    for class_name in CLASS_NAMES:
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            missing.append(class_name)
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                try:
                    with Image.open(path) as im:
                        im.verify()
                    rows.append((str(path), class_name))
                except Exception:
                    pass
    if missing and not allow_missing_healthy:
        raise FileNotFoundError(
            f"Missing class folder(s): {missing}. A valid four-class run needs all folders. "
            "Add the Healthy PlantVillage images or use --allow-missing-healthy for a pipeline smoke test."
        )
    if len(set(label for _, label in rows)) < 2:
        raise ValueError("At least two non-empty classes are required.")
    return pd.DataFrame(rows, columns=["path", "label"])


def make_ds(frame, training, batch_size):
    paths = frame["path"].values
    labels = frame["label_id"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        return image, tf.one_hot(label, depth=len(CLASS_NAMES))

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(frame), 2048), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(name, n_classes):
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")
    x = augmentation(inputs)
    constructors = {
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "ResNet50": tf.keras.applications.ResNet50,
    }
    preprocessors = {
        "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
        "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "ResNet50": tf.keras.applications.resnet50.preprocess_input,
    }
    backbone = constructors[name](include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    backbone.trainable = False
    x = preprocessors[name](x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"MaizeGuard_{name}")
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss=loss, metrics=["accuracy"])
    return model, backbone


def make_callbacks(checkpoint):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_accuracy", save_best_only=True
        ),
    ]


def evaluate_model(model, ds, y_true, model_name, elapsed, out_dir):
    probabilities = model.predict(ds, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), average="macro", zero_division=0
    )
    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "training_time_seconds": elapsed,
        "model_size_mb": os.path.getsize(out_dir / f"{model_name}.keras") / 1e6,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }
    return result


def main(args):
    set_seed()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = collect_files(args.data_dir, args.allow_missing_healthy)
    frame["label_id"] = frame["label"].map({name: i for i, name in enumerate(CLASS_NAMES)})
    train_val, test = train_test_split(frame, test_size=0.15, stratify=frame["label_id"], random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.1765, stratify=train_val["label_id"], random_state=SEED)
    print(f"Images: {len(frame)} | train={len(train)} val={len(val)} test={len(test)}")
    print(frame["label"].value_counts().sort_index())
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train["label_id"].to_numpy(),
    )
    class_weights = dict(enumerate(class_weights))
    print(f"Class weights: {class_weights}")
    train_ds = make_ds(train, True, args.batch_size)
    val_ds = make_ds(val, False, args.batch_size)
    test_ds = make_ds(test, False, args.batch_size)
    y_test = test["label_id"].to_numpy()

    results = []
    for model_name in ["EfficientNetB0", "MobileNetV2", "ResNet50"]:
        print(f"\nTraining {model_name}")
        model, backbone = build_model(model_name, len(CLASS_NAMES))
        checkpoint = out_dir / f"{model_name}.keras"
        start = time.perf_counter()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=make_callbacks(checkpoint),
            verbose=0,
            class_weight=class_weights,
        )
        if args.fine_tune_epochs > 0:
            backbone.trainable = True
            trainable_layers = backbone.layers[-args.unfreeze_layers:]
            for layer in backbone.layers:
                layer.trainable = (
                    layer in trainable_layers
                    and not isinstance(layer, tf.keras.layers.BatchNormalization)
                )
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                          loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                          metrics=["accuracy"])
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.fine_tune_epochs,
                callbacks=make_callbacks(checkpoint),
                verbose=0,
                class_weight=class_weights,
            )
        elapsed = time.perf_counter() - start
        model.save(checkpoint)
        results.append(evaluate_model(model, test_ds, y_test, model_name, elapsed, out_dir))
        tf.keras.backend.clear_session()

    result_df = pd.DataFrame([{k: v for k, v in row.items() if k != "confusion_matrix"} for row in results])
    result_df.to_csv(out_dir / "model_comparison.csv", index=False)
    with open(out_dir / "evaluation_details.json", "w") as f:
        json.dump(results, f, indent=2)
    best_name = result_df.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]["model"]
    best_path = out_dir / f"{best_name}.keras"
    tf.keras.models.load_model(best_path).save(out_dir / "best_model.keras")
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"class_names": CLASS_NAMES, "best_model": best_name, "image_size": list(IMG_SIZE)}, f, indent=2)
    print(result_df.to_string(index=False))
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../PDD/data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--unfreeze-layers", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-missing-healthy", action="store_true")
    main(parser.parse_args())
