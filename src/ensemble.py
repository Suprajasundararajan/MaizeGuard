import time
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]


def _load_model_input_size():
    metadata_path = ARTIFACTS_DIR / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
        image_size = metadata["image_size"]
        if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
            raise ValueError
        return (int(image_size[0]), int(image_size[1]))
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return (256, 256)


MODEL_INPUT_SIZE = _load_model_input_size()

# Cached model dictionary to prevent re-loading weights from disk on every prediction
_LOADED_MODELS = {}


def get_model(model_name="best_model"):
    """
    Loads and caches the specified model from the artifacts directory.
    Supported names: 'best_model', 'MobileNetV2', 'EfficientNetB0', 'ResNet50'.
    """
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    if not model_name.endswith(".keras"):
        filename = f"{model_name}.keras"
    else:
        filename = model_name

    model_path = ARTIFACTS_DIR / filename
    if not model_path.exists():
        fallback_path = ARTIFACTS_DIR / "best_model.keras"
        if fallback_path.exists():
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    _LOADED_MODELS[model_name] = model
    return model


def preprocess_image(image_pil, target_size=MODEL_INPUT_SIZE):
    """Prepares a PIL image as a batch of float32 RGB array."""
    resized = image_pil.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_with_tta(model, image_pil, target_size=MODEL_INPUT_SIZE):
    """
    Test-Time Augmentation (TTA):
    Evaluates:
      1. Original image
      2. Horizontal flip
      3. Center-crop 90% (slight zoom)
    Averages predictions to boost robustness against field orientation & angle variations.
    """
    # 1. Original
    b1 = preprocess_image(image_pil, target_size)

    # 2. Horizontal Flip
    flipped_pil = image_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    b2 = preprocess_image(flipped_pil, target_size)

    # 3. Center Crop 90%
    w, h = image_pil.size
    dw = int(w * 0.05)
    dh = int(h * 0.05)
    cropped_pil = image_pil.crop((dw, dh, w - dw, h - dh))
    b3 = preprocess_image(cropped_pil, target_size)

    batch = np.concatenate([b1, b2, b3], axis=0)
    preds = model.predict(batch, verbose=0)
    avg_probs = np.mean(preds, axis=0)
    return avg_probs


def predict_single_backbone(model_name, image_pil, use_tta=True):
    """Runs inference using a single backbone."""
    model = get_model(model_name)
    t0 = time.perf_counter()
    if use_tta:
        probs = predict_with_tta(model, image_pil)
    else:
        batch = preprocess_image(image_pil)
        probs = model.predict(batch, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    order = np.argsort(probs)[::-1]
    return {
        "model_name": model_name,
        "probabilities": probs,
        "predicted_class": CLASS_NAMES[int(order[0])],
        "confidence": float(probs[order[0]]),
        "ranked_order": order,
        "latency_ms": latency_ms,
    }


def predict_ensemble(image_pil, use_tta=True):
    """
    Ensemble inference across MobileNetV2, EfficientNetB0, and ResNet50.
    Uses reliability-weighted soft voting. The validated best model receives
    the majority vote, while the other backbones provide a disagreement check.
    """
    # The benchmark identifies the best model for this artifact set. Giving it
    # the majority vote prevents a weaker, disagreeing model from needlessly
    # flattening an otherwise strong diagnosis.
    try:
        metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
        best_model = str(metadata.get("best_model", "best_model"))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        best_model = "best_model"

    benchmark_scores = {}
    try:
        import pandas as pd

        comparison = pd.read_csv(ARTIFACTS_DIR / "model_comparison.csv")
        benchmark_scores = {
            str(row["model"]): max(float(row["f1_macro"]), 0.0)
            for _, row in comparison.iterrows()
            if "model" in row and "f1_macro" in row
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        benchmark_scores = {}

    model_names = ("MobileNetV2", "EfficientNetB0", "ResNet50")
    score_total = sum(benchmark_scores.get(name, 0.0) for name in model_names)
    secondary_weights = (
        {
            name: benchmark_scores.get(name, 1.0) / score_total
            for name in model_names
        }
        if score_total > 0
        else {name: 1.0 / len(model_names) for name in model_names}
    )
    weights = {
        name: (0.6 if name == best_model else 0.0) + 0.4 * secondary_weights[name]
        for name in model_names
    }

    t0 = time.perf_counter()
    individual_results = {}
    weighted_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    total_weight = 0.0

    for m_name, w in weights.items():
        try:
            res = predict_single_backbone(m_name, image_pil, use_tta=use_tta)
            individual_results[m_name] = res
            weighted_probs += w * res["probabilities"]
            total_weight += w
        except Exception as e:
            # If a model file is absent, ignore gracefully
            continue

    if total_weight > 0:
        ensemble_probs = weighted_probs / total_weight
    else:
        # Fallback to best_model
        fallback_res = predict_single_backbone("best_model", image_pil, use_tta=use_tta)
        ensemble_probs = fallback_res["probabilities"]
        individual_results["best_model"] = fallback_res

    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    order = np.argsort(ensemble_probs)[::-1]

    return {
        "model_name": "Triple-Backbone Weighted Ensemble",
        "probabilities": ensemble_probs,
        "predicted_class": CLASS_NAMES[int(order[0])],
        "confidence": float(ensemble_probs[order[0]]),
        "ranked_order": order,
        "latency_ms": total_latency_ms,
        "individual_results": individual_results,
        "weights": weights,
    }
