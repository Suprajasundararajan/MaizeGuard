import numpy as np
from PIL import Image


def compute_shannon_entropy(probabilities):
    """
    Computes Shannon Entropy (in bits) and normalized entropy in [0, 1].
    H(P) = - sum(p * log2(p))
    For 4 classes, max entropy is log2(4) = 2.0 bits.
    """
    probs = np.asarray(probabilities, dtype=np.float32)
    # Avoid log(0)
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    # Re-normalize just in case
    probs = probs / np.sum(probs)
    
    entropy_bits = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))
    normalized_entropy = float(entropy_bits / max_entropy)
    
    return float(entropy_bits), float(np.clip(normalized_entropy, 0.0, 1.0))


def evaluate_image_quality_and_ood(image_pil, probabilities):
    """
    Evaluates image quality, non-plant heuristic, and Out-of-Distribution (OOD) uncertainty.
    Returns a dict with uncertainty metrics, OOD flag, and recommendations.
    """
    entropy_bits, norm_entropy = compute_shannon_entropy(probabilities)
    max_prob = float(np.max(probabilities))

    # 1. Image checks (Mean, Std Dev, Excess Green Index)
    arr = np.asarray(image_pil.convert("RGB"), dtype=np.float32)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    
    # Excess Green Index: 2G - R - B (plant vegetation indicator)
    exg = (2.0 * g) - r - b
    mean_exg = float(np.mean(exg))
    
    warnings = []
    is_ood = False

    # Low contrast / blank image check
    if std_val < 15.0:
        is_ood = True
        warnings.append("Low contrast or blank image detected. Ensure the leaf is well-lit and in focus.")
    elif mean_val < 25.0:
        is_ood = True
        warnings.append("Image is extremely dark. Retake with adequate sunlight or flash.")
    elif mean_val > 240.0:
        is_ood = True
        warnings.append("Image is overexposed/blown-out. Adjust camera exposure.")

    # Non-vegetation check: If ExG is heavily negative and green channel is minimal
    if mean_exg < -40.0 and np.mean(g) < 40.0:
        is_ood = True
        warnings.append("Image does not exhibit typical foliar foliage color profiles (suspected non-plant image).")

    # High predictive entropy / flat distribution check
    if norm_entropy > 0.65 or max_prob < 0.50:
        is_ood = True
        warnings.append("Model confidence is diffuse across multiple classes (High Epistemic Uncertainty).")

    # Uncertainty tier
    if norm_entropy < 0.25 and max_prob >= 0.85:
        tier = "High Certainty"
        tier_color = "#10B981"  # Emerald
    elif norm_entropy < 0.55 and max_prob >= 0.60:
        tier = "Moderate Certainty"
        tier_color = "#F59E0B"  # Amber
    else:
        tier = "High Uncertainty / Ambiguous"
        tier_color = "#EF4444"  # Red

    return {
        "entropy_bits": round(entropy_bits, 3),
        "normalized_entropy": round(norm_entropy, 3),
        "is_out_of_distribution": is_ood,
        "uncertainty_tier": tier,
        "tier_color": tier_color,
        "warnings": warnings,
        "mean_brightness": round(mean_val, 1),
        "contrast_std": round(std_val, 1),
        "vegetation_index": round(mean_exg, 1),
    }
