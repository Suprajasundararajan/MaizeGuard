import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm


def find_backbone_and_target_layer(model):
    """
    Identifies the nested backbone model and the best convolutional layer for Grad-CAM.
    Supports MobileNetV2, EfficientNetB0, and ResNet50.
    """
    bb_idx = None
    bb_model = None
    for i, layer in enumerate(model.layers):
        name_lower = layer.name.lower()
        if any(arch in name_lower for arch in ["mobilenet", "resnet", "efficient"]):
            bb_idx = i
            bb_model = layer
            break

    if bb_model is None:
        # Fallback: look for any nested Functional or Sequential layer
        for i, layer in enumerate(model.layers):
            if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)) and i > 0:
                bb_idx = i
                bb_model = layer
                break

    if bb_model is None:
        raise ValueError("Could not find a convolutional backbone in the model.")

    # Target layers based on architecture
    candidate_names = [
        "out_relu",           # MobileNetV2
        "top_activation",     # EfficientNetB0
        "conv5_block3_out",   # ResNet50
        "block_16_project_BN",
        "top_conv",
    ]

    target_layer_name = None
    for cand in candidate_names:
        try:
            bb_model.get_layer(cand)
            target_layer_name = cand
            break
        except (ValueError, AttributeError):
            continue

    # Fallback: search backwards for last 4D output layer in backbone
    if target_layer_name is None:
        for layer in reversed(bb_model.layers):
            output_shape = getattr(layer, "output_shape", None)
            if output_shape and len(output_shape) == 4:
                target_layer_name = layer.name
                break

    return bb_idx, bb_model, target_layer_name


def compute_gradcam(model, image_array, class_index=None):
    """
    Computes a Grad-CAM heatmap for a given input image array.
    image_array: shape (1, H, W, 3) with float values in [0, 255].
    """
    bb_idx, bb_model, target_layer_name = find_backbone_and_target_layer(model)
    target_layer = bb_model.get_layer(target_layer_name)

    # Submodel for backbone feature extraction
    bb_submodel = tf.keras.Model(bb_model.inputs, [target_layer.output, bb_model.output])

    # Pass through input preprocessing layers before backbone
    x = tf.cast(image_array, tf.float32)
    for layer in model.layers[1:bb_idx]:
        x = layer(x)

    with tf.GradientTape() as tape:
        conv_outputs, bb_out = bb_submodel(x)
        tape.watch(conv_outputs)
        
        # Pass through remaining layers after backbone
        post_x = bb_out
        for layer in model.layers[bb_idx + 1:]:
            post_x = layer(post_x)
            
        if class_index is None:
            class_index = tf.argmax(post_x[0])
            
        loss = post_x[:, class_index]

    # Calculate gradients of top predicted class with respect to conv_outputs
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv_outputs feature map channels by pooled gradients
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    # Apply ReLU to keep only positive importance
    cam = np.maximum(cam.numpy(), 0.0)
    max_val = np.max(cam)
    if max_val > 1e-10:
        cam = cam / max_val
    else:
        cam = np.zeros_like(cam)

    return cam


def overlay_heatmap(original_pil, heatmap_2d, alpha=0.45, colormap_name="jet"):
    """
    Overlays a Grad-CAM heatmap onto the original PIL image.
    Returns:
      overlay_pil: PIL.Image of original blended with heatmap.
      heatmap_colored_pil: PIL.Image of heatmap alone with color palette.
      mask_pil: PIL.Image of isolated lesion activation mask.
      elar_percentage: Estimated Lesion Area Ratio (percentage of leaf area strongly activated).
    """
    w, h = original_pil.size
    
    # Resize heatmap to original image dimensions with bicubic interpolation
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap_2d))
    heatmap_resized = heatmap_img.resize((w, h), Image.Resampling.BICUBIC)
    heatmap_arr = np.asarray(heatmap_resized, dtype=np.float32) / 255.0

    # Apply Matplotlib colormap (compatible with Matplotlib 3.7+ and legacy)
    import matplotlib as mpl
    try:
        cmap = mpl.colormaps[colormap_name]
    except (AttributeError, KeyError):
        try:
            cmap = cm.get_cmap(colormap_name)
        except AttributeError:
            cmap = cm.viridis
    colored_heatmap = cmap(heatmap_arr)[:, :, :3]  # Drop alpha channel
    colored_heatmap = np.uint8(255 * colored_heatmap)
    heatmap_colored_pil = Image.fromarray(colored_heatmap)

    # Blend original image and heatmap
    original_arr = np.asarray(original_pil.convert("RGB"), dtype=np.float32)
    blended = (1.0 - alpha) * original_arr + alpha * (colored_heatmap.astype(np.float32))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    overlay_pil = Image.fromarray(blended)

    # Compute Lesion Activation Mask (threshold >= 0.40)
    lesion_mask_arr = (heatmap_arr >= 0.40).astype(np.uint8) * 255
    mask_pil = Image.fromarray(lesion_mask_arr, mode="L")

    # Estimated Lesion Area Ratio (ELAR%)
    elar_percentage = float(np.mean(heatmap_arr >= 0.40) * 100.0)

    return overlay_pil, heatmap_colored_pil, mask_pil, elar_percentage


def compute_severity(predicted_class, elar_percentage):
    """
    Computes disease severity grade and economic injury status based on ELAR%.
    """
    if predicted_class == "Healthy":
        return {
            "grade": "Healthy (Grade 0)",
            "short_label": "Healthy",
            "severity_index": 0.0,
            "status": "Optimum Health",
            "color": "#10B981",  # Emerald Green
            "action": "No chemical intervention needed. Maintain routine agronomic scouting."
        }

    if elar_percentage < 12.0:
        return {
            "grade": "Mild / Early Inoculum (Grade 1)",
            "short_label": "Mild",
            "severity_index": round(elar_percentage, 1),
            "status": "Below Economic Injury Threshold",
            "color": "#F59E0B",  # Amber
            "action": "Monitor canopy progression daily. Implement preventive biopesticides or biofungicides."
        }
    elif elar_percentage < 35.0:
        return {
            "grade": "Moderate Infection (Grade 2)",
            "short_label": "Moderate",
            "severity_index": round(elar_percentage, 1),
            "status": "At Economic Threshold — Intervention Justified",
            "color": "#F97316",  # Orange
            "action": "Fungicide application recommended within 48 hours to prevent upper canopy colonization."
        }
    else:
        return {
            "grade": "Severe / Necrotic Blight (Grade 3)",
            "short_label": "Severe",
            "severity_index": round(elar_percentage, 1),
            "status": "Critical Epidemic Level — High Yield Loss Risk",
            "color": "#EF4444",  # Red
            "action": "Immediate systemic therapeutic fungicide spray required. Restrict mechanical field movement to prevent spore dispersal."
        }
