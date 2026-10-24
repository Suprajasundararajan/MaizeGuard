import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.gradcam as gc
import src.ensemble as ens
import src.uncertainty as unc
import src.ipm as ipm
from PIL import Image

print("All modules imported successfully!")

# Test IPM calculator
calc = ipm.calculate_spray_dosage("Blight", 2.5, 16)
print("Dosage calculation test:", calc["product_name"], "Total tanks:", calc["total_tanks"])

# Test image loading and inference on sample image
sample_img = Image.open("sample_gallery/Blight.jpeg")
pred = ens.predict_single_backbone("best_model", sample_img, use_tta=True)
print(f"Prediction on sample Blight: {pred['predicted_class']} (Confidence: {pred['confidence']:.2%}, Latency: {pred['latency_ms']:.1f}ms)")

# Test uncertainty
unc_res = unc.evaluate_image_quality_and_ood(sample_img, pred["probabilities"])
print(f"Uncertainty test: {unc_res['uncertainty_tier']} (Entropy bits: {unc_res['entropy_bits']})")

# Test GradCAM
m = ens.get_model("best_model")
cam = gc.compute_gradcam(m, ens.preprocess_image(sample_img))
overlay, heat, mask, elar = gc.overlay_heatmap(sample_img, cam, alpha=0.45)
print(f"GradCAM test: ELAR%: {elar:.2f}%, Overlay size: {overlay.size}")
sev = gc.compute_severity(pred["predicted_class"], elar)
print(f"Severity test: {sev['grade']}")
