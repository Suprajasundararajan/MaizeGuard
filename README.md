# MaizeGuard-Net Pro: Explainable AI & Pareto-Optimized Maize Pathology

An advanced deep-learning benchmark, Explainable AI (XAI) engine, and futuristic animated diagnostic platform for four-class maize (*Zea mays*) foliar disease screening: **Northern Corn Leaf Blight**, **Common Rust**, **Gray Leaf Spot**, and **Healthy Foliage**.

---

## 🌟 Key Upgrades & Features

### 1. 🎨 Modern Animated AgriTech UI
- **Animated AI Laser Scanner HUD**: Live sweeping laser scanline (`@keyframes scanline`) with animated corner reticles and pulse glow indicators.
- **Glassmorphism Theme**: Dark forest backdrop with emerald glass panels, floating hero badges (`@keyframes floatCard`), and gradient shimmer effects.
- **Dynamic Probability Progress Bars**: Smooth CSS width and color transitions (`@keyframes fillBar`) shifting from emerald green to amber to ruby red based on confidence.
- **Streamlined Image Acquisition**: Clean segmented input selector (`📁 Upload Image`, `📷 Live Camera`, `🧪 Quick Demo Sample`) without bulky image previews cluttering the viewport.
- **Exportable Diagnostic Certificate**: Instant generation and download of a printable agronomic diagnostic certificate.

---

## 🔬 Research Gaps Addressed

### 1. Black-Box Interpretability Gap (Explainable AI - XAI)
* **Problem**: Standard CNNs achieve high benchmark accuracy but frequently rely on spurious background artifacts (soil, fingers, bench shadows, lighting) rather than true biological lesions.
* **Solution**: Integrated Keras 3 **Grad-CAM (Gradient-weighted Class Activation Mapping)** engine computing spatial class activation gradients:
  $$A_{k}^{c} = \text{ReLU}\left( \sum_{k} \alpha_{k}^{c} A^{k} \right), \quad \alpha_{k}^{c} = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^{c}}{\partial A_{i, j}^{k}}$$
* **Verification**: Visual heatmap overlays (Jet, Viridis, Plasma, Magma) and isolated lesion masks prove the network attends strictly to authentic necrotic lesions (e.g., elliptical blights vs raised rust pustules vs vein-bounded gray spots).

### 2. Edge-Deployability & Pareto Frontier Gap
* **Problem**: Heavy architectures (ResNet50, DenseNet) are frequently proposed without evaluating memory and compute constraints of offline field devices in rural smallholder farming communities.
* **Solution**: Comprehensive Pareto efficiency benchmark across **MobileNetV2** (24.4 MB), **EfficientNetB0** (37.2 MB), and **ResNet50** (230.5 MB):
  - **ResNet50** delivers the highest holdout accuracy (**96.34%**) and Macro F1 (**95.40%**).
  - MobileNetV2 remains a smaller alternative with **94.12%** accuracy and a 24.4 MB model size for resource-constrained deployment.

### 3. Out-of-Distribution (OOD) & Uncertainty Gap
* **Problem**: Closed-world softmax models output dangerously overconfident false diagnoses when fed non-maize leaves, blank backgrounds, or corrupted camera inputs.
* **Solution**: Dual-stage epistemic uncertainty filter combining **Shannon Entropy**:
  $$H(P) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$
  with **Excess Green Index ($ExG = 2G - R - B$)** and image contrast heuristics to flag non-foliar or ambiguous inputs with automated anomaly warnings.

### 4. Qualitative-to-Quantitative Severity Index (ELAR%)
* **Problem**: Conventional agricultural AI tools only provide a categorical diagnosis ("Blight" vs "Healthy"), omitting disease severity—the primary metric determining whether Economic Injury Levels (EIL) are crossed and if chemical spraying is economically justified.
* **Solution**: Automated lesion segmentation via Grad-CAM activation thresholding ($\ge 0.40$), deriving the **Estimated Lesion Area Ratio (ELAR%)**:
  $$\text{ELAR}\% = \frac{\sum_{x, y} \mathbb{I}(\text{CAM}_{x, y} \ge 0.40)}{\text{Total Leaf Pixels}} \times 100$$
  Categorizes disease into **Mild (<15%)**, **Moderate (15–40%)**, and **Severe (>40%)** with tailored IPM action triggers.

---

## 🏗️ Repository Architecture

```
maizguard_net/
├── app.py                     # Modern animated Streamlit diagnostic web application
├── train.py                   # Transfer learning benchmark & training pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive system documentation
├── artifacts/                 # Saved models and benchmark artifacts
│   ├── best_model.keras       # Best-performing backbone (MobileNetV2)
│   ├── MobileNetV2.keras      # Fine-tuned MobileNetV2 weights (9.7 MB)
│   ├── EfficientNetB0.keras   # Fine-tuned EfficientNetB0 weights (17.1 MB)
│   ├── ResNet50.keras         # Fine-tuned ResNet50 weights (95.1 MB)
│   ├── model_comparison.csv   # Empirical metrics (Accuracy, F1, Latency, Size)
│   ├── evaluation_details.json# Per-class confusion matrices
│   └── metadata.json          # Pipeline configuration
├── sample_gallery/            # 1-click test images for each class
│   ├── Blight.jpeg
│   ├── Common_Rust.jpg
│   ├── Gray_Leaf_Spot.jpg
│   └── Healthy.jpg
├── src/                       # Core scientific & diagnostic modules
│   ├── __init__.py
│   ├── gradcam.py             # Keras 3 Grad-CAM, heatmaps, & ELAR% calculation
│   ├── ensemble.py            # Multi-model inference, soft voting, & TTA
│   ├── uncertainty.py         # Shannon Entropy & Out-of-Distribution detection
│   └── ipm.py                 # Integrated Pest Management & field dosage calculator
└── tests/
    └── test_pipeline.py       # Automated end-to-end integration test
```

---

## 🚀 Quick Start

### 1. Installation
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Integration Tests
```bash
python tests/test_pipeline.py
```

### 3. Launch the Animated Diagnostic Web App
```bash
streamlit run app.py
```

---

## 🚜 Integrated Pest Management & Dosage Calculator

MaizeGuard-Net Pro features an interactive field calculator:
- **Input**: Field size (acres/hectares), sprayer tank size (16L knapsack, 200L boom), and preferred formulation.
- **Output**: Total active chemical required, water dilution volume, number of sprayer tanks to mix, Pre-Harvest Interval (PHI in days), Re-Entry Interval (REI in hours), and required Personal Protective Equipment (PPE).

---

## 📊 Benchmark Results

Evaluated on a stratified 15% holdout test set ($N=629$ images) using the saved evaluation artifacts:

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Size (MB) | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 (Best)** | **96.34%** | **95.04%** | **95.84%** | **95.40%** | **230.5 MB** | **24,361s** |
| **EfficientNetB0** | 94.59% | 92.87% | 93.42% | 93.13% | 37.2 MB | 3,561s |
| **MobileNetV2** | 94.12% | 92.23% | 93.22% | 92.61% | 24.4 MB | 7,300s |

The table reports backbone test metrics generated by `train.py`; ensemble confidence is calculated at inference time and is not a separately evaluated holdout accuracy metric.
