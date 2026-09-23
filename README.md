# Dog vs Cat Classification with XAI

This repository provides a simple Dog vs Cat image classification pipeline for validating Explainable AI (XAI) methods before applying them to medical image classification tasks.

The main purpose of this project is to verify whether XAI methods can highlight image regions that contribute to a classifier's actual prediction.

The supported XAI methods are:

- Grad-CAM
- Grad-CAM++
- LIME
- SHAP
- Attention Rollout

For XAI analysis, the target class is always the class predicted by the classifier.

---

## 1. Project Purpose

The workflow is:

```text
Dog / Cat image
      |
      v
Classification model
      |
      v
Predicted class
      |
      +--> Grad-CAM
      |
      +--> Grad-CAM++
      |
      +--> LIME
      |
      +--> SHAP
      |
      +--> Attention Rollout
```

For example, if the classifier predicts:

```text
Prediction: Dog
Probability: 0.95
```

Grad-CAM, Grad-CAM++, LIME, and SHAP explain the model output for the predicted `Dog` class.

The ground-truth class is not used as the XAI target.

This makes it possible to analyze both correct and incorrect predictions.

For example:

```text
Ground truth: Dog
Prediction: Cat
```

In this case, XAI explains why the model predicted `Cat`.

---

## 2. Dataset Structure

The original dataset is expected to have the following structure:

```text
dog_cat/
├── Cat/
│   ├── cat_001.jpg
│   ├── cat_002.jpg
│   └── ...
│
└── Dog/
    ├── dog_001.jpg
    ├── dog_002.jpg
    └── ...
```

The original image files are not copied or reorganized.

Instead, the dataset is split using a CSV manifest.

---

## 3. Train / Validation / Test Split

The dataset is split into:

```text
Train      80%
Validation 10%
Test       10%
```

The split is stratified by class so that the Cat/Dog distribution is maintained across the three subsets.

The split information is stored in:

```text
outputs/manifests/seed42.csv
```

Example:

```csv
split,relative_path,label,class_name
train,Cat/cat_001.jpg,0,Cat
train,Dog/dog_001.jpg,1,Dog
val,Cat/cat_100.jpg,0,Cat
test,Dog/dog_200.jpg,1,Dog
```

Using a manifest makes the experiment reproducible without modifying the original dataset.

---

## 4. Repository Structure

```text
Dog_vs_Cat_classification/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── dog_cat.yaml
│
├── scripts/
│   ├── create_split_manifest.py
│   ├── train.py
│   ├── run_xai.py
│   └── evaluate_faithfulness.py
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── utils.py
│   │
│   └── xai/
│       ├── __init__.py
│       ├── common.py
│       ├── cam.py
│       ├── lime_explainer.py
│       ├── shap_explainer.py
│       ├── attention_rollout.py
│       └── plotting.py
│
└── outputs/
    ├── manifests/
    ├── classification/
    ├── xai/
    └── faithfulness/
```

---

## 5. Supported Classification Models

The following models are supported:

- ResNet50
- DenseNet201
- ViT-B/16

The final classification layer is replaced for binary classification:

```text
Cat
Dog
```

The classification problem is implemented using two output logits with `CrossEntropyLoss`.

---

## 6. XAI Methods

### Grad-CAM

Grad-CAM uses gradients flowing into the final convolutional feature maps to identify image regions associated with the predicted class.

Supported models:

```text
ResNet50
DenseNet201
```

---

### Grad-CAM++

Grad-CAM++ is an extension of Grad-CAM that uses a more detailed gradient-weighting mechanism.

Supported models:

```text
ResNet50
DenseNet201
```

---

### LIME

LIME perturbs superpixels of the input image and measures how those changes affect the probability of the predicted class.

The visualization shows image regions that positively support the classifier's prediction.

Supported models:

```text
ResNet50
DenseNet201
ViT-B/16
```

---

### SHAP

SHAP estimates the contribution of image regions to the model output.

In this project, SHAP is calculated for the classifier's predicted class.

Supported models:

```text
ResNet50
DenseNet201
ViT-B/16
```

---

### Attention Rollout

Attention Rollout combines attention matrices across Transformer layers to visualize how information propagates from the CLS token to image patches.

Supported model:

```text
ViT-B/16
```

Attention Rollout is different from the other methods because it is not directly class-specific.

Therefore, it is reported as:

```text
Attention Rollout (class-agnostic)
```

---

## 7. XAI Target

The XAI target is always:

```text
target_class = predicted_class
```

For example:

```text
Model prediction:
Dog = 0.93
Cat = 0.07
```

The XAI methods explain the `Dog` output.

If the model incorrectly predicts Cat:

```text
Ground truth: Dog
Prediction: Cat
```

the XAI target is still:

```text
Cat
```

This allows the visualization to answer:

```text
Which image regions contributed to the model's actual decision?
```

rather than:

```text
Which regions correspond to the correct class?
```

---

## 8. Installation

Install the required Python packages:

```bash
pip install \
  torch \
  torchvision \
  numpy \
  pandas \
  pillow \
  pyyaml \
  scikit-learn \
  scikit-image \
  matplotlib \
  grad-cam \
  lime \
  shap
```

Alternatively:

```bash
pip install -r requirements.txt
```

---

## 9. Configuration

The experiment is controlled by:

```text
configs/dog_cat.yaml
```

Example:

```yaml
seed: 42

device:
  require_cuda: true

dataset:
  root: /media/share/Member/ueki/datasets/dog_cat

  classes:
    - Cat
    - Dog

  manifest_dir: outputs/manifests

  split:
    train: 0.8
    val: 0.1
    test: 0.1

image:
  resize_size: 256
  image_size: 224

models:

  resnet50:
    enabled: true
    pretrained: true
    batch_size: 32
    lr: 0.0001

  densenet201:
    enabled: false
    pretrained: true
    batch_size: 32
    lr: 0.0001

  vit_b_16:
    enabled: false
    pretrained: true
    batch_size: 16
    lr: 0.00001

train:
  epochs: 20
  num_workers: 0
  weight_decay: 0.0001

xai:

  backbones:
    - resnet50

  num_images: 10
  sample_seed: 42

  methods:

    gradcam:
      enabled: true

    gradcam_plus_plus:
      enabled: true

    lime:
      enabled: true

    shap:
      enabled: true

    attention_rollout:
      enabled: true

  lime:
    num_samples: 500
    num_features: 10

  shap:
    nsamples: 64
    background_size: 4

faithfulness:

  ratios:
    - 0.10
    - 0.20
    - 0.30

  random_repeats: 5

output:
  classification_root: outputs/classification
  xai_root: outputs/xai
  faithfulness_root: outputs/faithfulness
```

---

## 10. Create the Dataset Split

Create the stratified 8:1:1 split:

```bash
python scripts/create_split_manifest.py \
  --config configs/dog_cat.yaml
```

The output is:

```text
outputs/manifests/seed42.csv
```

The original Dog/Cat dataset is not modified.

---

## 11. Train the Classifier

Run:

```bash
python scripts/train.py \
  --config configs/dog_cat.yaml
```

For ResNet50, the best validation checkpoint is stored in:

```text
outputs/classification/
└── resnet50/
    └── seed42/
        └── best_model.pth
```

Models are selected using validation performance and evaluated on the held-out test set.

---

## 12. Generate XAI Visualizations

Run:

```bash
python scripts/run_xai.py \
  --config configs/dog_cat.yaml
```

The same randomly selected test images are used for all enabled XAI methods.

Example output:

```text
outputs/xai/
└── resnet50/
    └── seed42/
        ├── figures/
        │   ├── Cat__cat_001_xai.png
        │   ├── Dog__dog_001_xai.png
        │   └── ...
        │
        ├── heatmaps/
        │   ├── gradcam/
        │   ├── gradcam_plus_plus/
        │   ├── lime/
        │   └── shap/
        │
        └── xai_metadata.csv
```

Each visualization contains:

```text
Input
Grad-CAM
Grad-CAM++
LIME
SHAP
```

when using a CNN model.

For ViT-B/16:

```text
Input
LIME
SHAP
Attention Rollout (class-agnostic)
```

is generated.

---

## 13. Raw XAI Heatmaps

Raw heatmaps are saved as NumPy arrays:

```text
*.npy
```

For example:

```text
outputs/xai/resnet50/seed42/heatmaps/
├── gradcam/
├── gradcam_plus_plus/
├── lime/
└── shap/
```

These raw heatmaps can be used for further quantitative analysis.

---

## 14. XAI Metadata

Prediction information is stored in:

```text
xai_metadata.csv
```

Example columns:

```text
case_id
relative_path
source_class
backbone
predicted_class_index
predicted_class
predicted_probability
xai_target_class
xai_target_source
figure_path
```

The important fields are:

```text
predicted_class
predicted_probability
xai_target_class
```

The value of:

```text
xai_target_source
```

is:

```text
predicted_class
```

because the XAI methods explain the model's actual prediction.

---

## 15. Faithfulness Evaluation

Visual inspection alone is not sufficient to determine whether an XAI heatmap identifies regions that truly influence the classifier.

Therefore, this project also provides a perturbation-based faithfulness evaluation.

For each XAI heatmap, the most important:

```text
10%
20%
30%
```

of image regions are masked.

The model is then evaluated again.

For example:

```text
Original Dog probability:
0.95

After masking top 20% Grad-CAM region:
0.42

Probability drop:
0.53
```

A random region covering the same percentage of the image is also masked as a baseline.

Example:

```text
Grad-CAM probability drop:
0.53

Random probability drop:
0.07
```

If removing the XAI-selected region causes a substantially larger decrease than removing a random region, this provides evidence that the highlighted region was important to the classifier.

Run:

```bash
python scripts/evaluate_faithfulness.py \
  --config configs/dog_cat.yaml
```

Results are saved to:

```text
outputs/faithfulness/
└── faithfulness_results.csv
```

Example columns:

```text
image
predicted_class
method
ratio
baseline_probability
xai_masked_probability
xai_probability_drop
random_masked_probability
random_probability_drop
xai_minus_random_drop
```

The main quantity of interest is:

```text
xai_minus_random_drop
```

A larger positive value indicates that removing the XAI-selected region affected the classifier more strongly than removing an equally sized random region.

---

## 16. Recommended Experimental Workflow

Run the scripts in the following order:

### Step 1: Create train/validation/test split

```bash
python scripts/create_split_manifest.py \
  --config configs/dog_cat.yaml
```

### Step 2: Train classifier

```bash
python scripts/train.py \
  --config configs/dog_cat.yaml
```

### Step 3: Generate XAI explanations

```bash
python scripts/run_xai.py \
  --config configs/dog_cat.yaml
```

### Step 4: Evaluate XAI faithfulness

```bash
python scripts/evaluate_faithfulness.py \
  --config configs/dog_cat.yaml
```

The complete workflow is:

```text
dog_cat/
├── Cat/
└── Dog/
     |
     v
Stratified 8:1:1 split
     |
     v
Train classifier
     |
     v
Test prediction
     |
     v
Predicted class
     |
     +-----------------------------+
     |          |          |       |
     v          v          v       v
 Grad-CAM    Grad-CAM++   LIME    SHAP

ViT:
LIME / SHAP / Attention Rollout

     |
     v
Visual inspection
     |
     v
Faithfulness evaluation
     |
     +--> Mask XAI-important region
     |
     +--> Mask random region
     |
     v
Compare predicted-class probability drop
```

---

## 17. Relationship to the Chest X-ray Experiment

This repository is intentionally separated from the Chest X-ray classification repository.

Its purpose is to validate the XAI implementation using an easier visual classification task in which the relevant object, Dog or Cat, can be visually identified.

The same XAI methodology can then be applied to the Chest X-ray classification experiment:

```text
Dog vs Cat
    |
    v
Validate XAI implementation
    |
    v
Grad-CAM
Grad-CAM++
LIME
SHAP
Attention Rollout
    |
    v
Apply the same methodology
to Chest X-ray classification
```

The XAI target definition is kept consistent between the two projects:

```text
XAI target = classifier predicted class
```

---

## 18. Notes

- Grad-CAM and Grad-CAM++ are primarily used with CNN models.
- Attention Rollout is used with ViT models.
- Attention Rollout is class-agnostic and should not be interpreted in exactly the same way as class-specific attribution methods.
- LIME and SHAP can be applied to both CNN and ViT classifiers.
- XAI heatmaps should not automatically be interpreted as literal model "vision." They represent attribution or attention information depending on the method.
- Faithfulness evaluation is included to complement qualitative visualization.
- Dataset images, checkpoints, XAI outputs, and generated NumPy heatmaps should not be committed to Git.