# Project_Traffic_Sign_Detection

## Object Detection Pipeline – 2 Phases

The traffic sign recognition task is divided into two main phases:

---

### Phase 1: Classification

- `Goal`: Train a classification model capable of recognizing different types of traffic signs from pre-cropped images.

#### Steps:
- Extract object images from annotation files (based on bounding boxes).
- Preprocess images: convert to grayscale and resize to 32×32.
- Extract HOG (Histogram of Oriented Gradients) features.
- Encode labels using `LabelEncoder`.
- Train an SVM classifier with an RBF kernel.
- Evaluate model accuracy on both validation and test datasets.

> 📌 **Output**: A trained SVM classifier capable of distinguishing between multiple types of traffic signs.

✅ **Validation Accuracy**: **97.21%**

---

### Phase 2: Localization & Evaluation

- `Goal`: Detect the location of traffic signs in the original images and evaluate detection accuracy.

#### Steps:
- Apply an image pyramid to generate multi-scale image versions.
- Use a sliding window to scan each scale with different window sizes.
- For each window:
  - Extract HOG features.
  - Classify using the pre-trained SVM model.
- Keep windows with classification confidence above a threshold.
- Apply Non-Maximum Suppression (NMS) to remove redundant overlapping bounding boxes.
- Compare predictions with ground truth using IoU (Intersection over Union).
- Compute AP (Average Precision) per class and overall mAP (mean Average Precision).

> 📌 **Output**: Detection results visualized and saved in the `output_test` directory.
