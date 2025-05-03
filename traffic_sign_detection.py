import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from skimage.transform import resize
from skimage import feature
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Setup directories ===
DATA_ROOT = os.path.join("data", "traffic_sign_detection")
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "annotations")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")

# ========================
# PHASE 1: CLASSIFICATION
# ========================


def load_cropped_signs(annotation_dir, images_dir):
    """Load and crop traffic-sign objects from annotations."""
    cropped_images, labels = [], []

    for xml_file in os.listdir(annotation_dir):
        xml_path = os.path.join(annotation_dir, xml_file)
        root = ET.parse(xml_path).getroot()

        image_name = root.find("filename").text
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)

        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls == "trafficlight":
                continue

            bb = obj.find("bndbox")
            x1, y1 = int(bb.find("xmin").text), int(bb.find("ymin").text)
            x2, y2 = int(bb.find("xmax").text), int(bb.find("ymax").text)

            cropped = image[y1:y2, x1:x2]
            cropped_images.append(cropped)
            labels.append(cls)

    return cropped_images, labels


# 1) Load and crop dataset
images_cropped, labels_cropped = load_cropped_signs(ANNOTATIONS_DIR, IMAGES_DIR)
print(f"Loaded objects: {len(images_cropped)}")
CLASSES = sorted(set(labels_cropped))
print(f"Classes: {CLASSES}\n")


# 2) Extract HOG features
def extract_hog_features(image):
    """Convert to grayscale, resize, then compute HOG feature vector."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = resize(
        image.astype(np.float32), output_shape=(32, 32), anti_aliasing=True
    )

    hog_vector = feature.hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2",
        feature_vector=True,
    )

    return hog_vector


# Build feature matrix
le = LabelEncoder()
X = np.array([extract_hog_features(im) for im in images_cropped])
y_encoded = le.fit_transform(labels_cropped)
print(f"Feature matrix shape: {X.shape}\n")


# 3) Split into train/val/test (70/15/15)
RS = 0
TEST_RATIO = 0.15
VAL_RATIO = 0.15

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=TEST_RATIO, random_state=RS, stratify=y_encoded
)

val_of_temp = VAL_RATIO / (1 - TEST_RATIO)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_of_temp, random_state=RS, stratify=y_temp
)

print(f"Train/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}\n")


# 4) Scale features and train SVM
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

svm_clf = SVC(kernel="rbf", C=0.5, probability=True, random_state=RS)
svm_clf.fit(X_train_s, y_train)

# 5) Evaluate classification accuracy
for split_name, X_split, y_split in [
    ("Validation", X_val_s, y_val),
    ("Test", X_test_s, y_test),
]:
    acc = accuracy_score(y_split, svm_clf.predict(X_split))
    print(f"{split_name} Accuracy: {acc:.4f}")


# ========================
# PHASE 2: LOCALIZATION & EVALUATION
# ========================


def generate_sliding_windows(img, window_sizes, stride):
    """Yield window coords for each size and stride."""
    H, W = img.shape[:2]
    for h, w in window_sizes:
        for y in range(0, H - h + 1, stride):
            for x in range(0, W - w + 1, stride):
                yield x, y, x + w, y + h


def build_pyramid(img, scale=0.8, min_size=(30, 30)):
    """Return list of (scaled_image, scale_factor)."""
    pyramid = [(img, 1.0)]
    current = img

    while True:
        new_h = int(current.shape[0] * scale)
        new_w = int(current.shape[1] * scale)

        if new_h < min_size[0] or new_w < min_size[1]:
            break

        current = cv2.resize(current, (new_w, new_h))
        pyramid.append((current, pyramid[-1][1] * scale))

    return pyramid


def compute_iou(boxA, boxB):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)

    return inter_area / (areaA + areaB - inter_area + 1e-6)


def visualize_bbox(image, boxes, label_encoder, save_path=None):
    """
    Visualize bounding boxes on the image.
    :param image: The image on which boxes will be drawn
    :param boxes: List of boxes, each as [x1, y1, x2, y2, class_id, score]
    :param label_encoder: The label encoder to decode class ids back to names
    :param save_path: The path to save the image (optional)
    """
    for box in boxes:
        x1, y1, x2, y2, class_id, score = box
        label = label_encoder.inverse_transform([class_id])[0]

        # Draw rectangle around detected object
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label the bounding box with class name and score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Display the image with bounding boxes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    if save_path:
        cv2.imwrite(save_path, image)


def compute_ap(gt_boxes, pred_boxes, iou_thr=0.5):
    """Compute AP using 11-point interpolation."""
    pred_sorted = sorted(pred_boxes, key=lambda x: x[2], reverse=True)
    TP, FP = [], []
    matched = set()

    for box, cls, score in pred_sorted:
        ious = [
            (compute_iou(box, gt_box), idx)
            for idx, (gt_box, gt_cls) in enumerate(gt_boxes)
            if gt_cls == cls
        ]

        if ious and max(iou for iou, _ in ious) >= iou_thr:
            best_idx = max(ious, key=lambda x: x[0])[1]
            if best_idx not in matched:
                TP.append(1)
                FP.append(0)
                matched.add(best_idx)
            else:
                TP.append(0)
                FP.append(1)
        else:
            TP.append(0)
            FP.append(1)

    tp_cum = np.cumsum(TP)
    fp_cum = np.cumsum(FP)
    recall = tp_cum / len(gt_boxes)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)

    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0
        ap += p / 11

    return ap


# Detection parameters
WINDOW_SIZES = [(32, 32), (64, 64), (128, 128)]
STRIDE = 12
CONF_THRESH = 0.95
IOU_NMS = 0.3


""" # 2A) Localization on Validation Set (visualize only)
print("\nLocalization on VALIDATION set (visualize only):")
val_count = len(X_train_s) + len(X_val_s)
processed = 0

for xml_file in os.listdir(ANNOTATIONS_DIR):
    if processed >= val_count:
        break

    root = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file)).getroot()
    img_name = root.find("filename").text
    img = cv2.imread(os.path.join(IMAGES_DIR, img_name))
    processed += 1
    detections = []

    for img_s, scale_val in build_pyramid(img):
        for x1, y1, x2, y2 in generate_sliding_windows(img_s, WINDOW_SIZES, STRIDE):

            patch = img_s[y1:y2, x1:x2]
            feat = extract_hog_features(patch)
            prob = svm_clf.predict_proba([scaler.transform([feat])[0]])[0]

            if prob.max() < CONF_THRESH:
                continue

            cls_pred = le.inverse_transform([prob.argmax()])[0]
            detections.append(
                (
                    [
                        int(x1 / scale_val),
                        int(y1 / scale_val),
                        int(x2 / scale_val),
                        int(y2 / scale_val),
                    ],
                    cls_pred,
                    prob.max(),
                )
            )

    # Apply simple NMS
    final_detections = []
    for box, cls, sc in sorted(detections, key=lambda x: x[2], reverse=True):
        if all(compute_iou(box, fb) <= IOU_NMS for fb, _, _ in final_detections):
            final_detections.append((box, cls, sc))

    # Visualize
    viz_boxes = [
        [*box, le.transform([cls])[0], sc] for box, cls, sc in final_detections
    ]
    visualize_bbox(img, viz_boxes, le) """
    
# 2A) Localization on Validation Set (visualize only)
print("\nLocalization on VALIDATION set (visualize only):")
val_count = len(X_val)  # Sử dụng số lượng ảnh trong tập validation
processed = 0

for xml_file in os.listdir(ANNOTATIONS_DIR):
    if processed >= val_count:
        break

    root = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file)).getroot()
    img_name = root.find("filename").text
    img = cv2.imread(os.path.join(IMAGES_DIR, img_name))
    processed += 1
    detections = []

    for img_s, scale_val in build_pyramid(img):
        for x1, y1, x2, y2 in generate_sliding_windows(img_s, WINDOW_SIZES, STRIDE):

            patch = img_s[y1:y2, x1:x2]
            feat = extract_hog_features(patch)
            prob = svm_clf.predict_proba([scaler.transform([feat])[0]])[0]

            if prob.max() < CONF_THRESH:
                continue

            cls_pred = le.inverse_transform([prob.argmax()])[0]
            detections.append(
                (
                    [
                        int(x1 / scale_val),
                        int(y1 / scale_val),
                        int(x2 / scale_val),
                        int(y2 / scale_val),
                    ],
                    cls_pred,
                    prob.max(),
                )
            )

    # Apply simple NMS
    final_detections = []
    for box, cls, sc in sorted(detections, key=lambda x: x[2], reverse=True):
        if all(compute_iou(box, fb) <= IOU_NMS for fb, _, _ in final_detections):
            final_detections.append((box, cls, sc))

    # Visualize
    viz_boxes = [
        [*box, le.transform([cls])[0], sc] for box, cls, sc in final_detections
    ]
    visualize_bbox(img, viz_boxes, le)



""" # 2B) Localization on Test Set (save results + compute mAP)
print("\nLocalization on TEST set (save & evaluate):")
out_dir = os.path.join(DATA_ROOT, "output_test")
os.makedirs(out_dir, exist_ok=True)

gt_dict, pred_dict = {}, {}

for xml_file in os.listdir(ANNOTATIONS_DIR):
    root = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file)).getroot()
    img_name = root.find("filename").text
    img = cv2.imread(os.path.join(IMAGES_DIR, img_name))

    # Ground-truth boxes
    gt_list = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls == "trafficlight":
            continue

        bb = obj.find("bndbox")
        coords = [int(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        gt_list.append((coords, cls))

    gt_dict[img_name] = gt_list

    # Detection
    detections = []
    for img_s, scale_val in build_pyramid(img):
        for x1, y1, x2, y2 in generate_sliding_windows(img_s, WINDOW_SIZES, STRIDE):

            patch = img_s[y1:y2, x1:x2]
            feat = extract_hog_features(patch)
            prob = svm_clf.predict_proba([scaler.transform([feat])[0]])[0]

            if prob.max() < CONF_THRESH:
                continue

            cls_pred = le.inverse_transform([prob.argmax()])[0]
            detections.append(
                (
                    [
                        int(x1 / scale_val),
                        int(y1 / scale_val),
                        int(x2 / scale_val),
                        int(y2 / scale_val),
                    ],
                    cls_pred,
                    prob.max(),
                )
            )

    # Simple NMS
    final_detections = []
    for box, cls, sc in sorted(detections, key=lambda x: x[2], reverse=True):
        if all(compute_iou(box, fb) <= IOU_NMS for fb, _, _ in final_detections):
            final_detections.append((box, cls, sc))

    pred_dict[img_name] = final_detections

    # Save visualization
    viz_boxes = [
        [*box, le.transform([cls])[0], sc] for box, cls, sc in final_detections
    ]
    output_path = os.path.join(out_dir, img_name)
    visualize_bbox(img.copy(), viz_boxes, le, save_path=output_path)
    print(f"Saved {output_path}")

# Compute AP per class and mAP
aps = []
for cls in CLASSES:
    gt_accum, pred_accum = [], []

    for img_id in gt_dict:
        gt_accum += [g for g in gt_dict[img_id] if g[1] == cls]
        pred_accum += [p for p in pred_dict[img_id] if p[1] == cls]

    ap_val = compute_ap(gt_accum, pred_accum)
    print(f"AP {cls}: {ap_val:.4f}")
    aps.append(ap_val)

print(f"Overall mAP: {np.mean(aps):.4f}")
 """

# 2B) Localization on TEST set (save & evaluate)
print("\nLocalization on TEST set (save & evaluate):")
out_dir = os.path.join(DATA_ROOT, "output_test")
os.makedirs(out_dir, exist_ok=True)

gt_dict, pred_dict = {}, {}

# Duyệt qua tất cả các ảnh trong tập test
for xml_file in os.listdir(ANNOTATIONS_DIR):
    # Chỉ xử lý các ảnh trong tập test (tập test có thể được phân chia từ X_test)
    if not any(img_name in xml_file for img_name in [img_name for img_name, _ in zip(X_test, y_test)]):
        continue

    root = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file)).getroot()
    img_name = root.find("filename").text
    img = cv2.imread(os.path.join(IMAGES_DIR, img_name))

    # Ground-truth boxes
    gt_list = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls == "trafficlight":
            continue

        bb = obj.find("bndbox")
        coords = [int(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        gt_list.append((coords, cls))

    gt_dict[img_name] = gt_list

    # Detection
    detections = []
    for img_s, scale_val in build_pyramid(img):
        for x1, y1, x2, y2 in generate_sliding_windows(img_s, WINDOW_SIZES, STRIDE):

            patch = img_s[y1:y2, x1:x2]
            feat = extract_hog_features(patch)
            prob = svm_clf.predict_proba([scaler.transform([feat])[0]])[0]

            if prob.max() < CONF_THRESH:
                continue

            cls_pred = le.inverse_transform([prob.argmax()])[0]
            detections.append(
                (
                    [
                        int(x1 / scale_val),
                        int(y1 / scale_val),
                        int(x2 / scale_val),
                        int(y2 / scale_val),
                    ],
                    cls_pred,
                    prob.max(),
                )
            )

    # Apply NMS
    final_detections = []
    for box, cls, sc in sorted(detections, key=lambda x: x[2], reverse=True):
        if all(compute_iou(box, fb) <= IOU_NMS for fb, _, _ in final_detections):
            final_detections.append((box, cls, sc))

    pred_dict[img_name] = final_detections

    # Save visualization
    viz_boxes = [
        [*box, le.transform([cls])[0], sc] for box, cls, sc in final_detections
    ]
    output_path = os.path.join(out_dir, img_name)
    visualize_bbox(img.copy(), viz_boxes, le, save_path=output_path)
    print(f"Saved {output_path}")

# Compute AP per class and mAP
aps = []
for cls in CLASSES:
    gt_accum, pred_accum = [], []

    for img_id in gt_dict:
        gt_accum += [g for g in gt_dict[img_id] if g[1] == cls]
        pred_accum += [p for p in pred_dict[img_id] if p[1] == cls]

    ap_val = compute_ap(gt_accum, pred_accum)
    print(f"AP {cls}: {ap_val:.4f}")
    aps.append(ap_val)

print(f"Overall mAP: {np.mean(aps):.4f}")
