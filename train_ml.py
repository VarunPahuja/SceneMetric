import os
import cv2
import numpy as np
import pandas as pd
import joblib

from modules.shot_scale import analyse_frame

# ===========================
# MEDIAPIPE SETUP (global, init once)
# Works across mediapipe versions — solutions.pose is the stable path
# ===========================
try:
    import mediapipe as mp
    _pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.4
    )
    _MEDIAPIPE_OK = True
    print("✅ MediaPipe loaded")
except Exception as e:
    print(f"⚠️  MediaPipe unavailable ({e}) — keypoint features will be zero")
    _pose = None
    _MEDIAPIPE_OK = False

# ===========================
# CONFIG
# ===========================
DATASET_PATH = "datasets/processed"

label_map   = {"CLOSE": 0, "MEDIUM": 1, "WIDE": 2}
label_names = {0: "CLOSE", 1: "MEDIUM", 2: "WIDE"}

# ===========================
# FEATURE ENGINEERING
# ===========================
def compute_extended_features(features, img=None):
    """
    Extends the raw feature dict with new discriminative features.
    img (H x W x 3, RGB) is optional — used for spatial frequency + keypoints.
    """
    f = dict(features)

    face        = f.get("face_ratio", 0.0)
    center_edge = f.get("center_edge_ratio", 0.0)
    entropy     = f.get("entropy_score", 0.0)

    # --- Banded face_ratio features ---
    f["face_ratio_sq"]       = face ** 2
    f["face_ratio_log"]      = np.log1p(face)
    f["face_in_medium_band"] = float(0.01 <= face <= 0.045)

    # --- Bokeh proxy ---
    f["bokeh_proxy"] = center_edge / (entropy + 1e-6)

    # --- Spatial frequency ---
    if img is not None:
        gray      = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        fft_shift = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.abs(fft_shift)
        h, w      = magnitude.shape
        cy, cx    = h // 2, w // 2
        radius    = min(h, w) // 8
        Y, X      = np.ogrid[:h, :w]
        mask      = (X - cx)**2 + (Y - cy)**2 <= radius**2
        f["freq_low_high_ratio"] = magnitude[mask].sum() / (magnitude[~mask].sum() + 1e-6)
    else:
        f["freq_low_high_ratio"] = 0.0

    # --- MediaPipe body keypoint count ---
    # CLOSE:  head/shoulders only  -> low  (0-5)
    # MEDIUM: waist-up             -> mid  (6-17)
    # WIDE:   full body / no body  -> high (18-33) or 0
    if img is not None and _MEDIAPIPE_OK:
        result  = _pose.process(img)
        visible = 0
        if result.pose_landmarks:
            visible = sum(
                1 for lm in result.pose_landmarks.landmark
                if lm.visibility > 0.4
            )
        f["keypoint_count"]        = float(visible)
        f["keypoint_count_norm"]   = visible / 33.0
        f["keypoints_medium_band"] = float(5 <= visible <= 17)
    else:
        f["keypoint_count"]        = 0.0
        f["keypoint_count_norm"]   = 0.0
        f["keypoints_medium_band"] = 0.0

    return f


FEATURE_COLS = [
    "face_ratio",
    "center_face_score",
    "entropy_score",
    "center_edge_ratio",
    "depth_ratio",
    # extended
    "face_ratio_sq",
    "face_ratio_log",
    "face_in_medium_band",
    "bokeh_proxy",
    "freq_low_high_ratio",
    # mediapipe keypoints
    "keypoint_count",
    "keypoint_count_norm",
    "keypoints_medium_band",
]


def features_to_vector(f):
    return np.array([f.get(c, 0.0) for c in FEATURE_COLS], dtype=np.float32)


# ===========================
# RULE-BASED CLASSIFIER
# ===========================
def rule_based(features):
    """
    Fires only on unambiguous extremes.
    face_in_medium_band guard stops MEDIUM frames being mislabelled CLOSE.
    """
    face           = features["face_ratio"]
    depth          = features["depth_ratio"]
    in_medium_band = 0.01 <= face <= 0.045

    if face > 0.06 and depth < 1.1 and not in_medium_band:
        return 0  # CLOSE

    if face < 0.008 and depth > 1.7:
        return 2  # WIDE

    return None   # -> ML cascade


# ===========================
# CASCADE PREDICT
# ===========================
def cascade_predict(X_vec, clf1, clf2, scaler1, scaler2,
                    thresh_close=0.55, thresh_wide=0.55):
    """
    Stage 1: P(CLOSE) — if confident, return CLOSE.
    Stage 2: P(WIDE)  — if confident, return WIDE.
    Else: MEDIUM (residual).
    """
    x1         = scaler1.transform(X_vec.reshape(1, -1))
    prob_close = clf1.predict_proba(x1)[0][1]
    if prob_close >= thresh_close:
        return 0

    x2        = scaler2.transform(X_vec.reshape(1, -1))
    prob_wide = clf2.predict_proba(x2)[0][1]
    if prob_wide >= thresh_wide:
        return 2

    return 1  # MEDIUM


# ===========================
# BUILD DATASET
# ===========================
print("🚀 STARTING DATASET BUILD...\n")

X_raw, y_raw = [], []
file_count   = 0

for label in ["CLOSE", "MEDIUM", "WIDE"]:
    folder = os.path.join(DATASET_PATH, label)
    files  = os.listdir(folder)
    print(f"📂 {label} -> {len(files)} files")

    for file in files:
        path = os.path.join(folder, file)
        file_count += 1

        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            _, _, _, features = analyse_frame(img)
        except Exception as e:
            print(f"⚠️  Error processing {file}: {e}")
            continue

        ext = compute_extended_features(features, img)
        X_raw.append(features_to_vector(ext))
        y_raw.append(label_map[label])

        if len(X_raw) % 100 == 0:
            print(f"   Processed {len(X_raw)} samples...")

print(f"\nTotal files seen : {file_count}")
print(f"Valid samples    : {len(X_raw)}\n")

X = np.array(X_raw, dtype=np.float32)
y = np.array(y_raw)

# ===========================
# TRAIN / TEST SPLIT
# ===========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================
# SMOTE on MEDIUM (training only)
# ===========================
from imblearn.over_sampling import SMOTE

print("⚖️  Applying SMOTE to balance MEDIUM class...")
smote = SMOTE(
    sampling_strategy={1: int((y_train == 0).sum() * 0.85)},
    k_neighbors=3,
    random_state=42
)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE -> CLOSE:{(y_train_sm==0).sum()}  "
      f"MEDIUM:{(y_train_sm==1).sum()}  WIDE:{(y_train_sm==2).sum()}\n")

# ===========================
# STAGE 1: CLOSE vs NOT-CLOSE
# ===========================
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("🚀 Training Stage 1 — CLOSE vs {MEDIUM, WIDE}...")

y1   = (y_train_sm == 0).astype(int)
spw1 = (y1 == 0).sum() / ((y1 == 1).sum() + 1e-6)

scaler1  = StandardScaler()
X1_train = scaler1.fit_transform(X_train_sm)

clf1 = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw1,
    random_state=42, eval_metric="logloss"
)
clf1.fit(X1_train, y1)
print("✅ Stage 1 trained!\n")

# ===========================
# STAGE 2: WIDE vs MEDIUM
# (trained only on non-CLOSE samples)
# ===========================
print("🚀 Training Stage 2 — WIDE vs MEDIUM...")

mask2    = y_train_sm != 0
X2_base  = X_train_sm[mask2]
y2       = (y_train_sm[mask2] == 2).astype(int)
spw2     = (y2 == 0).sum() / ((y2 == 1).sum() + 1e-6)

scaler2  = StandardScaler()
X2_train = scaler2.fit_transform(X2_base)

clf2 = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw2,
    random_state=42, eval_metric="logloss"
)
clf2.fit(X2_train, y2)
print("✅ Stage 2 trained!\n")

# ===========================
# THRESHOLD CALIBRATION
# Calibrate on pre-SMOTE train set to avoid inflated F1
# ===========================
from sklearn.metrics import f1_score

print("🎯 Calibrating thresholds on pre-SMOTE training set...")

best_f1, best_tc, best_tw = 0, 0.55, 0.55

for tc in np.arange(0.40, 0.75, 0.05):
    for tw in np.arange(0.40, 0.75, 0.05):
        cal_preds = []
        for i in range(len(X_train)):          # pre-SMOTE X_train / y_train
            x    = X_train[i]
            rule = rule_based(dict(zip(FEATURE_COLS, x)))
            cal_preds.append(
                rule if rule is not None
                else cascade_predict(x, clf1, clf2, scaler1, scaler2, tc, tw)
            )
        f1 = f1_score(y_train, cal_preds, average="macro")
        if f1 > best_f1:
            best_f1, best_tc, best_tw = f1, tc, tw

print(f"   Best thresholds -> CLOSE: {best_tc:.2f}  WIDE: {best_tw:.2f}  "
      f"(macro-F1 on pre-SMOTE train: {best_f1:.3f})\n")

# ===========================
# HYBRID PREDICTION ON TEST
# ===========================
preds     = []
rule_used = 0
ml_used   = 0

for i in range(len(X_test)):
    x         = X_test[i]
    raw_feat  = dict(zip(FEATURE_COLS, x))
    rule_pred = rule_based(raw_feat)

    if rule_pred is not None:
        preds.append(rule_pred)
        rule_used += 1
    else:
        preds.append(
            cascade_predict(x, clf1, clf2, scaler1, scaler2, best_tc, best_tw)
        )
        ml_used += 1

preds = np.array(preds)

# ===========================
# EVALUATION
# ===========================
from sklearn.metrics import classification_report, confusion_matrix

print("===========================")
print("📈 RESULTS (HYBRID + CASCADE)")
print("===========================")
print(classification_report(y_test, preds, target_names=["CLOSE", "MEDIUM", "WIDE"]))

print("📉 CONFUSION MATRIX")
print(confusion_matrix(y_test, preds))

print("\n===========================")
print("⚙️  SYSTEM USAGE")
print("===========================")
print(f"Rule-based used : {rule_used}")
print(f"ML used         : {ml_used}")

# ===========================
# MISCLASSIFICATION DIAGNOSTIC
# True MEDIUM predicted as CLOSE
# ===========================
print("\n===========================")
print("🔍 MEDIUM -> CLOSE MISCLASSIFICATION AUDIT")
print("===========================")

miss_rows = []
for i in range(len(X_test)):
    if y_test[i] == 1 and preds[i] == 0:
        x        = X_test[i]
        raw_feat = dict(zip(FEATURE_COLS, x))
        fired    = rule_based(raw_feat) is not None
        row      = {"sample_idx": i, "rule_fired": fired}
        row.update({k: round(float(x[j]), 4) for j, k in enumerate(FEATURE_COLS)})
        miss_rows.append(row)

if miss_rows:
    df_miss = pd.DataFrame(miss_rows)
    n_rule  = int(df_miss["rule_fired"].sum())
    n_ml    = len(df_miss) - n_rule
    print(f"Total MEDIUM->CLOSE misses : {len(df_miss)}")
    print(f"  Caused by rule-based     : {n_rule}")
    print(f"  Caused by ML cascade     : {n_ml}")
    print("\nPer-sample breakdown:")
    print(df_miss[["sample_idx", "rule_fired", "face_ratio",
                    "depth_ratio", "keypoint_count",
                    "face_in_medium_band"]].to_string(index=False))
else:
    print("No MEDIUM->CLOSE misclassifications! ✅")

# ===========================
# FEATURE IMPORTANCE
# ===========================
print("\n===========================")
print("🧠 FEATURE IMPORTANCE")
print("===========================")

print("\n— Stage 1 (CLOSE detector) —")
for name, score in sorted(zip(FEATURE_COLS, clf1.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:25s}: {score:.4f}")

print("\n— Stage 2 (WIDE vs MEDIUM) —")
for name, score in sorted(zip(FEATURE_COLS, clf2.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:25s}: {score:.4f}")

# ===========================
# SAVE
# ===========================
os.makedirs("models", exist_ok=True)

joblib.dump(clf1,    "models/cascade_clf1.pkl")
joblib.dump(clf2,    "models/cascade_clf2.pkl")
joblib.dump(scaler1, "models/cascade_scaler1.pkl")
joblib.dump(scaler2, "models/cascade_scaler2.pkl")
joblib.dump({"thresh_close": best_tc, "thresh_wide": best_tw},
            "models/cascade_thresholds.pkl")
joblib.dump(FEATURE_COLS, "models/feature_cols.pkl")

print("\n💾 All models saved!")
print("   models/cascade_clf1.pkl")
print("   models/cascade_clf2.pkl")
print("   models/cascade_scaler1.pkl")
print("   models/cascade_scaler2.pkl")
print("   models/cascade_thresholds.pkl")
print("   models/feature_cols.pkl")