# Cell 1
!pip install tensorflow scikit-learn pandas numpy matplotlib seaborn -q

# Cell 2 - Part A: Heart Rate Prediction (Regression)

import os, warnings, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import pickle

tf.get_logger().setLevel("ERROR")

BLUE   = "#1e3a5f"
TEAL   = "#17a2b8"
CORAL  = "#e8533f"
GREEN  = "#28a745"
PURPLE = "#7F77DD"
GRAY   = "#6c757d"

sns.set_theme(style="whitegrid", palette=[BLUE, CORAL, TEAL, GREEN])
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("PART A - HEART RATE PREDICTION (REGRESSION)")
print("="*60)

# Generate dataset
print("\nGenerating synthetic heart rate dataset...")
N = 2000

age            = np.random.randint(18, 80, N).astype(float)
weight         = np.random.uniform(45, 120, N)
height         = np.random.uniform(150, 200, N)
bmi            = weight / ((height / 100) ** 2)
activity_level = np.random.choice([1,2,3,4,5], N, p=[0.1,0.2,0.3,0.25,0.15])
steps          = np.random.uniform(1000, 20000, N)
sleep_hours    = np.random.uniform(4, 10, N)
exercise_min   = np.random.uniform(0, 120, N)
stress_level   = np.random.uniform(1, 10, N)
caffeine_mg    = np.random.uniform(0, 400, N)

heart_rate = (
    75
    - 0.15 * (age - 40)
    + 0.12 * (bmi - 25)
    + 2.5  * activity_level
    - 0.002 * steps
    - 1.5  * (sleep_hours - 7)
    + 0.08 * exercise_min
    + 1.2  * stress_level
    + 0.02 * caffeine_mg
    + np.random.normal(0, 4, N)
)
heart_rate = np.clip(heart_rate, 45, 130)

df_hr = pd.DataFrame({
    "Age": age, "Weight_kg": weight, "Height_cm": height,
    "BMI": bmi.round(2), "Activity_Level": activity_level,
    "Steps_Daily": steps.round(0), "Sleep_Hours": sleep_hours,
    "Exercise_Min": exercise_min, "Stress_Level": stress_level,
    "Caffeine_mg": caffeine_mg, "Heart_Rate_BPM": heart_rate.round(1)
})

print(f"Dataset shape   : {df_hr.shape}")
print(f"Heart rate range: {df_hr.Heart_Rate_BPM.min():.1f} - {df_hr.Heart_Rate_BPM.max():.1f} BPM")
print(f"Mean heart rate : {df_hr.Heart_Rate_BPM.mean():.1f} BPM")

# EDA - Figure 1: Feature distributions
feat_cols_hr = [c for c in df_hr.columns if c != "Heart_Rate_BPM"]
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("Part A - Feature Distributions (Heart Rate Dataset)",
             fontsize=15, fontweight="bold", color=BLUE, y=0.98)
for ax, col in zip(axes.flatten(), feat_cols_hr):
    ax.hist(df_hr[col], bins=30, color=BLUE, alpha=0.75, edgecolor="white")
    ax.set_title(col, fontweight="bold", fontsize=10, color=BLUE)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=8)
axes.flatten()[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figA1_distributions.png", bbox_inches="tight", facecolor="white")
plt.show()
print("Figure A1 saved: Feature distributions")

# EDA - Figure 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr = df_hr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, ax=ax, linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap - Heart Rate Dataset",
             fontsize=13, fontweight="bold", color=BLUE, pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figA2_correlation.png", bbox_inches="tight", facecolor="white")
plt.show()
print("Figure A2 saved: Correlation heatmap")

# EDA - Figure 3: Scatter plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Heart Rate vs Key Features", fontsize=14,
             fontweight="bold", color=BLUE)
top_feats  = ["Age","BMI","Exercise_Min","Sleep_Hours","Activity_Level","Stress_Level"]
colors_sc  = [BLUE, CORAL, TEAL, GREEN, PURPLE, CORAL]
for ax, feat, color in zip(axes.flatten(), top_feats, colors_sc):
    ax.scatter(df_hr[feat], df_hr["Heart_Rate_BPM"],
               alpha=0.3, s=15, color=color)
    z = np.polyfit(df_hr[feat], df_hr["Heart_Rate_BPM"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df_hr[feat].min(), df_hr[feat].max(), 100)
    ax.plot(xs, p(xs), color="black", lw=2, ls="--", label="Trend")
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel("Heart Rate (BPM)", fontsize=10)
    ax.set_title(f"HR vs {feat}", fontweight="bold", color=BLUE, fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figA3_scatter.png", bbox_inches="tight", facecolor="white")
plt.show()
print("Figure A3 saved: Scatter plots")

# Preprocessing
print("\nPreprocessing data...")
X_hr = df_hr.drop("Heart_Rate_BPM", axis=1).values
y_hr = df_hr["Heart_Rate_BPM"].values

X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(
    X_hr, y_hr, test_size=0.20, random_state=42
)

scaler_hr = MinMaxScaler()
X_train_hr_sc = scaler_hr.fit_transform(X_train_hr)
X_test_hr_sc  = scaler_hr.transform(X_test_hr)
print(f"Train: {X_train_hr.shape[0]} | Test: {X_test_hr.shape[0]}")

# Build model
print("\nBuilding neural network...")
model_hr = keras.Sequential([
    layers.Input(shape=(X_train_hr_sc.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="linear")
])
model_hr.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="huber",
    metrics=["mae"]
)
model_hr.summary()

cb_list = [
    callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss"),
    callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6, monitor="val_loss")
]

print("\nTraining model...")
history_hr = model_hr.fit(
    X_train_hr_sc, y_train_hr,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    callbacks=cb_list,
    verbose=1
)
print(f"Training complete - {len(history_hr.history['loss'])} epochs")

# Evaluate
y_pred_hr = model_hr.predict(X_test_hr_sc, verbose=0).flatten()
mae  = mean_absolute_error(y_test_hr, y_pred_hr)
rmse = np.sqrt(mean_squared_error(y_test_hr, y_pred_hr))
r2   = r2_score(y_test_hr, y_pred_hr)

print("\n" + "="*60)
print("PART A - RESULTS")
print("="*60)
print(f"MAE  : {mae:.3f} BPM")
print(f"RMSE : {rmse:.3f} BPM")
print(f"R2   : {r2:.4f}")

# Figure 4: Evaluation dashboard
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history_hr.history["loss"], color=BLUE, lw=2, label="Train Loss")
ax1.plot(history_hr.history["val_loss"], color=CORAL, lw=2, ls="--", label="Val Loss")
ax1.set_title("Training & Validation Loss", fontweight="bold", color=BLUE, fontsize=12)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Huber Loss")
ax1.legend(fontsize=9)
ax1.spines[["top","right"]].set_visible(False)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history_hr.history["mae"], color=TEAL, lw=2, label="Train MAE")
ax2.plot(history_hr.history["val_mae"], color=GREEN, lw=2, ls="--", label="Val MAE")
ax2.set_title("Training & Validation MAE", fontweight="bold", color=BLUE, fontsize=12)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MAE (BPM)")
ax2.legend(fontsize=9)
ax2.spines[["top","right"]].set_visible(False)

ax3 = fig.add_subplot(gs[0, 2])
metric_names = ["MAE (BPM)", "RMSE (BPM)", "R2 Score"]
metric_vals  = [mae, rmse, r2]
bars = ax3.bar(metric_names, metric_vals, color=[BLUE, CORAL, GREEN], alpha=0.85, width=0.5)
for bar, val in zip(bars, metric_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
ax3.set_title("Regression Metrics", fontweight="bold", color=BLUE, fontsize=12)
ax3.spines[["top","right"]].set_visible(False)

ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_test_hr, y_pred_hr, alpha=0.4, s=20, color=BLUE)
lims = [min(y_test_hr.min(), y_pred_hr.min()),
        max(y_test_hr.max(), y_pred_hr.max())]
ax4.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
ax4.set_xlabel("Actual Heart Rate (BPM)", fontsize=10)
ax4.set_ylabel("Predicted Heart Rate (BPM)", fontsize=10)
ax4.set_title("Actual vs Predicted", fontweight="bold", color=BLUE, fontsize=12)
ax4.legend(fontsize=9)
ax4.spines[["top","right"]].set_visible(False)

ax5 = fig.add_subplot(gs[1, 1])
residuals = y_test_hr - y_pred_hr
ax5.scatter(y_pred_hr, residuals, alpha=0.4, s=20, color=TEAL)
ax5.axhline(0, color="red", lw=2, ls="--")
ax5.set_xlabel("Predicted Heart Rate (BPM)", fontsize=10)
ax5.set_ylabel("Residual (BPM)", fontsize=10)
ax5.set_title("Residual Plot", fontweight="bold", color=BLUE, fontsize=12)
ax5.spines[["top","right"]].set_visible(False)

ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(residuals, bins=30, color=CORAL, alpha=0.8, edgecolor="white")
ax6.axvline(0, color="black", lw=2, ls="--")
ax6.set_xlabel("Residual (BPM)", fontsize=10)
ax6.set_ylabel("Count", fontsize=10)
ax6.set_title("Residual Distribution", fontweight="bold", color=BLUE, fontsize=12)
ax6.spines[["top","right"]].set_visible(False)

fig.suptitle("Part A - Neural Network Heart Rate Prediction: Evaluation Dashboard",
             fontsize=15, fontweight="bold", color=BLUE, y=1.01)
plt.savefig(f"{OUTPUT_DIR}/figA4_regression_evaluation.png", bbox_inches="tight", facecolor="white")
plt.show()
print("Figure A4 saved: Regression evaluation dashboard")

# Sample prediction
print("\n" + "="*60)
print("SAMPLE PREDICTION")
print("="*60)

def predict_heart_rate(patient, model, scaler, feature_names):
    row = pd.DataFrame([patient])[feature_names]
    row_sc = scaler.transform(row)
    pred = model.predict(row_sc, verbose=0)[0][0]
    if pred < 60:    zone = "Below normal (Bradycardia risk)"
    elif pred < 100: zone = "Normal range"
    else:            zone = "Elevated (Tachycardia risk)"
    return round(pred, 1), zone

feat_names_hr = list(df_hr.drop("Heart_Rate_BPM", axis=1).columns)

patient_a = {
    "Age": 30, "Weight_kg": 70, "Height_cm": 175,
    "BMI": 22.9, "Activity_Level": 3, "Steps_Daily": 8000,
    "Sleep_Hours": 8, "Exercise_Min": 45,
    "Stress_Level": 4, "Caffeine_mg": 100
}
pred_bpm, zone = predict_heart_rate(patient_a, model_hr, scaler_hr, feat_names_hr)
print(f"\nPatient  : {patient_a}")
print(f"Predicted: {pred_bpm} BPM")
print(f"Zone     : {zone}")

# Save
with open(f"{OUTPUT_DIR}/scaler_hr.pkl", "wb") as f:
    pickle.dump(scaler_hr, f)
with open(f"{OUTPUT_DIR}/feature_names_hr.json", "w") as f:
    json.dump(feat_names_hr, f)
with open(f"{OUTPUT_DIR}/metrics_regression.json", "w") as f:
    json.dump({"MAE": round(mae,3), "RMSE": round(rmse,3), "R2": round(r2,4)}, f)

print("\nPart A completed successfully")
print(f"MAE={mae:.2f} BPM | RMSE={rmse:.2f} | R2={r2:.4f}")

# Cell 1
!pip install tensorflow scikit-learn pandas numpy matplotlib seaborn -q

# Cell 2 - Part B: Arrhythmia Detection (Classification)

import os, warnings, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import pickle

tf.get_logger().setLevel("ERROR")

BLUE   = "#1e3a5f"
TEAL   = "#17a2b8"
CORAL  = "#e8533f"
GREEN  = "#28a745"
PURPLE = "#7F77DD"
GRAY   = "#6c757d"

sns.set_theme(style="whitegrid", palette=[BLUE, CORAL, TEAL, GREEN])
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("PART B - ARRHYTHMIA DETECTION (CLASSIFICATION)")
print("="*60)

# Generate ECG feature dataset
print("\nGenerating synthetic ECG feature dataset...")
N2 = 3000
class_names  = ["Normal","Atrial Fibrillation","Bradycardia","Tachycardia","PVC"]
class_labels = np.random.choice([0,1,2,3,4], N2, p=[0.50,0.15,0.12,0.13,0.10])

def gen_ecg_features(label, n):
    if label == 0:
        rr=np.random.normal(0.83,0.08,n); hr=np.random.normal(72,8,n)
        pr=np.random.normal(0.16,0.02,n); qrs=np.random.normal(0.09,0.01,n)
        qt=np.random.normal(0.40,0.03,n); sdnn=np.random.normal(50,15,n)
        rmssd=np.random.normal(40,12,n); pnn50=np.random.normal(20,8,n)
        lf=np.random.normal(500,100,n); hf=np.random.normal(400,80,n)
    elif label == 1:
        rr=np.random.normal(0.65,0.18,n); hr=np.random.normal(110,25,n)
        pr=np.random.normal(0.0,0.01,n); qrs=np.random.normal(0.10,0.02,n)
        qt=np.random.normal(0.38,0.05,n); sdnn=np.random.normal(120,40,n)
        rmssd=np.random.normal(100,30,n); pnn50=np.random.normal(60,15,n)
        lf=np.random.normal(300,80,n); hf=np.random.normal(600,120,n)
    elif label == 2:
        rr=np.random.normal(1.20,0.15,n); hr=np.random.normal(45,8,n)
        pr=np.random.normal(0.22,0.03,n); qrs=np.random.normal(0.10,0.015,n)
        qt=np.random.normal(0.44,0.04,n); sdnn=np.random.normal(65,20,n)
        rmssd=np.random.normal(55,18,n); pnn50=np.random.normal(30,10,n)
        lf=np.random.normal(450,90,n); hf=np.random.normal(350,70,n)
    elif label == 3:
        rr=np.random.normal(0.50,0.08,n); hr=np.random.normal(130,20,n)
        pr=np.random.normal(0.14,0.02,n); qrs=np.random.normal(0.09,0.01,n)
        qt=np.random.normal(0.34,0.04,n); sdnn=np.random.normal(30,10,n)
        rmssd=np.random.normal(20,8,n); pnn50=np.random.normal(5,3,n)
        lf=np.random.normal(650,120,n); hf=np.random.normal(200,50,n)
    else:
        rr=np.random.normal(0.80,0.12,n); hr=np.random.normal(78,12,n)
        pr=np.random.normal(0.16,0.03,n); qrs=np.random.normal(0.14,0.02,n)
        qt=np.random.normal(0.42,0.05,n); sdnn=np.random.normal(70,20,n)
        rmssd=np.random.normal(60,18,n); pnn50=np.random.normal(25,10,n)
        lf=np.random.normal(480,100,n); hf=np.random.normal(380,80,n)
    return np.column_stack([rr,hr,pr,qrs,qt,sdnn,rmssd,pnn50,lf,hf])

data_parts, label_parts = [], []
counts = np.bincount(class_labels)
for lbl, cnt in enumerate(counts):
    data_parts.append(gen_ecg_features(lbl, cnt))
    label_parts.extend([lbl]*cnt)

X_ecg_raw = np.vstack(data_parts)
y_ecg_raw = np.array(label_parts)
idx = np.random.permutation(len(y_ecg_raw))
X_ecg_raw = X_ecg_raw[idx]
y_ecg_raw = y_ecg_raw[idx]

ecg_feat_names = ["RR_Interval","Heart_Rate","PR_Interval","QRS_Duration",
                  "QT_Interval","SDNN","RMSSD","pNN50","LF_Power","HF_Power"]

df_ecg = pd.DataFrame(X_ecg_raw, columns=ecg_feat_names)
df_ecg["Arrhythmia_Type"]  = y_ecg_raw
df_ecg["Arrhythmia_Label"] = df_ecg["Arrhythmia_Type"].map(dict(enumerate(class_names)))

print(f"ECG dataset shape: {df_ecg.shape}")
print(f"Class distribution:\n{df_ecg.Arrhythmia_Label.value_counts()}")

# Figure B1: Class distribution
colors_cls = [BLUE, CORAL, TEAL, GREEN, PURPLE]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Part B - Arrhythmia Dataset Overview",
             fontsize=14, fontweight="bold", color=BLUE)
counts_cls = df_ecg["Arrhythmia_Label"].value_counts()
axes[0].bar(counts_cls.index, counts_cls.values, color=colors_cls, alpha=0.85)
axes[0].set_title("Class Distribution", fontweight="bold", color=BLUE)
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=20)
axes[0].spines[["top","right"]].set_visible(False)
for i, v in enumerate(counts_cls.values):
    axes[0].text(i, v+10, str(v), ha="center", fontsize=9, fontweight="bold")
axes[1].pie(counts_cls.values, labels=counts_cls.index,
            colors=colors_cls, autopct="%1.1f%%", startangle=90, pctdistance=0.85)
axes[1].set_title("Class Proportions", fontweight="bold", color=BLUE)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figB1_class_distribution.png", bbox_inches="tight", facecolor="white")
plt.show()
print("Figure B1 saved: Class distribution")

# Figure B2: ECG box plots by class
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
fig.suptitle("ECG Feature Distributions by Arrhythmia Type",
             fontsize=14, fontweight="bold", color=BLUE, y=0.98)
for ax, feat in zip(axes.flatten(), ecg_feat_names):
    data_by_class = [df_ecg[df_ecg.Arrhythmia_Type==i][feat].values for i in range(5)]
    bp = ax.boxplot(data_by_class, patch_artist=True,
                    medianprops={"color":"white","linewidth":2})
    for patch, color in zip(bp["boxes"], colors_cls):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(range(1,6))
    ax.set_xticklabels(["Norm","AF","Brady","Tachy","PVC"], fontsize=7, rotation=15)
    ax.set_title(feat, fontweight="bold", fontsize=9, color=BLUE)
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figB2_ecg_boxplots.png", bbox_inches="tight", facecolor="white")
plt.show()
print("Figure B2 saved: ECG box plots")

# Preprocessing
X_ecg = df_ecg[ecg_feat_names].values
y_ecg = df_ecg["Arrhythmia_Type"].values

X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg = train_test_split(
    X_ecg, y_ecg, test_size=0.20, random_state=42, stratify=y_ecg
)

scaler_ecg = MinMaxScaler()
X_train_ecg_sc = scaler_ecg.fit_transform(X_train_ecg)
X_test_ecg_sc  = scaler_ecg.transform(X_test_ecg)

y_train_ecg_oh = keras.utils.to_categorical(y_train_ecg, 5)
y_test_ecg_oh  = keras.utils.to_categorical(y_test_ecg, 5)
print(f"\nTrain: {X_train_ecg.shape[0]} | Test: {X_test_ecg.shape[0]}")

# Build model
print("\nBuilding neural network...")
model_ecg = keras.Sequential([
    layers.Input(shape=(X_train_ecg_sc.shape[1],)),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(5, activation="softmax")
])
model_ecg.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model_ecg.summary()

cb_list_ecg = [
    callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_accuracy"),
    callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
]

print("\nTraining model...")
history_ecg = model_ecg.fit(
    X_train_ecg_sc, y_train_ecg_oh,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    callbacks=cb_list_ecg,
    verbose=1
)
print(f"Training complete - {len(history_ecg.history['loss'])} epochs")

# Evaluate
y_pred_ecg_prob = model_ecg.predict(X_test_ecg_sc, verbose=0)
y_pred_ecg      = np.argmax(y_pred_ecg_prob, axis=1)
acc_ecg = accuracy_score(y_test_ecg, y_pred_ecg)
auc_ecg = roc_auc_score(y_test_ecg_oh, y_pred_ecg_prob,
                         multi_class="ovr", average="macro")

print("\n" + "="*60)
print("PART B - RESULTS")
print("="*60)
print(f"Accuracy : {acc_ecg:.4f}")
print(f"ROC-AUC  : {auc_ecg:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_ecg, y_pred_ecg, target_names=class_names))

# Figure B3: Evaluation dashboard
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history_ecg.history["loss"], color=BLUE, lw=2, label="Train Loss")
ax1.plot(history_ecg.history["val_loss"], color=CORAL, lw=2, ls="--", label="Val Loss")
ax1.set_title("Training & Validation Loss", fontweight="bold", color=BLUE, fontsize=12)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Categorical Crossentropy")
ax1.legend(fontsize=9)
ax1.spines[["top","right"]].set_visible(False)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history_ecg.history["accuracy"], color=TEAL, lw=2, label="Train Accuracy")
ax2.plot(history_ecg.history["val_accuracy"], color=GREEN, lw=2, ls="--", label="Val Accuracy")
ax2.set_title("Training & Validation Accuracy", fontweight="bold", color=BLUE, fontsize=12)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend(fontsize=9)
ax2.spines[["top","right"]].set_visible(False)

ax3 = fig.add_subplot(gs[0, 2])
cm_ecg = confusion_matrix(y_test_ecg, y_pred_ecg)
sns.heatmap(cm_ecg, annot=True, fmt="d", cmap="Blues", ax=ax3,
            xticklabels=["Norm","AF","Brady","Tachy","PVC"],
            yticklabels=["Norm","AF","Brady","Tachy","PVC"],
            linewidths=0.5, linecolor="white",
            annot_kws={"size":10, "weight":"bold"})
ax3.set_title("Confusion Matrix", fontweight="bold", color=BLUE, fontsize=12)
ax3.set_ylabel("Actual")
ax3.set_xlabel("Predicted")

ax4 = fig.add_subplot(gs[1, 0])
per_class_acc = cm_ecg.diagonal() / cm_ecg.sum(axis=1)
ax4.barh(class_names, per_class_acc*100, color=colors_cls, alpha=0.85)
for i, v in enumerate(per_class_acc):
    ax4.text(v*100+0.5, i, f"{v*100:.1f}%", va="center",
             fontsize=9, fontweight="bold")
ax4.set_title("Per-Class Accuracy", fontweight="bold", color=BLUE, fontsize=12)
ax4.set_xlabel("Accuracy (%)")
ax4.set_xlim(0, 115)
ax4.spines[["top","right"]].set_visible(False)

ax5 = fig.add_subplot(gs[1, 1])
for i, (name, color) in enumerate(zip(class_names, colors_cls)):
    fpr, tpr, _ = roc_curve(y_test_ecg_oh[:, i], y_pred_ecg_prob[:, i])
    auc_i = roc_auc_score(y_test_ecg_oh[:, i], y_pred_ecg_prob[:, i])
    ax5.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_i:.2f})")
ax5.plot([0,1],[0,1], ls="--", color=GRAY, lw=1)
ax5.set_xlabel("False Positive Rate")
ax5.set_ylabel("True Positive Rate")
ax5.set_title("ROC Curves (One-vs-Rest)", fontweight="bold", color=BLUE, fontsize=12)
ax5.legend(fontsize=7)
ax5.spines[["top","right"]].set_visible(False)

ax6 = fig.add_subplot(gs[1, 2])
max_probs = y_pred_ecg_prob.max(axis=1)
ax6.hist(max_probs, bins=30, color=PURPLE, alpha=0.85, edgecolor="white")
ax6.axvline(0.9, ls="--", color=CORAL, lw=2, label="0.90 threshold")
ax6.set_xlabel("Prediction Confidence")
ax6.set_ylabel("Count")
ax6.set_title("Prediction Confidence Distribution",
              fontweight="bold", color=BLUE, fontsize=12)
ax6.legend(fontsize=9)
ax6.spines[["top","right"]].set_visible(False)

fig.suptitle("Part B - Neural Network Arrhythmia Detection: Evaluation Dashboard",
             fontsize=15, fontweight="bold", color=BLUE, y=1.01)
plt.savefig(f"{OUTPUT_DIR}/figB3_classification_evaluation.png",
            bbox_inches="tight", facecolor="white")
plt.show()
print("Figure B3 saved: Classification evaluation dashboard")

# Figure B4: Architecture diagram
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Neural Network Architectures", fontsize=14,
             fontweight="bold", color=BLUE)
for ax_idx, (ax, title, layers_info) in enumerate(zip(axes,
    ["Part A - Regression", "Part B - Classification"],
    [
        [("Input",10,TEAL),("Dense 128 + BN + Dropout",128,BLUE),
         ("Dense 64 + BN + Dropout",64,BLUE),("Dense 32 + Dropout",32,BLUE),
         ("Dense 16",16,BLUE),("Output (Linear)",1,GREEN)],
        [("Input",10,TEAL),("Dense 256 + BN + Dropout",256,CORAL),
         ("Dense 128 + BN + Dropout",128,CORAL),("Dense 64 + BN + Dropout",64,CORAL),
         ("Dense 32 + Dropout",32,CORAL),("Output (Softmax)",5,GREEN)]
    ])):
    ax.set_xlim(0,10)
    ax.set_ylim(0,len(layers_info)+1)
    ax.axis("off")
    ax.set_title(title, fontweight="bold", color=BLUE, fontsize=12)
    for i, (name, size, color) in enumerate(reversed(layers_info)):
        y_pos = i + 0.8
        bar_w = min(size/30, 8)
        rect = plt.Rectangle((5-bar_w/2, y_pos-0.3), bar_w, 0.55,
                               color=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(5, y_pos-0.025, f"{name}\n({size} units)",
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="white")
        if i < len(layers_info)-1:
            ax.annotate("", xy=(5, y_pos+0.25), xytext=(5, y_pos+0.55),
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figB4_architecture.png", bbox_inches="tight", facecolor="white")
plt.show()
print("Figure B4 saved: Architecture diagram")

# Sample prediction
print("\n" + "="*60)
print("SAMPLE PREDICTION")
print("="*60)

def predict_arrhythmia(ecg_features, model, scaler):
    row = np.array(ecg_features).reshape(1, -1)
    row_sc = scaler.transform(row)
    probs  = model.predict(row_sc, verbose=0)[0]
    pred   = np.argmax(probs)
    return class_names[pred], probs[pred]*100, dict(zip(class_names, (probs*100).round(2)))

sample_ecg = [0.83, 72, 0.16, 0.09, 0.40, 50, 40, 20, 500, 400]
diagnosis, confidence, all_probs = predict_arrhythmia(sample_ecg, model_ecg, scaler_ecg)
print(f"\nECG Features : {dict(zip(ecg_feat_names, sample_ecg))}")
print(f"Diagnosis    : {diagnosis}")
print(f"Confidence   : {confidence:.1f}%")
print(f"All probs    : {all_probs}")

# Save
with open(f"{OUTPUT_DIR}/scaler_ecg.pkl", "wb") as f:
    pickle.dump(scaler_ecg, f)
with open(f"{OUTPUT_DIR}/ecg_feature_names.json", "w") as f:
    json.dump(ecg_feat_names, f)
with open(f"{OUTPUT_DIR}/class_names.json", "w") as f:
    json.dump(class_names, f)
with open(f"{OUTPUT_DIR}/metrics_classification.json", "w") as f:
    json.dump({"accuracy": round(acc_ecg*100,2), "roc_auc": round(auc_ecg,4)}, f)

print("\nPart B completed successfully")
print(f"Accuracy={acc_ecg*100:.1f}% | AUC={auc_ecg:.4f}")
