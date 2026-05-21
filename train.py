"""
train.py — Train on MediaPipe landmark vectors extracted from your image dataset.
Run AFTER extract_landmarks.py
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

DATASET_PATH = "dataset/asl_alphabet_train"

# ── Load ──────────────────────────────────────────────────────────
X, y_raw = [], []

print("📂 Loading landmark data...\n")
labels_found = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

for label in labels_found:
    npy = os.path.join(DATASET_PATH, label, "landmarks.npy")
    if not os.path.isfile(npy):
        print(f"   ⚠️  '{label}': no landmarks.npy — run extract_landmarks.py first")
        continue
    data = np.load(npy)
    print(f"   ✅ {label:20s}: {len(data)} samples  (feature size: {data.shape[1]})")
    X.extend(data)
    y_raw.extend([label] * len(data))

if not X:
    raise SystemExit("❌ No data found. Run extract_landmarks.py first.")

X = np.array(X, dtype=np.float32)
print(f"\n📊 Total: {len(X)} samples  |  Features: {X.shape[1]}  |  Classes: {len(set(y_raw))}\n")

# ── Encode ────────────────────────────────────────────────────────
le    = LabelEncoder()
y_enc = le.fit_transform(y_raw)
y_cat = to_categorical(y_enc)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.15, random_state=42, stratify=y_enc
)

# ── Model ─────────────────────────────────────────────────────────
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(), Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(), Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.2),

    Dense(64, activation='relu'),
    Dropout(0.1),

    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=8, factor=0.4, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
)

best_val = max(history.history['val_accuracy'])
print(f"\n🏆 Best val accuracy: {best_val:.2%}")

# ── Save ──────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
model.save("model/sign_model.keras")
np.save("model/labels.npy", le.classes_)
print("💾 model/sign_model.keras")
print("💾 model/labels.npy")
print("\n✅ Training complete! Run:  streamlit run app.py")
