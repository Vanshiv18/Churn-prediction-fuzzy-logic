"""Stage A: ANN Architecture Experiment (Section 11) using Keras."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

tf.random.set_seed(42); np.random.seed(42)

X_train = pd.read_csv("X_train.csv"); X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, "final_scaler.joblib")
n_features = X_train_s.shape[1]

def build_model(hidden_layers, lr):
    model = keras.Sequential([layers.Input(shape=(n_features,))])
    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

results = []
for label, hidden in [("ANN-1 (16)", (16,)), ("ANN-2 (32,16)", (32,16)), ("ANN-3 (64,32)", (64,32))]:
    model = build_model(hidden, 0.001)
    model.fit(X_train_s, y_train, epochs=50, batch_size=32, validation_split=0.15, verbose=0)
    test_preds = (model.predict(X_test_s, verbose=0) > 0.5).astype(int).ravel()
    train_preds = (model.predict(X_train_s, verbose=0) > 0.5).astype(int).ravel()
    tr_acc = accuracy_score(y_train, train_preds)
    te_acc = accuracy_score(y_test, test_preds)
    res = {
        "architecture": label, "hidden_layers": str(hidden), "learning_rate": 0.001,
        "epochs": 50, "batch_size": 32,
        "accuracy": round(te_acc, 4),
        "precision": round(precision_score(y_test, test_preds), 4),
        "recall": round(recall_score(y_test, test_preds), 4),
        "f1": round(f1_score(y_test, test_preds), 4),
        "train_accuracy": round(tr_acc, 4),
        "gap": round(tr_acc - te_acc, 4),
    }
    print(label, res)
    results.append(res)

pd.DataFrame(results).to_csv("keras_arch_results.csv", index=False)
print("\nSaved keras_arch_results.csv")
