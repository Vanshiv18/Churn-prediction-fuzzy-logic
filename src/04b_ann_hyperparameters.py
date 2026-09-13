"""Stage B: Hyperparameter Analysis (Section 12) using Keras."""
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

scaler = joblib.load("final_scaler.joblib")
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
n_features = X_train_s.shape[1]

def build_model(hidden_layers, lr):
    model = keras.Sequential([layers.Input(shape=(n_features,))])
    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

def run_exp(label, hidden, lr, epochs, batch):
    model = build_model(hidden, lr)
    model.fit(X_train_s, y_train, epochs=epochs, batch_size=batch, validation_split=0.15, verbose=0)
    preds = (model.predict(X_test_s, verbose=0) > 0.5).astype(int).ravel()
    res = {
        "experiment": label, "hidden_layers": str(hidden), "learning_rate": lr,
        "epochs": epochs, "batch_size": batch,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
    }
    print(label, res)
    return res

results = []
results.append(run_exp("Exp-1 (base)",       (32, 16), 0.001, 50,  32))
results.append(run_exp("Exp-2 (lr=0.01)",    (32, 16), 0.01,  50,  32))
results.append(run_exp("Exp-3 (epochs=100)", (32, 16), 0.001, 100, 32))
results.append(run_exp("Exp-4 (batch=64)",   (32, 16), 0.001, 100, 64))
results.append(run_exp("Exp-5 (lr=0.1)",     (32, 16), 0.1,   50,  32))

pd.DataFrame(results).to_csv("keras_hyper_results.csv", index=False)
best_idx = pd.DataFrame(results)["f1"].idxmax()
print(f"\nBest configuration by F1-score: {results[best_idx]}")
print("Saved keras_hyper_results.csv")
