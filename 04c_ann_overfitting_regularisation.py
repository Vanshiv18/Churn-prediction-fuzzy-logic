"""Stage C: Overfitting Analysis (Section 13) + Regularisation comparison, using Keras."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import joblib

tf.random.set_seed(42); np.random.seed(42)

X_train = pd.read_csv("X_train.csv"); X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

scaler = joblib.load("final_scaler.joblib")
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
n_features = X_train_s.shape[1]

def build_model(hidden_layers, lr, l2_alpha=0.0):
    model = keras.Sequential([layers.Input(shape=(n_features,))])
    for units in hidden_layers:
        reg = regularizers.l2(l2_alpha) if l2_alpha > 0 else None
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=reg))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- Overfitting-demo model: no regularisation, 150 epochs, track full history ---
overfit_model = build_model((64, 32), 0.001, l2_alpha=0.0)
hist = overfit_model.fit(X_train_s, y_train, epochs=150, batch_size=32,
                          validation_split=0.15, verbose=0)

train_losses = hist.history["loss"]
val_losses = hist.history["val_loss"]
train_accs = hist.history["accuracy"]
val_accs = hist.history["val_accuracy"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(train_losses, color="tab:blue"); axes[0, 0].set_title("Training Loss vs Epoch"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
axes[0, 1].plot(val_losses, color="tab:orange"); axes[0, 1].set_title("Validation Loss vs Epoch"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Loss")
axes[1, 0].plot(train_accs, color="tab:green"); axes[1, 0].set_title("Training Accuracy vs Epoch"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Accuracy")
axes[1, 1].plot(val_accs, color="tab:red"); axes[1, 1].set_title("Validation Accuracy vs Epoch"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig("plot8b_four_panel_overfitting.png", dpi=140)
plt.close()

min_val_epoch = int(np.argmin(val_losses))
print(f"Validation loss lowest at epoch {min_val_epoch+1}; rises afterward -> overfitting onset.")
print(f"Final training accuracy: {train_accs[-1]:.4f} | Final validation accuracy: {val_accs[-1]:.4f} | Gap: {train_accs[-1]-val_accs[-1]:.4f}")

test_preds = (overfit_model.predict(X_test_s, verbose=0) > 0.5).astype(int).ravel()
train_preds = (overfit_model.predict(X_train_s, verbose=0) > 0.5).astype(int).ravel()
final_train_acc = accuracy_score(y_train, train_preds)
final_test_acc = accuracy_score(y_test, test_preds)
print(f"Train accuracy: {final_train_acc:.4f} | Test accuracy: {final_test_acc:.4f}")

overfit_summary = pd.DataFrame({
    "epoch": range(1, len(train_losses)+1), "train_loss": train_losses, "val_loss": val_losses,
    "train_acc": train_accs, "val_acc": val_accs
})
overfit_summary.to_csv("overfit_epoch_log.csv", index=False)

# --- Regularisation: before vs after ---
def eval_model(model, epochs, batch, l2_alpha, early_stop):
    callbacks = []
    if early_stop:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True))
    h = model.fit(X_train_s, y_train, epochs=epochs, batch_size=batch,
                  validation_split=0.15, callbacks=callbacks, verbose=0)
    preds = (model.predict(X_test_s, verbose=0) > 0.5).astype(int).ravel()
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "epochs_actually_run": len(h.history["loss"]),
    }

before_model = build_model((64, 32), 0.001, l2_alpha=0.0)
before_reg = eval_model(before_model, 150, 32, 0.0, early_stop=False)
print("Before regularisation:", before_reg)

after_model = build_model((64, 32), 0.001, l2_alpha=0.01)
after_reg = eval_model(after_model, 150, 32, 0.01, early_stop=True)
print("After regularisation:", after_reg)

reg_compare = pd.DataFrame([before_reg, after_reg], index=["Before Regularisation", "After Regularisation"])
reg_compare.to_csv("regularisation_comparison.csv")
print("\nRegularisation comparison:\n", reg_compare[["accuracy", "precision", "recall", "f1"]])

# Save the final ANN (used going forward for the Fuzzy Logic stage)
overfit_model.save("final_ann.keras")
print("\nSaved final_ann.keras")
