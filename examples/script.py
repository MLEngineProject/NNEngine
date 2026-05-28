import argparse
import time

import nn_core
import numpy as np
from sklearn.datasets import fetch_olivetti_faces, load_digits, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    nn_core.set_seed(seed)


def test_dataset(name, X, y, epochs, lr, batch_size, hidden_size, seed, use_l2=True):
    print(f"\n--- Testing {name} ---")

    seed_everything(seed)

    num_features = X.shape[1]
    num_classes = len(np.unique(y))

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float64),
        y_onehot,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = nn_core.Model()
    model.add(nn_core.DenseLayer(num_features, hidden_size))
    model.add(nn_core.ReLULayer())
    model.add(nn_core.DenseLayer(hidden_size, num_classes))
    model.add(nn_core.SoftmaxLayer())

    optimizer = nn_core.Adam(learning_rate=lr)
    loss_fn = nn_core.CategoricalCrossEntropyLoss()
    regularizer = nn_core.L2Regularizer(l2=0.0001) if use_l2 else None

    model.compile(optimizer, loss_fn, regularizer)

    t0 = time.perf_counter()
    model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        tol=1e-4,
        n_iter_no_change=10,
        verbose=False,
    )

    nn_pred_probs = model.predict(X_test_scaled)
    nn_preds = np.argmax(nn_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    t1 = time.perf_counter()

    nn_time = t1 - t0
    nn_acc = accuracy_score(y_test_labels, nn_preds) * 100
    print(f"NNEngine (Adam) — Accuracy: {nn_acc:.2f}%  |  Time: {nn_time:.4f}s")

    sk_model = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),
        activation="relu",
        solver="adam",
        batch_size=batch_size,
        learning_rate_init=lr,
        max_iter=epochs,
        tol=1e-4,
        n_iter_no_change=10,
        alpha=0.0001 if use_l2 else 0.0,
        random_state=seed,
    )

    t0 = time.perf_counter()
    y_train_flat = np.argmax(y_train, axis=1)
    sk_model.fit(X_train_scaled, y_train_flat)
    sk_preds = sk_model.predict(X_test_scaled)
    t1 = time.perf_counter()

    sk_time = t1 - t0
    sk_acc = accuracy_score(y_test_labels, sk_preds) * 100
    print(f"Sklearn  (Adam) — Accuracy: {sk_acc:.2f}%  |  Time: {sk_time:.4f}s")
    print(f"Speedup: {sk_time / nn_time:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NNEngine benchmark datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    iris = load_iris()
    test_dataset(
        "Iris Flower",
        iris.data,
        iris.target,
        epochs=100,
        lr=0.01,
        batch_size=16,
        hidden_size=16,
        seed=args.seed,
    )

    digits = load_digits()
    test_dataset(
        "Handwritten Digits",
        digits.data,
        digits.target,
        epochs=50,
        lr=0.001,
        batch_size=32,
        hidden_size=128,
        seed=args.seed + 1,
    )

    faces = fetch_olivetti_faces()
    test_dataset(
        "Olivetti Faces",
        faces.data,
        faces.target,
        epochs=50,
        lr=0.001,
        batch_size=32,
        hidden_size=128,
        seed=args.seed + 2,
    )


if __name__ == "__main__":
    main()