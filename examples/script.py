import nn_core
import numpy as np
import time
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def test_regression_performance():
    print("\n--- Testing NNEngine: Non-Linear Regression ---")
    
    # 1. Generate Data
    # 5000 samples, 20 features, making it slightly non-linear
    X, y = make_regression(n_samples=5000, n_features=20, noise=0.1, random_state=42)
    
    # NNEngine strictly expects 2D matrices (MatrixRM) for both X and y.
    # Scikit-learn generates y as (N,), so we must reshape it to (N, 1).
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float64), y.astype(np.float64), test_size=0.2, random_state=42
    )

    # Neural Networks require scaled data to prevent exploding gradients
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Architecture Config
    epochs = 200
    lr = 0.01

    # ==========================================
    # Custom NNEngine (C++ Native)
    # ==========================================
    model = nn_core.Model()
    model.add(nn_core.DenseLayer(20, 64))
    model.add(nn_core.ReLULayer())
    model.add(nn_core.DenseLayer(64, 1))
    model.compile(nn_core.MSELoss())

    print("Training NNEngine...")
    t0 = time.perf_counter()
    # Verbose=False to avoid cluttering the benchmark output
    model.fit(X_train_scaled, y_train_scaled, epochs=epochs, learning_rate=lr, verbose=False)
    t1 = time.perf_counter()

    nn_preds = model.predict(X_test_scaled)
    nn_mse = mean_squared_error(y_test_scaled, nn_preds)
    nn_time = t1 - t0

    print(f"NNEngine   | MSE: {nn_mse:.6f} | Time: {nn_time:.4f}s")

    # ==========================================
    # Scikit-Learn (MLPRegressor)
    # ==========================================
    print("Training Scikit-Learn...")
    # Configure sklearn to mirror our C++ Engine:
    # solver='sgd' with momentum=0 mirrors our basic Vanilla Gradient Descent.
    # batch_size=X_train.shape[0] forces Full-Batch GD instead of mini-batches.
    sk_model = MLPRegressor(
        hidden_layer_sizes=(64,), 
        activation='relu', 
        solver='sgd',
        batch_size=X_train.shape[0], 
        learning_rate_init=lr,
        max_iter=epochs,
        momentum=0.0,
        random_state=42
    )

    t0 = time.perf_counter()
    # sklearn expects y as (N,)
    sk_model.fit(X_train_scaled, y_train_scaled.ravel())
    t1 = time.perf_counter()

    sk_preds = sk_model.predict(X_test_scaled)
    sk_mse = mean_squared_error(y_test_scaled.ravel(), sk_preds)
    sk_time = t1 - t0

    print(f"Scikit-Learn | MSE: {sk_mse:.6f} | Time: {sk_time:.4f}s")
    
    if nn_time < sk_time:
        print(f"Speedup: {sk_time / nn_time:.2f}x faster than Sklearn!")
    else:
        print(f"Speedup: Sklearn is {nn_time / sk_time:.2f}x faster.")

if __name__ == "__main__":
    test_regression_performance()