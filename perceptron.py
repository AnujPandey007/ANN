import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

def initialize_weights(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias

def step_function(z):
    return 1 if z >= 0 else 0

def predict(x, weights, bias):
    z = np.dot(x, weights) + bias
    return step_function(z)

def update_weights(weights, bias, xi, target, prediction, learning_rate):
    error = target - prediction
    weights += learning_rate * error * xi
    bias += learning_rate * error
    return weights, bias

def train_perceptron(X, y, learning_rate=0.1, epochs=10):
    n_samples, n_features = X.shape
    weights, bias = initialize_weights(n_features)

    for epoch in range(epochs):
        for xi, target in zip(X, y):
            prediction = predict(xi, weights, bias)
            weights, bias = update_weights(weights, bias, xi, target, prediction, learning_rate)

    return weights, bias


def predict_all(X, weights, bias):
    return [predict(x, weights, bias) for x in X]


# AND Gate
X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
y_and = np.array([0, 0, 0, 1])

# OR Gate
X_or = np.array([[0,0], [0,1], [1,0], [1,1]])
y_or = np.array([0, 1, 1, 1])

# Train Perceptron on AND
weights_and, bias_and = train_perceptron(X_and, y_and)
predictions_and = predict_all(X_and, weights_and, bias_and)

# Train Perceptron on OR
weights_or, bias_or = train_perceptron(X_or, y_or)
predictions_or = predict_all(X_or, weights_or, bias_or)

def plot_decision_boundary(X, y, weights, bias, title):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = np.array([step_function(np.dot(x, weights) + bias) for x in grid])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    # Separate plotting for clarity and legend
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', edgecolors='k', label='0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='red', edgecolors='k', label='1')

    plt.legend(title="Output Class", fontsize=8, title_fontsize=9)
    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True)
    plt.show()

# Create prediction tables for AND, OR, XOR
df_and = pd.DataFrame({
    "x₁": X_and[:, 0],
    "x₂": X_and[:, 1],
    "Actual": y_and,
    "Predicted": predictions_and
})

df_or = pd.DataFrame({
    "x₁": X_or[:, 0],
    "x₂": X_or[:, 1],
    "Actual": y_or,
    "Predicted": predictions_or
})

print("AND Gate Prediction Results:")
display(df_and)

print("\nOR Gate Prediction Results:")
display(df_or)

plot_decision_boundary(X_and, y_and, weights_and, bias_and, "Perceptron - AND Gate")

plot_decision_boundary(X_or, y_or, weights_or, bias_or, "Perceptron - OR Gate")

