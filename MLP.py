import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.neural_network import MLPClassifier

def prepare_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    return X, y

def build_mlp(hidden_neurons=4, activation='logistic', solver='lbfgs', max_iter=2000):
    model = MLPClassifier(hidden_layer_sizes=(hidden_neurons,), activation=activation,  solver=solver, max_iter=max_iter, random_state=42)
    return model

def train_model(model, X, y):
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    accuracy = model.score(X, y)
    predictions = model.predict(X)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("Predictions:", predictions)
    return accuracy, predictions

def plot_decision_boundary(model, X, y, accuracy):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Plot class 0 points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', edgecolors='k', s=100, label='Class 0')

    # Plot class 1 points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', edgecolors='k', s=100, label='Class 1')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('MLP Decision Boundary for XOR')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()


#Prepare data
X, y = prepare_data()

#Build MLP
mlp_model = build_mlp(hidden_neurons=4, activation='logistic', solver='lbfgs')

#Train model
mlp_model = train_model(mlp_model, X, y)

#Evaluate model
accuracy, predictions = evaluate_model(mlp_model, X, y)

df_xor = pd.DataFrame({
    "x₁": X[:, 0],
    "x₂": X[:, 1],
    "Actual": y,
    "Predicted": predictions
})

print("\nXOR Gate Prediction Results:")
display(df_xor)

#Visualize model
plot_decision_boundary(mlp_model, X, y, accuracy)