import numpy as np
import matplotlib.pyplot as plt
from utils.mse import MSE
from sklearn.metrics import mean_squared_error

np.random.seed(42)


class LinearRegression:

    def __init__(self):
        """
        Linear regression model.

        w: parameters of the model
        n: number of features
        """
        self.w = None
        self.n = None
        self.bias = None

    def train(
        self,
        X_train: np.array,
        y_train: np.array,
        batch_size: int,
        epochs: int,
        lr: float,
    ):

        self.n = X_train.shape[1]
        self.w = np.random.rand(self.n)
        self.bias = np.random.rand(1)
        n_samples = X_train.shape[0]

        for _ in range(epochs):

            # Select random batch
            batch_indexes = np.random.choice(n_samples, batch_size, replace=False)

            # Select from input array
            X_train_batch = X_train[batch_indexes]
            y_train_batch = y_train[batch_indexes]

            # Compute predictions
            y_pred = np.dot(X_train_batch, self.w.T) + self.bias

            # Compute MSE error
            mse_error = MSE(y_train_batch, y_pred)
            # print(f"{mse_error=}")

            # Compute gradients
            error = y_pred - y_train_batch
            gradients_w = (2 / batch_size) * np.dot(X_train_batch.T, error)
            gradients_b = (2 / batch_size) * np.sum(error)

            # Update weights and bias
            self.w -= lr * gradients_w.T  # Ensure self.w has shape (1, n_features)
            self.bias -= lr * gradients_b

    def predict(self, x):
        return np.dot(x, self.w.T) + self.bias

def plot_regression_line(X: np.array, y: np.array, model, title: str = "Linear Regression Fit"):
    """
    Plot the regression line along with the actual data points.

    X: input features
    y: target values
    model: trained model that can make predictions
    title: title of the plot
    """
    plt.figure(figsize=(10, 6))

    # Plotting the actual data points
    plt.scatter(X[:, 0], y, color="blue", label="Data points")

    # Sort the X values for plotting the line correctly
    sorted_indices = np.argsort(X[:, 0])
    sorted_X = X[sorted_indices]
    sorted_y = y[sorted_indices]

    # Generate predictions from the model for the sorted X values
    predictions = model.predict(sorted_X)

    # Plotting the regression line
    plt.plot(sorted_X[:, 0], predictions, color="red", label="Regression line")

    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title(title)
    plt.legend()
    plt.show()

        
if __name__ == "__main__":
    linear_regression = LinearRegression()
    X_train = np.random.rand(100, 2)
    y_train = np.random.rand(100, 1)
    batch_size = 25
    epochs = 100
    lr = 1e-3
    linear_regression.train(X_train, y_train, batch_size, epochs, lr)
    linear_regression.plot_regression_line(X_train, y_train)
