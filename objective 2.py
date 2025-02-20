import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.activation(np.dot(self.weights, X[i]))
                if y[i] == 1 and y_pred == 0:
                    self.weights += self.learning_rate * X[i]
                elif y[i] == 0 and y_pred == 1:
                    self.weights -= self.learning_rate * X[i]

    def evaluate(self, X, y):
        y_pred = [self.predict(x) for x in X]
        count = 0
        for i in range(len(y)) :
            if y_pred[i]==y[i] : count+=1
        accuracy = count / len(y)
        return accuracy, y_pred

    def compute_confusion_matrix(self, X, y):
        y_pred = [self.predict(x) for x in X]
        cm = confusion_matrix(y, y_pred)
        return cm

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0 (False)", "1 (True)"], yticklabels=["0 (False)", "1 (True)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"Confusion Matrix - {title}")
    plt.show()

# NAND and XOR truth tables

fun1_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
fun1_y = np.array([0, 0, 0, 1])  # fun1 output

fun2_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
fun2_y = np.array([0, 0, 1, 0])  # fun2 output

# Train and evaluate Perceptron for fun1
fun1_perceptron = Perceptron(input_size=2)
fun1_perceptron.train(fun1_X, fun1_y)
fun1_accuracy, fun1_predictions = fun1_perceptron.evaluate(fun1_X, fun1_y)

fun2_perceptron = Perceptron(input_size=2)
fun2_perceptron.train(fun2_X, fun2_y)
fun2_accuracy, fun2_predictions = fun2_perceptron.evaluate(fun2_X, fun2_y)

# Print the learned weights and predictions
# print(f"\nfun1 Perceptron Weights: {fun1_perceptron.weights}")
# print(f"fun1 Perceptron Predictions: {fun1_predictions}")
# print(f"fun1 Perceptron Accuracy: {fun1_accuracy * 100:.2f}%")

#Print the learned weights and predictions
# print(f"\nfun2 Perceptron Weights: {fun2_perceptron.weights}")
# print(f"fun2 Perceptron Predictions: {fun2_predictions}")
# print(f"fun2 Perceptron Accuracy: {fun2_accuracy * 100:.2f}%")

fun3_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
fun3_y = np.array([0, 1, 0, 0])  # fun2 output

fun3_perceptron = Perceptron(input_size=2)
fun3_perceptron.train(fun3_X, fun3_y)
fun3_accuracy, fun3_predictions = fun3_perceptron.evaluate(fun3_X, fun3_y)

#Print the learned weights and predictions
# print(f"\nfun3 Perceptron Weights: {fun3_perceptron.weights}")
# print(f"fun3 Perceptron Predictions: {fun3_predictions}")
# print(f"fun3 Perceptron Accuracy: {fun3_accuracy * 100:.2f}%")

fun4_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
fun4_y = np.array([1, 0, 0, 0])  # fun4 output

fun4_perceptron = Perceptron(input_size=2)
fun4_perceptron.train(fun4_X, fun4_y)
fun4_accuracy, fun4_predictions = fun4_perceptron.evaluate(fun4_X, fun4_y)

#Print the learned weights and predictions
# print(f"\nfun4 Perceptron Weights: {fun4_perceptron.weights}")
# print(f"fun4 Perceptron Predictions: {fun4_predictions}")
# print(f"fun4 Perceptron Accuracy: {fun4_accuracy * 100:.2f}%")

final_X = np.array([fun1_predictions, fun2_predictions, fun3_predictions, fun4_predictions])
final_y = np.array ([0,1,1,0]) # final output
final_perceptron = Perceptron(input_size=4)
final_perceptron.train(final_X, final_y)
final_accuracy, final_predictions = final_perceptron.evaluate(final_X, final_y)

#Print the learned weights and predictions
print(f"\nfinal Perceptron Weights: {final_perceptron.weights}")
print(f"final Perceptron Predictions: {final_predictions}")
print(f"final Perceptron Accuracy: {final_accuracy * 100:.2f}%")



