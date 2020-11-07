import numpy as np


# Main class to fit a perceptron on given data
class Perceptron():
    def __init__(self):
        pass

    # Function to fit a perceptron on given data
    def fit_perceptron(self, data):
        # Splitting the dataset for features and classifiers
        X_vectors = np.asarray(data.iloc[:, :-1])
        y_values = np.asarray(data.iloc[:, -1:])
        # Getting size of sample and dimension from first split
        m, k = X_vectors.shape

        # Declaring to class variables weights and bias. These will be updated for every mis classified point
        self.weights = np.zeros(shape=(k, 1))
        self.bias_val = 0

        # Initializing number of steps count
        number_of_steps = 0
        # Vector for storing y values calculated by perceptron at that instance
        Calculated_val = [0] * m
        i = 0
        # Running an indefinite loop until all points are correctly classified. Here we know that perceptron will
        # converge because of the dataset we generate
        while True:
            # Looping over the dataset looking for misclassified points
            for i in range(m):
                # Getting the dot product of weights and current X vector. This is w1*x1+ w2*x2 + ...... +wk*xk
                if (np.dot(X_vectors[i].reshape((k, 1)).T, self.weights) + self.bias_val)[0][0] > 0:
                    Calculated_val[i] = 1
                else:
                    Calculated_val[i] = -1
                # If calculated value agrees with y value of vector, perceptron correctly classified the point
                if float(Calculated_val[i]) != float(y_values[i]):
                    # Updating weights and bias if wrongly classified
                    self.weights += np.dot(X_vectors[i].reshape((k, 1)), y_values[i].reshape((1, 1)))
                    self.bias_val += y_values[i]
                    number_of_steps += 1
            # Storing all calculated values in the dataset
            Calculated_val = np.asarray(Calculated_val).reshape((m, 1))
            i += 1
            # If all points in dataset correctly classified, break the loop and return the number of steps taken. The
            # class will have weights and bias values if needed
            if (np.array_equal(y_values, Calculated_val)) or (number_of_steps >= 10000):
                break
        return number_of_steps
