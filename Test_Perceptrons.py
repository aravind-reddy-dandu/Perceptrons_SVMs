from Perceptron import Perceptron
import Generate_data as gen
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm


# File for testing perceptrons by varying different parameters. Attempted bonus question as well

# Varying M and getting number of steps taken and distance from ideal perceptron in each case
def test_vary_m():
    storage = []
    # Range of m values from 10, 5000 with increments of 100
    for m in tqdm(range(10, 5000, 100)):
        steps = []
        weights = []
        biases = []
        # Looping 20 times and taking average number of steps to get a better estimate
        for i in range(0, 20):
            # Get perceptron data with e = 0.1, m as a variable and k as 5
            data = gen.gen_perceptron_data(0.1, m, 5)
            pt = Perceptron()
            steps.append(pt.fit_perceptron(data))
            # Normalizing all the weights to make them uniform
            den = np.sqrt(np.square(np.linalg.norm(pt.weights)) + np.square(pt.bias_val))
            pt.weights = pt.weights / den
            pt.bias_val = pt.bias_val / den
            # Get weights of perceptron and store them
            weights.append(pt.weights)
            # Get biases of perceptron and store them
            biases.append(pt.bias_val)
        # Calculating average weights using np.mean and axis as 0
        avg_w = np.mean(weights, axis=0)
        # Calculating bias the same way
        avg_b = np.mean(biases, axis=0)
        # Getting the distance from "ideal" perceptron
        dist = get_dist_from_ideal(weights=avg_w, bias=avg_b)
        # Returning all values calculated
        storage.append(str(m) + "," + str(np.average(steps)) + "," + str(dist))
    return storage


# Varying K and getting number of steps taken and distance from ideal perceptron in each case
def test_vary_k():
    storage = []
    # Range of k values from 1, 200 with increments of 5
    for k in tqdm([100]):
        steps = []
        weights = []
        biases = []
        # Looping 50 times and taking average number of steps to get a better estimate
        for i in range(0, 50):
            # Get perceptron data with e = 0.05, m as 100 and k as a variable
            data = gen.gen_perceptron_data(0.05, 100, k)
            pt = Perceptron()
            steps.append(pt.fit_perceptron(data))
            den = np.sqrt(np.square(np.linalg.norm(pt.weights)) + np.square(pt.bias_val))
            pt.weights = pt.weights / den
            pt.bias_val = pt.bias_val / den
            # Get weights of perceptron and store them
            weights.append(pt.weights)
            # Get biases of perceptron and store them
            biases.append(pt.bias_val)
        # Calculating average weights and biases using np.mean and axis as 0
        avg_w = np.mean(weights, axis=0)
        avg_b = np.mean(biases, axis=0)
        # Getting the distance from "ideal" perceptron
        dist = get_dist_from_ideal(weights=avg_w, bias=avg_b)
        # Returning all values calculated
        storage.append(str(k) + "," + str(np.average(steps)) + "," + str(dist))
    return storage


# Varying M and getting number of steps taken and distance from ideal perceptron in each case
def test_vary_e():
    storage = []
    # Range of e values from 0.01, 1 with increments of 0.02
    for e in tqdm(range(90, 101, 2)):
        steps = []
        weights = []
        biases = []
        e = e / 100
        # Looping 30 times and taking average number of steps to get a better estimate
        for i in range(0, 80):
            # Get perceptron data with e as a variable, m as 100 and k = 5
            data = gen.gen_perceptron_data(e, 100, k=5)
            pt = Perceptron()
            steps.append(pt.fit_perceptron(data))
            den = np.sqrt(np.square(np.linalg.norm(pt.weights)) + np.square(pt.bias_val))
            pt.weights = pt.weights / den
            pt.bias_val = pt.bias_val / den
            # Get weights and biases of perceptron and store them
            weights.append(pt.weights)
            biases.append(pt.bias_val)
        # Calculating average weights and biases using np.mean and axis as 0
        avg_w = np.mean(weights, axis=0)
        avg_b = np.mean(biases, axis=0)
        # Getting the distance from "ideal" perceptron
        dist = get_dist_from_ideal(weights=avg_w, bias=avg_b)
        # Returning all values calculated
        storage.append(str(e) + "," + str(np.average(steps)) + "," + str(dist))
    return storage


def test_m_100():
    storage = []
    for m in [100]:
        steps = []
        for i in range(0, 1000):
            data = gen.gen_perceptron_data(0.1, m, 5)
            pt = Perceptron()
            nu = pt.fit_perceptron(data)
            steps.append(nu)
        storage.append(steps)
        # storage.append(str(m) + "," + str(np.max(steps)))
    steps = storage[0]
    lesscnt = 0
    for n in steps:
        if n < 25:
            lesscnt += 1
    print('Number of times maximum steps is less than 20 is ' + str(lesscnt) + ' out of 1000')
    print('maximum steps is ' + str(max(steps)))


# Method to get distance of any perceptron my the ideal perceptron
def get_dist_from_ideal(weights, bias):
    # getting dimension from input
    k = len(weights)
    # Calculating the denominator for normalizing the input perceptron to 1
    den = np.sqrt(np.square(np.linalg.norm(weights)) + np.square(bias))
    weights = weights / den
    bias = bias / den
    # Defining ideal perceptron
    ideal_weights = np.zeros(shape=(k, 1))
    ideal_weights[k - 1] = 1
    ideal_bias_val = 0
    # Calculating distance
    distance = np.square(np.linalg.norm(weights - ideal_weights)) + np.square(np.linalg.norm(bias - ideal_bias_val))
    return distance


# test_m_100()
# Driver code
data_m_var = test_vary_e()
for v in data_m_var:
    print(v)
# data = gen.gen_perceptron_data(0.1, 100, 5)
# pt = Perceptron()
# pt.fit_perceptron(data)
# print(get_dist_from_ideal(pt))
