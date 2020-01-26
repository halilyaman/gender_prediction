import numpy as np
import pandas as pd
import random

gender_data = pd.read_csv("datasets/gender.csv")

# data preprocessing
test_size = round(33/100*len(gender_data))
indices = gender_data.index.tolist()
test_indices = random.sample(population=indices, k=test_size)

# splitting train and test data
training_data = gender_data.drop(test_indices)
test_data = gender_data.loc[test_indices]

# splitting the inputs and outputs
# Gender column is encoded to 1 and 0
training_input = training_data.drop(columns="Gender").values
training_output = np.where(training_data["Gender"].str.contains("Male"), 1, 0)

test_input = test_data.drop(columns="Gender").values
test_output = np.where(test_data["Gender"].str.contains("Male"), 1, 0)

# for keeping values on the same scale
def max_min_normalization(old_data, new_min, new_max):
    if new_min == new_max:
        raise Exception()
    max_value = max(old_data)
    min_value = min(old_data)

    old_range = max_value - min_value
    new_range = new_max - new_min

    new_data = []
    for i in old_data:
        normalized_value = ((i - min_value) / old_range) * new_range + new_min
        new_data.append(normalized_value)
    return new_data

# normalizing training and test data to restrict values between 0 and 1
for i, data in enumerate(training_input.T):
    training_input.T[i] = max_min_normalization(data, new_min=0, new_max=1)

for i, data in enumerate(test_input.T):
    test_input.T[i] = max_min_normalization(data, new_min=0, new_max=1)

# ==== perceptron parameters ====
# treshold and alpha values can be adjusted to get better prediction results
np.random.seed(1)
random_weights = 2 * np.random.random((2, 1)) - 1
weights = random_weights
treshold = 50
alpha = 0.01

training_set = list(zip(training_input, training_output))
test_set = list(zip(test_input, test_output))

def get_output(summation):
    if summation > treshold:
        return 1
    else:
        return 0

def train(cycle_count):
    for i in range(cycle_count):
        for data in training_set:
            summation = np.dot(data[0], weights)
            output = get_output(summation)
            error = data[1] - output
            adjustment = error * alpha
            for i, _ in enumerate(weights):
                weights[i] += adjustment

# decide how many times the model will be trained
train(50)

# ==== TEST ====
def predict(weights, input):
    summation = np.dot(input, weights)
    return get_output(summation)

correct = 0
wrong = 0
for data in test_set:
    if predict(weights, data[0]) == data[1]:
        correct += 1
    else:
        wrong += 1

print("Accuracy:",correct/(correct+wrong)*100,"%")