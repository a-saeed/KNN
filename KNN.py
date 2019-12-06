from math import sqrt

import pandas as pd
from pandas import read_csv

filename = "TrainData.csv"
filename2 = "TestData.csv"

names = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'Class']
trainData = read_csv(filename, names=names)
testData = read_csv(filename2, names=names)

dataset = [[2.78, 2.55, 0],
           [1.46, 2.36, 0],
           [3.39, 4.00, 0],
           [1.38, 1.85, 0],
           [3.06, 3.00, 0],
           [7.62, 2.75, 1],
           [5.32, 2.08, 1],
           [8.67, -0.24, 1],
           [7.67, 3.50, 1]]


# calculate the Euclidean distance between two vectors(rows)
def euclidean_distance(row1, row2):
    distance = 0.0
    # last column is the output, ignored in calculation.
    for i in range(len(row1) - 1):
        distance += pow(float((row1[i]) - float(row2[i])), 2)

    return sqrt(distance)


# locate the most similar neighbors to a test row(K nearest neighbors)
def get_neighbors(train_set, test_row, k_neighbors):
    distances = list()
    # for each row in train data set , compute the EUC distance from tested row

    for train_row in train_set:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))

    # sort calculated distances in descending order, tup[1] to sort based on distance
    distances.sort(key=lambda tup: tup[1])
    # return k nearest neighbors to test row
    neighbors = list()
    for i in range(k_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# make a classification prediction with neighbors
def predict_classification(train_set, test_row, k_neighbors):
    neighbors = get_neighbors(train_set, test_row, k_neighbors)
    # extract the output class from each of the k-neighbors
    output_values = [row[-1] for row in neighbors]
    # make the prediction based on the most represented class in the neighbors
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# calculate the accuracy of our prediction
def evaluate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# KNN algorithm
def k_nearest_neighbors(train_set, test_set, k):
    predictions = list()
    test_accuracy = list()
    correct = 0

    for row in test_set:
        output = predict_classification(train_set, row, k)
        predictions.append(output)
    for i in range(len(predictions)):
        if predictions[i] == test_set[i][-1]:
            correct += 1

        print('%s ' % i + ')PREDICTED : ' + predictions[i] + ', ACTUAL : ' + test_set[i][-1])
        test_accuracy.append(test_set[i][-1])
    print("---------------------------------------------")
    print('K value : %s ' % k)
    print("number of correctly classified instances : %s " % correct + "out of  %s" % 445)
    print("accuracy")
    print(evaluate_accuracy(test_accuracy, predictions))


test_set = []
test = pd.DataFrame(testData)
for index, rows in test.iterrows():
    my_list = [rows.col1, rows.col2, rows.col3, rows.col4, rows.col5, rows.col6, rows.col7, rows.col8, rows.Class]
    test_set.append(my_list)

train_set = []
train = pd.DataFrame(trainData)
for index, rows in train.iterrows():
    my_list = [rows.col1, rows.col2, rows.col3, rows.col4, rows.col5, rows.col6, rows.col7, rows.col8, rows.Class]
    train_set.append(my_list)

k_nearest_neighbors(train_set, test_set, 7)
