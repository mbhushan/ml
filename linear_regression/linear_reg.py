#!/usr/bin/python
from random import seed
from random import randrange
from csv import reader
from math import sqrt


# Load a CSV file
def load_csv(filename):
    dataset = list()
    firstrow = True
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row or firstrow:
                firstrow = False
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    row_len = len(row)
    for i in range(row_len-1):
        yhat += row[i] * coefficients[i+1]
    return yhat


# Estimate linear regression coefficients using stochastic gradient descent.
def coefficients_sgd(train, learning_rate, num_epoch):
    coeff = [0.0 for i in range(len(train[0]))]

    for epoch in range(num_epoch):
        sum_error = 0.0
        for row in train:
            yhat = predict(row, coeff)
            error = yhat - row[-1]
            sum_error += error**2
            coeff[0] = coeff[0] - learning_rate * error
            for i in range(len(row)-1):
                coeff[i+1] = coeff[i+1] - learning_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
    return coeff


# Estimate linear regression coefficients using stochastic gradient descent - wmape version.
def coefficients_sgd_wmape(train, learning_rate, num_epoch):
    coeff = [0.0 for i in range(len(train[0]))]

    for epoch in range(num_epoch):
        sum_error = 0.0
        for row in train:
            yhat = predict(row, coeff)
            # if (row[-1] + yhat) == 0:
            #     error = 1
            # else:
            error = ((abs(row[-1] - yhat)*2) / abs(row[-1] + yhat)) / len(train)
            # if error > 100 or error < 0:
            #     error = 1
            sum_error += error
            coeff[0] = coeff[0] - learning_rate * error
            for i in range(len(row)-1):
                coeff[i+1] = coeff[i+1] - learning_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
    return coeff


# Calculate coefficients
def calculate_coeff():
    dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    l_rate = 0.001
    n_epoch = 50
    coef = coefficients_sgd(dataset, l_rate, n_epoch)
    print(coef)


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Calculate root mean squared error
def wmape_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        if actual[i] != 0:
            sum_error += (abs(actual[i] - predicted[i])*2) / abs(actual[i] + predicted[i])
    mean_error = sum_error / float(len(actual))
    return mean_error


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)
    return scores


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm_wmape(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = wmape_metric(actual, predicted)
        scores.append(rmse)
    return scores

# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return(predictions)


# Linear Regression Algorithm With Stochastic Gradient Descent - WMAPE
def linear_regression_sgd_wmape(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd_wmape(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return(predictions)


def predict_test():
    dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    coef = [0.4, 0.8]
    for row in dataset:
        yhat = predict(row, coef)
        print("Expected=%.3f, Predicted=%.3f" % (row[-1], yhat))


# Linear Regression on wine quality dataset
def linear_reg(filename):
    seed(1)
    # load and prepare data
    dataset = load_csv(filename)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    # normalize
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    # evaluate algorithm
    n_folds = 5
    l_rate = 0.0255
    n_epoch = 50
    scores = evaluate_algorithm(dataset, linear_regression_sgd, n_folds, l_rate, n_epoch)
    return scores


# Linear Regression on wine quality dataset - WMAPE
def linear_reg_wmape(filename):
    seed(1)
    # load and prepare data
    dataset = load_csv(filename)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    # normalize
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    # evaluate algorithm
    n_folds = 5
    l_rate = 0.0915
    n_epoch = 50
    scores = evaluate_algorithm_wmape(dataset, linear_regression_sgd_wmape, n_folds, l_rate, n_epoch)
    return scores


def linear_rmse(file_names):
    scores_list = list()
    for fname in file_names:
        score = linear_reg(fname)
        scores_list.append(score)
    for scores in scores_list:
        print('Scores: %s' % scores)
        print('Mean RMSE: %.3f' % (sum(scores) / float(len(scores))))


def linear_wmape(file_names):
    scores_list = list()
    for fname in file_names:
        score = linear_reg_wmape(fname)
        scores_list.append(score)
    for scores in scores_list:
        print('Scores: %s' % scores)
        print('WMAPE: %.3f' % (sum(scores) / float(len(scores))))


def main():
    file_names = ['data/winequality-white.csv', 'data/winequality-red.csv']
    linear_wmape(file_names)


if __name__ == '__main__':
    main()