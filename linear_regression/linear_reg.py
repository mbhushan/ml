#!/usr/bin/python


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


# Calculate coefficients
def calculate_coeff():
    dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    l_rate = 0.001
    n_epoch = 50
    coef = coefficients_sgd(dataset, l_rate, n_epoch)
    print(coef)

def predict_test():
    dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    coef = [0.4, 0.8]
    for row in dataset:
        yhat = predict(row, coef)
        print("Expected=%.3f, Predicted=%.3f" % (row[-1], yhat))


def main():
    predict_test()
    calculate_coeff()

if __name__ == '__main__':
    main()