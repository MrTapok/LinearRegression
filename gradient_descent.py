import numpy as np
from metrics import compute_mse


def gradient_step(w_current, e_current, x_points, y_points, learning_rate, batch_size):  # шаг градиента
    w_gradient = np.zeros(52)
    e_gradient = 0
    k = len(x_points)//batch_size

    for i in range(0, k):
        prediction_error = -(2.0/batch_size)*(y_points[i*batch_size : (i+1)*batch_size] - np.dot(w_current, np.transpose(x_points[i*batch_size : (i+1)*batch_size] + e_current)))
        w_gradient += np.dot(prediction_error,  x_points[i*batch_size: (i+1)*batch_size])  # сбор поправки коэффицентов
        e_gradient += np.sum(prediction_error)  # сбор поправки свободного члена

        w_current = np.copy(w_current - (learning_rate * w_gradient))
        e_current = np.copy(e_current - (learning_rate * e_gradient))
        w_gradient = np.zeros(52)
        e_gradient = 0

    new_w = np.copy(w_current - (learning_rate * w_gradient))  # попрака коэффицентов
    new_e = np.copy(e_current - (learning_rate * e_gradient))  # поправка свободного члена
    return [new_w, new_e]


def gradient_descent_runner(x_points, y_points, starting_w, starting_e, learning_rate, amount_of_iterations, batch_size):
    w = starting_w
    e = starting_e
    loss = 1000000
    p_loss = 100000000
    counter = 0
    while True:
        counter+=1
        w, e = gradient_step(w, e, x_points, y_points, learning_rate,batch_size)
        p_loss = loss
        loss = compute_mse(w, e, x_points, y_points)
        if (loss - p_loss > 0) or (abs(loss - p_loss) < 0.001):
            break
        print(counter, " Loss = ", loss)
    return [w, e]
