import numpy as np
from data_work import shuffler
from metrics import compute_mse


def gradient_step(w_current, e_current, x_points, y_points, learning_rate):  # шаг градиента
    w_gradient = np.zeros(52)
    e_gradient = 0
    counter = 0
    n = float(len(x_points))  # примерно 32760 элементов, будем отправлять пакет каждые 1000 элементов
    indexes = []
    for i in range(0, len(x_points)):
        indexes.append(i)
    indexes = shuffler(indexes)

    for i in range(0, len(indexes)):
        w_gradient += -(2.0/n) * (y_points[indexes[i]] - (np.dot(w_gradient, x_points[indexes[i]]) + e_current)) * x_points[indexes[i]]  # сбор поправки коэффицентов
        e_gradient += -(2.0/n) * (y_points[indexes[i]] - (np.dot(w_gradient, x_points[indexes[i]]) + e_current))  # сбор поправки свободного члена
        counter += 1
        if counter % 1000 == 0:
            w_current = w_current - (learning_rate * w_gradient)
            e_current = e_current - (learning_rate * e_gradient)
            w_gradient = np.zeros(52)
            e_gradient = 0

    new_w = w_current - (learning_rate * w_gradient)  # попрака коэффицентов
    new_e = e_current - (learning_rate * e_gradient)  # поправка свободного члена
    return [new_w, new_e]


def gradient_descent_runner(x_points, y_points, starting_w, starting_e, learning_rate, amount_of_iterations):
    w = starting_w
    e = starting_e
    loss = 1000000
    p_loss = 100000000
    for i in range(amount_of_iterations):
        w, e = gradient_step(w, e, x_points, y_points, learning_rate)
        p_loss = loss
        loss = compute_mse(w, e, x_points, y_points)
        #if(loss - p_loss>0):
        #    break
        print(i + 1, " Loss = ", compute_mse(w, e, x_points, y_points))
    return [w, e]
