import numpy as np
from metrics import compute_r2
from metrics import compute_rmse
from metrics import compute_mse
from gradient_descent import gradient_descent_runner
from data_work import normalize_data_st
from data_work import log_data
from data_work import shuffler
from data_work import get_means_and_sds
from data_work import normalize_data_sklearn_mm
from data_work import normalize_data_sklearn_st


def run():

    cols = []
    for i in range(0, 54):
        if i != 37:
            cols.append(i)
    data = np.genfromtxt("Features_Variant_1.csv", delimiter=",", usecols=cols)  # читка данных, константный столбец можно было из самой csv-шки удалить, но я решил не трогать

    data = shuffler(data)  # перемешиваем данные
    x = data[:, 0:len(data[0]) - 1]  # забираем данные без целевой переменной
    y = data[:, len(data[0]) - 1]  # целевая переменная

    # листы с фолдами
    x_list = []
    y_list = []

    # начальные данные
    number_of_folds = 5
    learning_rate = 0.00001
    batch_size = 2500
    initial_w = np.zeros(len(x[0]))
    initial_e = 0
    amount_of_iterations = 1000

    for i in range(0, number_of_folds):
        x_list.append(x[len(x) * i // number_of_folds : len(x) * (i+1) // number_of_folds, :])
        y_list.append(y[len(y) * i // number_of_folds : len(y) * (i+1) // number_of_folds])

    for i in range(0, number_of_folds):
        costil = 0
        if i != 0:
            x_train = np.copy(x_list[0])
            y_train = np.copy(y_list[0])
        else:
            x_train = -np.copy(x_list[1])
            y_train = np.copy(y_list[1])
            costil = 1
        x_test = np.copy(x_list[i])
        y_test = np.copy(y_list[i])
        for j in range(1 + costil, number_of_folds):
            if j != i:
                x_train = np.concatenate((x_train, x_list[j]), axis=0)
                y_train = np.concatenate((y_train, y_list[j]), axis=0)
        x_means, x_sds = get_means_and_sds(x_train)
        x_train = normalize_data_st(x_train, x_means, x_sds)
        x_test = normalize_data_st(x_test, x_means, x_sds)

        [w, e] = gradient_descent_runner(x_train, y_train, initial_w, initial_e, learning_rate, amount_of_iterations, batch_size)
        log_data(w, e, compute_rmse(w, e, x_train, y_train), compute_r2(w, e, x_train, y_train),
                 compute_rmse(w, e, x_test, y_test), compute_r2(w, e, x_test, y_test), compute_mse(w, e, x_train, y_train), compute_mse(w, e, x_test, y_test),
                 i+1, learning_rate, 'output.txt', batch_size)


if __name__ == '__main__':
    run()
