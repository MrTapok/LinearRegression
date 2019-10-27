from numpy import zeros
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from random import shuffle
from math import sqrt


def shuffler(data):
    shuffle(data)
    return data


def normalize_data_sklearn_mm(data):  # нормализация из коробки, чтобы было с чем сравнить
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def normalize_data_sklearn_st(data):  # нормализация из коробки, чтобы было с чем сравнить
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def get_means_and_sds(data):
    inside_length = len(data[0])
    x_means = zeros(inside_length)
    x_sds = zeros(inside_length)

    for i in range(0, len(data)):  # получение средних по параметру
        for j in range(0, inside_length):
            x_means[j] += data[i][j]
    for i in range(0, inside_length):
        x_means[i] = float(x_means[i]) / len(data)

    for i in range(0, len(data)):  # получение среднеквадратичных отклонений
        for j in range(0, inside_length):
            x_sds[j] += (data[i][j] - x_means[j]) ** 2
    for i in range(0, inside_length):
        x_sds[i] = sqrt(float(x_sds[i]) / len(data))

    return [x_means, x_sds]


def normalize_data_st(data, x_means, x_sds):  # нормализация
    new_data = data
    inside_length = len(data[0])
    for i in range(0, len(data)):  # нормировка значений
        for j in range(0, inside_length):
            new_data[i][j] = (data[i][j] - x_means[j])/float(x_sds[j])
    return new_data


def log_data(w, e, train_rmse, train_r2, test_rmse, test_r2, train_loss, test_loss, fold_number, learning_rate, file_name, batch_size):  # вывод в файл
    data_file = open(file_name, 'a')
    data_file.write("Learning rate = " + str(learning_rate) + "\n")
    data_file.write("Batch size = " + str(batch_size) + "\n")
    data_file.write("Model coefficients: " + str(w) + "\n")
    data_file.write("Model bias: " + str(e) + "\n")
    data_file.write("Metrics on " + str(fold_number) + " fold\n")
    data_file.write("Train RMSE = " + str(train_rmse) + "\n")
    data_file.write("Train R2 = " + str(train_r2) + "\n")
    data_file.write("Test RMSE = " + str(test_rmse) + "\n")
    data_file.write("Test R2 = " + str(test_r2) + "\n")
    data_file.write("Train loss = " + str(train_loss) + "\n")
    data_file.write("Test loss = " + str(test_loss) + "\n\n")
    data_file.close()


