from numpy import zeros
from numpy import mean
from numpy import std
from math import sqrt


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


def log_local_data(w, e, train_loss, test_loss, fold_number, file_name):  # вывод в файл
    data_file = open(file_name, 'a')
    data_file.write("Model coefficients: " + str(w) + "\n")
    data_file.write("Model bias: " + str(e) + "\n")
    data_file.write("Loss on " + str(fold_number) + " fold\n")
    data_file.write("Train loss = " + str(train_loss) + "\n")
    data_file.write("Test loss = " + str(test_loss) + "\n\n")
    data_file.close()


def log_data(train_rmse, test_rmse, train_r2, test_r2, learning_rate, batch_size, file_name):
    data_file = open(file_name, 'a')
    data_file.write("Learning rate = " + str(learning_rate) + "\n")
    data_file.write("Batch size = " + str(batch_size) + "\n")

    data_file.write("Train RMSE on folds 1-5 \n")
    for i in range(0, len(train_rmse)):
        data_file.write(str(train_rmse[i]) + " | ")
    data_file.write("\n")

    data_file.write("Test RMSE on folds 1-5 \n")
    for i in range(0, len(test_rmse)):
        data_file.write(str(test_rmse[i]) + " | ")
    data_file.write("\n")

    data_file.write("Train R2 on folds 1-5 \n")
    for i in range(0, len(train_r2)):
        data_file.write(str(train_r2[i]) + " | ")
    data_file.write("\n")

    data_file.write("Test R2 on folds 1-5 \n")
    for i in range(0, len(test_r2)):
        data_file.write(str(test_r2[i]) + " | ")
    data_file.write("\n")

    data_file.write("Train RMSE Mean = " + str(mean(train_rmse)) + "\n")
    data_file.write("Test RMSE Mean = " + str(mean(test_rmse)) + "\n")
    data_file.write("Train R2 Mean = " + str(mean(train_r2)) + "\n")
    data_file.write("Test R2 Mean = " + str(mean(test_r2)) + "\n")

    data_file.write("Train RMSE STD = " + str(std(train_rmse)) + "\n")
    data_file.write("Test RMSE STD = " + str(std(test_rmse)) + "\n")
    data_file.write("Train R2 STD = " + str(std(train_r2)) + "\n")
    data_file.write("Test R2 STD = " + str(std(test_r2)) + "\n")

    data_file.write("\n")
    data_file.write("-------------------------------------")
    data_file.write("\n")

    data_file.close()


