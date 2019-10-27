import numpy as np
from math import sqrt


def compute_rmse(w, e, x_points, y_points):  # подсчет RMSE
    se = y_points - (np.dot(w, np.transpose(x_points)) + e)
    return sqrt(np.dot(se, np.transpose(se))/float(len(x_points)))


def compute_r2(w, e, x_points, y_points): # подсчет R2
    ss_res = 0.0
    ss_tot = 0.0
    y_mean = np.mean(y_points)
    ss_res += y_mean - (np.dot(w, np.transpose(x_points)) + e)
    ss_tot += (y_points - y_mean)
    return 1 - (float(np.dot(ss_res, np.transpose(ss_res)))/float(np.dot(ss_tot, np.transpose(ss_tot))))


def compute_mse(w, e, x_points, y_points):  # подсчет LOSS
    se = 0
    se = y_points - (np.dot(w, np.transpose(x_points)) + e)
    return np.dot(se, np.transpose(se)) / float(len(x_points))
