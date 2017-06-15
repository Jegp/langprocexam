import os
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import normalize

def map_lines(filename, f):
    with open(filename) as file:
        for line in file:
            f(line)

data_x = []
data_y = []
data_groups = []

for index, author in enumerate(os.listdir("scaledata")):
    reviews_file = "scaledata/" + author + "/subj." + author
    ratings_file = "scaledata/" + author + "/rating." + author
    map_lines(reviews_file, lambda line: data_x.append(line))
    map_lines(ratings_file, lambda line: data_y.append(float(line)))
    map_lines(ratings_file, lambda line: data_groups.append(index))

data_x = np.array(data_x)
data_y = np.array(data_y)

splits = []

splitter = LeaveOneGroupOut()
for (train_indices, test_indices) in splitter.split(data_x, data_y, data_groups):
    x_train = data_x[train_indices]
    y_train = np.array(data_y[train_indices])
    x_test = data_x[test_indices]
    y_test = np.array(data_y[test_indices])
    splits.append(((x_train, y_train), (x_test, y_test)))

np.savez_compressed('data.npz', splits = splits)
