# from keras.utils import HDF5Matrix
#
#
# def load_data():
#     root_path = "D:/Dataset/utk_dataset/"
#     train_x_data = HDF5Matrix(root_path + "train.hdf5", "x_data")
#     train_y_gender = HDF5Matrix(root_path + "train.hdf5", "y_gender")
#     train_y_age = HDF5Matrix(root_path + "train.hdf5", "y_age")
#     val_x_data = HDF5Matrix(root_path + "val.hdf5", "x_data")
#     val_y_gender = HDF5Matrix(root_path + "val.hdf5", "y_gender")
#     val_y_age = HDF5Matrix(root_path + "val.hdf5", "y_age")
#     test_x_data = HDF5Matrix(root_path + "test.hdf5", "x_data")
#     test_y_gender = HDF5Matrix(root_path + "test.hdf5", "y_gender")
#     test_y_age = HDF5Matrix(root_path + "test.hdf5", "y_age")
#
#     return train_x_data, train_y_gender, train_y_age, \
#         val_x_data, val_y_gender, val_y_age, \
#         test_x_data, test_y_gender, test_y_age

from keras.utils import to_categorical
import os
import random
import cv2
import numpy as np


# def load_data():
#     # root_path = "C:/Users/OwlNight/Dataset/utkcropped/"
#     root_path = "utk_data/"
#     dataset, train_x_data, train_y_gender, train_y_age, \
#         val_x_data, val_y_gender, val_y_age, \
#         test_x_data, test_y_gender, test_y_age = [{}, {}], [], [], [], [], [], [], [], [], []
#
#     file_list = os.listdir(root_path + "utk_train")
#     for file in file_list:
#         age, gender, _, _ = file.split("_")
#         gender = int(gender) ^ 1
#         if not str(age) in dataset[gender]:
#             dataset[gender][str(age)] = []
#         dataset[gender][str(age)].append(file)
#
#     train_set, val_set, test_set = [], [], []
#     for gender in range(2):
#         for age in dataset[gender]:
#             data_size = len(dataset[gender][age])
#             _train_set, _test_set = dataset[gender][age][int(data_size * 0.1):], \
#                                   dataset[gender][age][:int(data_size * 0.1)]
#             _val_set = _test_set[:int(len(_test_set) / 2)]
#             _test_set = _test_set[int(len(_test_set) / 2):]
#             train_set += _train_set
#             val_set += _val_set
#             test_set += _test_set
#
#     random.shuffle(train_set)
#     random.shuffle(val_set)
#     random.shuffle(test_set)
#     train_size, val_size, test_size = len(train_set), len(val_set), len(test_set)
#
#     i = 0
#     for filename in train_set:
#         if i % 100 == 0:
#             print("{}/{}".format(i, train_size))
#         age, gender, _, _ = filename.split("_")
#         if int(age) < 10 or int(age) >= 70:
#             i += 1
#             continue
#         # img = cv2.resize(cv2.imread(root_path + filename), dsize=(63, 63)).astype("float32") / 255.
#         img = cv2.imread(root_path + filename).astype("float32") / 255.
#         gender = to_categorical(int(gender) ^ 1, num_classes=2)
#         age = to_categorical(int((int(age) - 10) / 10), num_classes=6)
#         train_x_data.append(img)
#         train_y_gender.append(gender)
#         train_y_age.append(age)
#         i += 1
#     print("Finished to load the train dataset.")
#
#     i = 0
#     for filename in val_set:
#         if i % 100 == 0:
#             print("{}/{}".format(i, val_size))
#         age, gender, _, _ = filename.split("_")
#         if int(age) < 10 or int(age) >= 70:
#             continue
#         # img = cv2.resize(cv2.imread(root_path + filename), dsize=(63, 63)).astype("float32") / 255.
#         img = cv2.imread(root_path + filename).astype("float32") / 255.
#         gender = to_categorical(int(gender) ^ 1, num_classes=2)
#         age = to_categorical(int((int(age) - 10) / 10), num_classes=6)
#         val_x_data.append(img)
#         val_y_gender.append(gender)
#         val_y_age.append(age)
#         i += 1
#     print("Finished to load the validation dataset.")
#
#     i = 0
#     for filename in test_set:
#         if i % 100 == 0:
#             print("{}/{}".format(i, test_size))
#         age, gender, _, _ = filename.split("_")
#         if int(age) < 10 or int(age) >= 70:
#             continue
#         # img = cv2.resize(cv2.imread(root_path + filename), dsize=(63, 63)).astype("float32") / 255.
#         img = cv2.imread(root_path + filename).astype("float32") / 255.
#         gender = to_categorical(int(gender) ^ 1, num_classes=2)
#         age = to_categorical(int((int(age) - 10) / 10), num_classes=6)
#         test_x_data.append(img)
#         test_y_gender.append(gender)
#         test_y_age.append(age)
#         i += 1
#     print("Finished to load the test dataset.")
#
#     train_x_data = np.array(train_x_data)
#     train_y_gender = np.array(train_y_gender)
#     train_y_age = np.array(train_y_age)
#     val_x_data = np.array(val_x_data)
#     val_y_gender = np.array(val_y_gender)
#     val_y_age = np.array(val_y_age)
#     test_x_data = np.array(test_x_data)
#     test_y_gender = np.array(test_y_gender)
#     test_y_age = np.array(test_y_age)
#
#     return train_x_data, train_y_gender, train_y_age, \
#         val_x_data, val_y_gender, val_y_age, \
#         test_x_data, test_y_gender, test_y_age


def load_data():
    train_data_list = os.listdir("utk_data/utk_train")
    val_data_list = os.listdir("utk_data/utk_val")
    test_data_list = os.listdir("utk_data/utk_test")

    random.shuffle(train_data_list)
    random.shuffle(val_data_list)
    random.shuffle(test_data_list)

    train_x_data, val_x_data, test_x_data = [], [], []
    train_y_gender, val_y_gender, test_y_gender = [], [], []
    train_y_age, val_y_age, test_y_age = [], [], []

    def age_to_index(_age):
        return int((int(_age) - 10) / 10)

    i = 0
    for filename in train_data_list:
        if i % 200 == 0:
            print("{}/{}".format(i, len(train_data_list)))
        i += 1
        if os.path.exists("utk_data/utk_train/" + filename):
            gender, age, _, _ = filename.split("_")
            train_x_data.append(cv2.resize(cv2.imread("utk_data/utk_train/" + filename, 0), dsize=(100, 100)).reshape(100, 100, 1).astype("float32") / 255.)
            train_y_gender.append(to_categorical(int(gender) ^ 1, num_classes=2))
            train_y_age.append(to_categorical(age_to_index(age), num_classes=6))

    i = 0
    for filename in val_data_list:
        if i % 200 == 0:
            print("{}/{}".format(i, len(val_data_list)))
        i += 1
        if os.path.exists("utk_data/utk_val/" + filename):
            gender, age, _, _ = filename.split("_")
            val_x_data.append(cv2.resize(cv2.imread("utk_data/utk_val/" + filename, 0), dsize=(100, 100)).reshape(100, 100, 1).astype("float32") / 255.)
            val_y_gender.append(to_categorical(int(gender) ^ 1, num_classes=2))
            val_y_age.append(to_categorical(age_to_index(age), num_classes=6))

    i = 0
    for filename in test_data_list:
        if i % 200 == 0:
            print("{}/{}".format(i, len(test_data_list)))
        i += 1
        if os.path.exists("utk_data/utk_test/" + filename):
            gender, age, _, _ = filename.split("_")
            test_x_data.append(cv2.resize(cv2.imread("utk_data/utk_test/" + filename, 0), dsize=(100, 100)).reshape(100, 100, 1).astype("float32") / 255.)
            test_y_gender.append(to_categorical(int(gender) ^ 1, num_classes=2))
            test_y_age.append(to_categorical(age_to_index(age), num_classes=6))

    train_x_data = np.array(train_x_data)
    train_y_gender = np.array(train_y_gender)
    train_y_age = np.array(train_y_age)
    print("Finished to convert train dataset.")
    val_x_data = np.array(val_x_data)
    val_y_gender = np.array(val_y_gender)
    val_y_age = np.array(val_y_age)
    print("Finished to convert val dataset.")
    test_x_data = np.array(test_x_data)
    test_y_gender = np.array(test_y_gender)
    test_y_age = np.array(test_y_age)
    print("Finished to convert test dataset.")

    return train_x_data, train_y_gender, train_y_age, \
        val_x_data, val_y_gender, val_y_age, \
        test_x_data, test_y_gender, test_y_age


# import h5py
#
# if __name__ == '__main__':
#     with h5py.File("")
