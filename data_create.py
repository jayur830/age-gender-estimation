import os
import cv2
import h5py
from keras.utils import to_categorical

dst_path = "D:/Dataset/face_dataset/"
src_path = dst_path + "imdb_train/"

with h5py.File(dst_path + "train.hdf5", "w") as f:
    data_size = 1531892
    x_data = f.create_dataset("x_data", shape=(data_size, 63, 63, 3), dtype="float32")
    y_gender = f.create_dataset("y_gender", shape=(data_size, 2), dtype="float32")
    y_age = f.create_dataset("y_age", shape=(data_size, 60), dtype="float32")

    folders = os.listdir(src_path)
    i = 0
    for folder in folders:
        file_list = os.listdir(src_path + folder)
        for filename in file_list:
            gender, age, _, _ = filename.split("_")
            img = cv2.imread(src_path + folder + "/" + filename).astype("float32") / 255.
            x_data[i] = img
            y_gender[i] = to_categorical(int(gender), num_classes=2)
            y_age[i] = to_categorical(int(age) - 10, num_classes=60)
            if i % 10 == 0:
                print("{}/{}".format(i, data_size))
            i += 1
