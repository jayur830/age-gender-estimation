import os
import cv2
import h5py
import random
from keras.utils import to_categorical

dst_path = "D:/Dataset/face_dataset/"

data_category = ["train", "val", "test"]
data_sizes = [740826, 88902, 83668]

for i in range(len(data_category)):
    src_path = dst_path + "utk_{}/".format(data_category[i])
    with h5py.File(dst_path + "{}.hdf5".format(data_category[i]), "w") as f:
        folders = os.listdir(src_path)
        file_list = []
        for age in folders:
            if os.path.exists(src_path + age + "/0"):
                file_list += os.listdir(src_path + age + "/0")
            if os.path.exists(src_path + age + "/1"):
                file_list += os.listdir(src_path + age + "/1")
        random.shuffle(file_list)
        random.shuffle(file_list)

        x_data = f.create_dataset("x_data", shape=(len(file_list), 63, 63, 3), dtype="float32")
        y_gender = f.create_dataset("y_gender", shape=(len(file_list), 2), dtype="float32")
        y_age = f.create_dataset("y_age", shape=(len(file_list), 6), dtype="float32")

        j = 0
        for filename in file_list:
            gender, age, _, _ = filename.split("_")
            img = cv2.imread(src_path + age + "/" + str(gender) + "/" + filename).astype("float32") / 255.
            x_data[j] = img
            y_gender[j] = to_categorical(int(gender) ^ 1, num_classes=2)
            y_age[j] = to_categorical(int((int(age) - 10) / 10), num_classes=6)
            j += 1
            if j % 10 == 0:
                print("{}/{}".format(j, data_sizes[i]))
