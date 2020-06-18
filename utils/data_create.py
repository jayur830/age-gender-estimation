import os
import cv2
import h5py
import random
from keras.utils import to_categorical

if __name__ == '__main__':
    #  dst_path = "D:/Dataset/face_dataset/"
    dst_path = "D:/Dataset/utk_dataset/"

    data_category = ["train", "val", "test"]
    # data_sizes = [740826, 88902, 83668]
    data_sizes = [65639, 3838, 3617]

    for i in range(len(data_category)):
        src_path = "utk_data/utk_{}/".format(data_category[i])
        with h5py.File(dst_path + "{}_200x200.hdf5".format(data_category[i]), "w") as f:
            # folders = os.listdir(src_path)
            # file_list = []
            # for age in folders:
            #     if os.path.exists(src_path + age + "/0"):
            #         file_list += os.listdir(src_path + age + "/0")
            #     if os.path.exists(src_path + age + "/1"):
            #         file_list += os.listdir(src_path + age + "/1")
            file_list = os.listdir(src_path)
            random.shuffle(file_list)

            x_data = f.create_dataset("x_data", shape=(data_sizes[i], 200, 200, 3), dtype="float32")
            y_gender = f.create_dataset("y_gender", shape=(data_sizes[i], 2), dtype="float32")
            y_age = f.create_dataset("y_age", shape=(data_sizes[i], 6), dtype="float32")

            j = 0
            for filename in file_list:
                gender, age, _, _ = filename.split("_")
                # img = cv2.imread(src_path + age + "/" + str(gender) + "/" + filename).astype("float32") / 255.
                img = cv2.imread(src_path + filename).astype("float32") / 255.
                x_data[j] = img
                y_gender[j] = to_categorical(int(gender) ^ 1, num_classes=2)
                y_age[j] = to_categorical(int((int(age) - 10) / 10), num_classes=6)
                j += 1
                if j % 10 == 0:
                    print("{}/{}".format(j, data_sizes[i]))
