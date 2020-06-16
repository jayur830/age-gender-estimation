from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
import cv2
import json
import os

data_category = ["train", "val", "test"]

for category in data_category:
    with open("utk_{}.json".format(category), "r") as json_file:
        dataset = json.load(json_file)
        generator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True)
        src_path = "C:/Users/OwlNight/Dataset/utkcropped/"
        # dst_path = "D:/Dataset/utk_dataset/utk_{}/".format(category)
        dst_path = "utk_data/utk_{}".format(category)
        # factor = 100
        factor = 4
        data_size = len(dataset) * factor

        i = 0
        for data in dataset:
            # if not os.path.exists(dst_path + str(data["age"])):
            #     os.makedirs(dst_path + str(data["age"]))
            # if not os.path.exists(dst_path + str(data["age"]) + "/" + str(data["gender"])):
            #     os.makedirs(dst_path + str(data["age"]) + "/" + str(data["gender"]))
            if os.path.exists(src_path + data["img_path"]):
                img = load_img(src_path + data["img_path"])
                x = img_to_array(img)
                x = cv2.resize(src=x, dst=x, dsize=(63, 63))
                x = x.reshape((1,) + x.shape)

                j = 0
                for batch in generator.flow(x,
                                            batch_size=1,
                                            save_to_dir=dst_path,
                                            save_prefix="{}_{}".format(data["gender"], data["age"]),
                                            save_format="jpg"):
                    j += 1
                    if j >= factor:
                        break
                i += factor
                if i % 200 == 0:
                    print("{}/{}".format(i, data_size))
