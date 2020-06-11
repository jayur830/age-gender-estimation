from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
import cv2
import json
import os

with open("_imdb.json", "r") as json_file:
    dataset = json.load(json_file)
    generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    src_path = "C:/Users/OwlNight/Dataset/imdb_crop/"
    dst_path = "D:/Dataset/face_dataset/imdb_train/"
    data_size = len(dataset) * 5

    i = 0
    for data in dataset:
        if not os.path.exists(dst_path + data["img_path"][:3]):
            os.makedirs(dst_path + data["img_path"][:3])
        face_location = data["face_location"]
        img = load_img(src_path + data["img_path"])
        x = img_to_array(img)
        x = x[int(face_location["y"]):int(face_location["y"]) + int(face_location["height"]),
              int(face_location["x"]):int(face_location["x"]) + int(face_location["width"])]
        x = cv2.resize(src=x, dst=x, dsize=(63, 63))
        x = x.reshape((1,) + x.shape)

        j = 0
        for batch in generator.flow(x,
                                    batch_size=1,
                                    save_to_dir=dst_path + data["img_path"][:3],
                                    save_prefix="{}_{}".format(data["gender"], data["age"]),
                                    save_format="jpg"):
            j += 1
            if j >= 5:
                break
        i += 5
        if i % 200 == 0:
            print("{}/{}".format(i, data_size))
