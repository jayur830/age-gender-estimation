from keras.callbacks import ModelCheckpoint
from keras.utils import HDF5Matrix
from model import create_model
# from mnist_model import mnist_model
import matplotlib.pyplot as plt
import h5py

batch_size = 100

root_path = "D:/Dataset/face_dataset/"
# with h5py.File(root_path + "face_train.hdf5", "r") as dataset:
#     train_x_data = dataset["x_data"]
#     train_y_gender = dataset["y_gender"]
#     train_y_age = dataset["y_age"]
#
# with h5py.File(root_path + "face_test.hdf5", "r") as dataset:
#     test_x_data = dataset["x_data"]
#     test_y_gender = dataset["y_gender"]
#     test_y_age = dataset["y_age"]

train_x_data = HDF5Matrix(root_path + "train.hdf5", "x_data")
train_y_gender = HDF5Matrix(root_path + "train.hdf5", "y_gender")
train_y_age = HDF5Matrix(root_path + "train.hdf5", "y_age")
val_x_data = HDF5Matrix(root_path + "test.hdf5", "x_data", end=18036)
val_y_gender = HDF5Matrix(root_path + "test.hdf5", "y_gender", end=18036)
val_y_age = HDF5Matrix(root_path + "test.hdf5", "y_age", end=18036)
test_x_data = HDF5Matrix(root_path + "test.hdf5", "x_data", start=18036)
test_y_gender = HDF5Matrix(root_path + "test.hdf5", "y_gender", start=18036)
test_y_age = HDF5Matrix(root_path + "test.hdf5", "y_age", start=18036)

model = create_model()
history = model.fit(
    x=train_x_data,
    y=[train_y_gender, train_y_age],
    batch_size=batch_size,
    epochs=20,
    shuffle="batch",
    validation_data=(val_x_data, [val_y_gender, val_y_age]),
    callbacks=[
        ModelCheckpoint(
            filepath="checkpoint/{epoch:02d}-{val_loss:.2f}.h5",
            verbose=1,
            monitor="val_loss",
            save_weights_only=False,
            period=1)])


def plot_eval(hist, x_label, y_label, attr_list):
    for name in attr_list:
        plt.plot(hist.history[name])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(attr_list, loc="upper left")
    plt.show()


plot_eval(history, "Epoch", "Loss", ["loss", "val_loss"])
plot_eval(history, "Epoch", "Loss", ["pred_gender_loss", "pred_age_loss", "val_pred_gender_loss", "val_pred_age_loss"])
plot_eval(history, "Epoch", "Accuracy", ["pred_gender_accuracy", "pred_age_accuracy", "val_pred_gender_accuracy", "val_pred_age_accuracy"])

model.save("age_gender_estimation.h5")
print("score: {}".format(
    model.evaluate(
        test_x_data, [test_y_gender, test_y_age],
        batch_size=batch_size)))
