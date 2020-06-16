from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import plot_model
from model import create_model
from load_data import load_data
from plot import plot_eval
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    batch_size = 50

    train_x_data, train_y_gender, train_y_age, \
        val_x_data, val_y_gender, val_y_age, \
        test_x_data, test_y_gender, test_y_age = load_data()

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
                save_weights_only=True,
                period=1),
            TensorBoard(
                log_dir="graph",
                histogram_freq=1,
                write_images=True),
            EarlyStopping(
                monitor="val_loss",
                verbose=1,
                min_delta=-0.007)])

    plot_eval(history, "Epoch", "Loss", ["loss", "val_loss"])
    plot_eval(history, "Epoch", "Loss", ["pred_gender_loss", "pred_age_loss", "val_pred_gender_loss", "val_pred_age_loss"])
    plot_eval(history, "Epoch", "Accuracy", ["pred_gender_accuracy", "pred_age_accuracy", "val_pred_gender_accuracy", "val_pred_age_accuracy"])

    test_loss, \
        test_gender_loss, test_age_loss, \
        test_gender_accuracy, test_age_accuracy = \
            model.evaluate(
                test_x_data, [test_y_gender, test_y_age],
                batch_size=batch_size)
    print("Test loss: {}\nTest gender loss: {}\nTest age loss: {}\nTest gender accuracy: {}\nTest age accuracy: {}".format(test_loss, test_gender_loss, test_age_loss, test_gender_accuracy, test_age_accuracy))
    plot_model(model, to_file="model.png")
    with open("model.json", "w") as json_write:
        json.dump(model.to_json(), json_write, indent="\t")
    model.save_weights("model_weights.hdf5")
