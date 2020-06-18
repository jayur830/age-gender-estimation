from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from model.age_predictor import age_predictor
from utils.load_data import load_data
from utils.plot import plot_eval

if __name__ == "__main__":
    batch_size = 32

    train_x_data, _, train_y_age, \
        val_x_data, _, val_y_age, \
        test_x_data, _, test_y_age = load_data()

    model = age_predictor()
    history = model.fit(
        x=train_x_data,
        y=train_y_age,
        batch_size=batch_size,
        epochs=20,
        shuffle="batch",
        validation_data=(val_x_data, val_y_age),
        callbacks=[
            ModelCheckpoint(
                filepath="checkpoint/a_{epoch:02d}-{val_loss:.2f}.h5",
                verbose=1,
                monitor="val_loss",
                save_weights_only=True,
                period=1),
            TensorBoard(
                log_dir="graph",
                histogram_freq=1,
                write_images=True)])

    plot_eval(history, "Epoch", "Loss", ["loss", "val_loss"])
    plot_eval(history, "Epoch", "Accuracy", ["accuracy", "val_accuracy"])

    loss, acc = \
            model.evaluate(
                test_x_data, test_y_age,
                batch_size=batch_size)
    print("Test loss: {}".format(loss))
    print("Test acc: {}".format(acc))
    plot_model(model, to_file="model/age_predictor.png")
    with open("age_predictor_config.json", "w") as json_write:
        json_write.write(model.to_json())
    model.save_weights("age_predictor_weights.hdf5")
