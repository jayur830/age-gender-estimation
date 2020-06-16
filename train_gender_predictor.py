from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import plot_model
from gender_predictor import gender_predictor
from load_data import load_data
from plot import plot_eval

if __name__ == "__main__":
    batch_size = 32

    train_x_data, train_y_gender, _, \
        val_x_data, val_y_gender, _, \
        test_x_data, test_y_gender, _ = load_data()

    model = gender_predictor()
    history = model.fit(
        x=train_x_data,
        y=train_y_gender,
        batch_size=batch_size,
        epochs=20,
        shuffle="batch",
        validation_data=(val_x_data, val_y_gender),
        callbacks=[
            ModelCheckpoint(
                filepath="checkpoint/g_{epoch:02d}-{val_loss:.2f}.h5",
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

    test_loss, test_accuracy = \
            model.evaluate(
                test_x_data, test_y_gender,
                batch_size=batch_size)
    print("Test loss: {}\nTest accuracy: {}".format(test_loss, test_accuracy))
    plot_model(model, to_file="gender_predictor.png")
    with open("gender_predictor_config.json", "w") as json_write:
        json_write.write(model.to_json())
    model.save_weights("gender_predictor_weights.hdf5")
