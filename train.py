from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from model import create_model
import matplotlib.pyplot as plt
from load_data import load_data

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
                save_weights_only=False,
                period=1),
            TensorBoard(
                log_dir="graph",
                histogram_freq=1,
                write_images=True)])


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
    test_loss, \
        test_gender_loss, test_age_loss, \
        test_gender_accuracy, test_age_accuracy = \
            model.evaluate(
                test_x_data, [test_y_gender, test_y_age],
                batch_size=batch_size)
    print("Test loss: {}\nTest gender loss: {}\nTest age loss: {}\nTest gender accuracy: {}\nTest age accuracy: {}".format(test_loss, test_gender_loss, test_age_loss, test_gender_accuracy, test_age_accuracy))
