import matplotlib.pyplot as plt


def plot_eval(hist, x_label, y_label, attr_list):
    for name in attr_list:
        plt.plot(hist.history[name])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(attr_list, loc="upper left")
    plt.show()
