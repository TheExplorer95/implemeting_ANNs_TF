from params import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.manifold import TSNE


def plot_classifier_training(history, epochs, save_path):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt_fn = "loss.png"
    plt.savefig(os.path.join(save_path, plt_fn), bbox_inches="tight")


def plot_confusion_matrix(test_ds, model, save_path):
    test_em = []
    test_labels = []
    classes = [
        "blues",
        "reggae",
        "metal",
        "rock",
        "pop",
        "classical",
        "country",
        "disco",
        "jazz",
        "hiphop",
    ]

    for em, label in test_ds:
        test_em.append(np.reshape(em.numpy(), (c_dim)))
        test_labels.append(np.reshape(label.numpy(), (1, len(classes))))
    test_em = np.array(test_em)
    test_labels = np.reshape(np.array(test_labels), (len(test_labels), len(classes)))
    y_pred = np.argmax(model.predict(test_em), axis=1)
    y_true = np.argmax(test_labels, axis=1)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    confusion_mtx = confusion_mtx / tf.reduce_sum(confusion_mtx, axis=None)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mtx, xticklabels=classes, yticklabels=classes, annot=True, fmt="g"
    )
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    plt_fn = "confusion.png"
    plt.savefig(os.path.join(save_path, plt_fn), bbox_inches="tight")




def plot_tsne(
    data_train, data_test, labels_train, labels_test, save_path, classes
):
    # get and fit data
    data = np.concatenate((data_train, data_test))  # (total_num_embeddings, c_dim)
    tsne_data = TSNE(n_components=2).fit_transform(data)
    xmin = np.min(tsne_data[:, 0])
    xmax = np.max(tsne_data[:, 0])
    ymin = np.min(tsne_data[:, 1])
    ymax = np.max(tsne_data[:, 1])
    eps = (ymax - ymin) / 10  # white boundary

    tsne_train = tsne_data[:data_train.shape[0]]
    tsne_test  = tsne_data[data_train.shape[0]:]
    legend_font_size = 20
    title_font_size = 25
    # create a separate tsne plot for both train and test in the same space
    for i, tsne_i in enumerate([tsne_train, tsne_test]):
        if i:
            fn = "tsne_plot_test.eps"
            title ="YouTube (test)"
            labels = labels_test
        else:
            fn = "tsne_plot_train.eps"
            title = "GTZAN (train)"
            labels = labels_train

        plt.figure(figsize=(10, 10))
        #sns.set(font_scale=2)
        tsne_plot = sns.scatterplot(
            x=tsne_i[:, 0],
            y=tsne_i[:, 1],
            hue=labels,
            palette=[
                "purple",
                "lightgreen",
                "red",
                "orange",
                "brown",
                "blue",
                "dodgerblue",
                "green",
                "darkcyan",
                "black",
            ],
            hue_order = classes,
            legend="full",
        )

        tsne_plot.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, fontsize = legend_font_size)
        #plt.legend(fontsize='x-large', title_fontsize='40')
        tsne_plot.set_title(
            f"{title} Embeddings", fontdict={"fontsize": title_font_size}
        )
        plt.savefig(os.path.join(save_path, fn), bbox_inches="tight")

    for genre in classes:
        print(f"T-SNE plot for {genre} created.")
        # logical array to select correct indices
        logic_genre_train = labels_train == genre
        logic_genre_test = labels_test == genre
        # take correct points
        tsne_data_train = tsne_data[: labels_train.shape[0]]
        tsne_data_test = tsne_data[labels_train.shape[0] :]

        selected_train = tsne_data_train[logic_genre_train]
        selected_test = tsne_data_test[logic_genre_test]
        selected_ems = np.concatenate((selected_train, selected_test))
        labels_joint = np.concatenate(
            (
                np.repeat(["GTZAN (train)"], repeats=selected_train.shape[0]),
                np.repeat(["YouTube (test)"], repeats=selected_test.shape[0]),
            )
        )

        # create figure
        plt.figure(figsize=(10, 10))
        plt.xlim([xmin - eps, xmax + eps])
        plt.ylim([ymin - eps, ymax + eps])

        tsne_plot = sns.scatterplot(
            x=selected_ems[:, 0],
            y=selected_ems[:, 1],
            hue=labels_joint,
            palette=["blue", "red"],
            hue_order = ["GTZAN (train)","YouTube (test)"],
            legend="full",
        )

        tsne_plot.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, fontsize = legend_font_size)
        tsne_plot.set_title(
            f"{genre} Embeddings", fontdict={"fontsize": title_font_size}
        )
        plt.savefig(
            os.path.join(save_path, f"tsne_plot_{genre}.eps"), bbox_inches="tight"
        )
