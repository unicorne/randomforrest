import logging
import json
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# set up logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_mnist_data_and_split_train_test(
    train_size: int, test_size: int, shuffle: bool = True
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # get data from Mnist
    logger.info("getting data from mnist_784")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    # split data to train and test
    logger.info("split data to train and test set")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, shuffle=shuffle
    )
    return x_train, x_test, y_train, y_test


def create_test_data_json(
    x_test: np.ndarray, y_test: np.ndarray, predictions_y_test: np.ndarray
) -> None:
    """store data of test data and model predictions in dictionary and save as json"""

    # create dictionary
    predictions_dictionary = {
        i: {
            "pixel_array": list(x_test[i]),
            "true_label": str(y_test[i]),
            "predicted_label": str(predictions_y_test[i]),
        }
        for i in range(len(predictions_y_test))
    }

    # store dict to json
    with open("/app/static/data.json", "w") as fp:
        json.dump(predictions_dictionary, fp)


def create_statistics_json(
    model_score_train: float,
    model_score_test: float,
    y_test: np.ndarray,
    predictions_y_test: np.ndarray,
) -> None:
    """saves and creates advanced statistics and saves them in json format"""

    # prepare classification report
    classification_report_stats = classification_report(
        y_test, predictions_y_test, output_dict=True
    )
    logger.info("Classification Report: ")
    logger.info(classification_report_stats)

    # prepare confusion matrix
    confunction_matrix_stats = confusion_matrix(y_test, predictions_y_test).tolist()
    logger.info("Confusion Report: ")
    logger.info(confunction_matrix_stats)

    # create dictionary
    statistics_dictionary = {
        "score_train": model_score_train,
        "score_test": model_score_test,
        "classification_report": classification_report_stats,
        "confusion_matrix_stats": confunction_matrix_stats,
    }

    # store data as json
    with open("/app/static/stats.json", "w") as fp:
        json.dump(statistics_dictionary, fp)


def save_image(pixel_array: np.ndarray) -> None:
    """saves greyscale image of digit"""
    plt.imsave(
        "/app/static/tmp.jpeg", pixel_array.reshape(28, 28), cmap=plt.get_cmap("gray"),
    )


def confusion_matrix_image(confusion_matrix: np.ndarray) -> None:
    """creates and saves image of confusion matrix"""

    # get max value for color distribution
    max_val = np.percentile(confusion_matrix, 100)

    # create matshow and set x and y ticks
    plt.matshow(np.array(confusion_matrix))
    plt.xticks(np.arange(0, 10, 1))
    plt.yticks(np.arange(0, 10, 1))

    # add text to each cell
    for (x, y), value in np.ndenumerate(np.array(confusion_matrix)):
        # configure color based on max value
        if value < max_val * 0.5:
            color = "yellow"
        else:
            color = "black"
        plt.text(y, x, f"{int(value)}", va="center", ha="center", color=color)

    # save figure
    plt.savefig("/app/static/confusion_matrix.jpeg")
