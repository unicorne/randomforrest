# imports
import logging
import sys
import requests
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from flask import Flask, render_template, Response, url_for, json

import tools

# get current date for logger
current_date: str = str(datetime.today().strftime("%Y-%m-%d-%H:%M:%S"))
log_path: str = "logs/" + current_date + "-logs.log"

# set up logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger(__name__)

# set up app for flask
app = Flask(__name__, template_folder="")


def main():
    logger.info("application started")

    # parameters
    TRAIN_SIZE: int = 60000
    TEST_SIZE: int = 10000
    TUNE_PARAMETER: bool = True
    logger.info("Train size: " + str(TRAIN_SIZE))
    logger.info("Test size: " + str(TEST_SIZE))
    logger.info("Tune parameter: " + str(TUNE_PARAMETER))

    # get data from mnist and split to train and test
    x_train, x_test, y_train, y_test = tools.get_mnist_data_and_split_train_test(
        train_size=TRAIN_SIZE, test_size=TEST_SIZE, shuffle=True
    )

    # tune parameters
    if TUNE_PARAMETER:
        logger.info("tune parameters")
        tuned_params = random_forrest_parameter_optimization(x_train, y_train)
        logger.info("Best parameters found via tuning: ")
        logger.info(tuned_params)
    else:
        logger.info("no parameters tuned")
        tuned_params = None

    # fit model
    logger.info("create and fit model")
    rf = random_forrest_digit_classifier(
        x_train, y_train, random_forrest_params=tuned_params
    )

    # score model on train and test set
    model_train_score: str = str(rf.score(x_train, y_train))
    logger.info("Score of model on train set: " + model_train_score)
    model_test_score: str = str(rf.score(x_test, y_test))
    logger.info("Score of model on test set: " + model_test_score)

    # predict on test data
    predictions_y_test: np.ndarray = rf.predict(x_test)

    # create json data
    tools.create_test_data_json(x_test, y_test, predictions_y_test)

    # create statistics
    tools.create_statistics_json(
        model_train_score, model_test_score, y_test, predictions_y_test
    )

    # start flask application
    app.run(host="0.0.0.0", use_reloader=False)


@app.route("/")
def get_stats():
    # load json data for statistics
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "/app/static/stats.json")
    data = json.load(open(json_url))

    # get scores
    score_train: str = data["score_train"]
    score_test: str = data["score_test"]

    # prepare classification report
    classification_report_stats: str = data["classification_report"]
    classification_report_stats: str = pd.DataFrame(
        classification_report_stats
    ).T.to_html()

    # make confusion matrix image
    confunction_matrix_stats = data["confusion_matrix_stats"]
    tools.confusion_matrix_image(confunction_matrix_stats)

    # write to html statistics template
    return render_template(
        "/static/statistics_html_template.html",
        score_train=score_train,
        score_test=score_test,
        classification_report_stats=classification_report_stats,
    )


@app.route("/<int:index_test_set>")
def get_prediction(index_test_set: int):

    # load json data for test set
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "/app/static/data.json")
    data = json.load(open(json_url))

    # get data from json
    prediction = data[str(index_test_set)]["predicted_label"]
    true_label = data[str(index_test_set)]["true_label"]
    pixel_array = np.array(data[str(index_test_set)]["pixel_array"])

    # prepare image of digit that was predicted
    tools.save_image(pixel_array)

    return render_template(
        "/static/index_html_template.html",
        true_label=true_label,
        predicted_label=prediction,
    )


def random_forrest_parameter_optimization(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 3,
    verbose: int = 2,
    n_jobs: int = 4,
):

    # model parameters for Tuning
    n_estimators = np.linspace(start=10, stop=300, num=10, dtype=int)
    criterion = ["entropy", "gini"]
    max_features = ["auto"]
    max_depth = [None]
    bootstrap = [True, False]

    # create parameter dictionary
    param_grid = {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_features": max_features,
        "max_depth": max_depth,
        "bootstrap": bootstrap,
    }

    # initialize model
    rf_model = RandomForestClassifier()

    # hyperparameter tuning usig GridCV
    rf_grid = GridSearchCV(
        estimator=rf_model, param_grid=param_grid, cv=3, verbose=2, n_jobs=4
    )
    rf_grid.fit(x_train, y_train)

    return rf_grid.best_params_


def random_forrest_digit_classifier(
    x_train: np.ndarray, y_train: np.ndarray, random_forrest_params: bool = None
):
    logger.info("initialize model")
    if random_forrest_params is not None:
        rf = RandomForestClassifier(**random_forrest_params)
    else:
        rf = RandomForestClassifier(n_jobs=-1, n_estimators=10)

    # train model
    logger.info("fitting model on train data")
    rf.fit(x_train, y_train)
    return rf


if __name__ == "__main__":
    main()
