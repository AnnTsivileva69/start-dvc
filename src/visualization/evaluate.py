"""
Команда запуска скрипта, параметры использованны по умолчанию,
запускается из корневой директории проекта
python src/visualization/evaluate.py \
-x_train='data/processed/x_train.csv' \
-x_test='data/processed/x_test.csv' \
-y_test='data/processed/y_test.csv' \
-path_pkl='src/models/decision_tree_classifier.pkl' \
-out_json='reports/scores.json'

Команда для запуска DVC
dvc run -n evaluate \
-d src/visualization/evaluate.py \
-d data/processed \
-d src/models/decision_tree_classifier.pkl \
-M reports/scores.json \
--plots-no-cache reports/plot.json \
python src/visualization/evaluate.py \
-x_train='data/processed/x_train.csv' \
-x_test='data/processed/x_test.csv' \
-y_test='data/processed/y_test.csv' \
-path_pkl='src/models/decision_tree_classifier.pkl' \
-out_json='reports/scores.json'
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import yaml

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    exp_metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    return exp_metrics
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x_train',
                 action="store",
                 dest="x_train",
                 required=True)
    parser.add_argument('-x_test',
                 action="store",
                 dest="x_test",
                 required=True)
    parser.add_argument('-y_test',
                 action="store",
                 dest="y_test",
                 required=True)
    parser.add_argument('-path_pkl',
                 action="store",
                 dest="path_pkl")
    parser.add_argument('-out_json',
                 action="store",
                 dest="out_json",
                 required=True)
    args = parser.parse_args()
    return args

args = get_args()
x_train = pd.read_csv(args.x_train)
x_test = pd.read_csv(args.x_test)
y_test = pd.read_csv(args.y_test)
model_file = args.path_pkl
scores_file = args.out_json
plot_json = {}
with open(model_file, 'rb') as fd:
    model = pickle.load(fd)
predicted_qualities = model.predict(x_test)
exp_metrics = {f"{model.__class__.__name__}": eval_metrics(y_test, predicted_qualities)}
m1_roc=plot_roc_curve(model, x_test, y_test)
plot_json[f"{model.__class__.__name__}"] = [{
    'False_Positive_Rate': r,
    'True_Positive_Rate': f
} for r, f in zip(m1_roc.fpr, m1_roc.tpr)]
with open(scores_file, 'w') as fd:
    json.dump(exp_metrics, fd)
with open('reports/plot.json', 'w') as fd:
    json.dump(plot_json, fd)
