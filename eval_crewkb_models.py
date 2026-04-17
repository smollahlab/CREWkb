import json
import argparse

import pandas as pd

from models import models
from utils import clean_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate REW classifiers using LOOCV.")
    parser.add_argument(
        "input_data_filename",
        type=str,
        help="Input csv file containing ground truth data.",
    )
    parser.add_argument(
        'n_folds',
        type=int,
        default=0,
        help="Number of folds used in cross validation. Default is size of dataset, i.e. leave-one-out cross validation."
    )

    args = parser.parse_args()
    input_data_filename = args.input_data_filename
    n_folds = args.n_folds
    
    data = pd.read_csv(input_data_filename)
    X, y, class_labels = clean_data(data)

    accuracies = {}
    f1_scores = {}
    auc_rocs = {}
    confusion_matrices = []
    for model in models:
        print(f'Evaluating {model.name} model...')
        results = None
        if not n_folds:
            results = model.loocv_eval(X, y)
        else:
            results = model.fold_cv_eval(X, y, n_folds)
        accuracies[model.name] = results[0]
        f1_scores[model.name] = results[1]
        auc_rocs[model.name] = results[2]
        confusion_matrices.append(results[3])
        model.get_confusion_matrix(confusion_matrix=results[3], 
                                   class_labels=class_labels, 
                                   save_to=f'{model.name}_cm.png')

    # write numerical metrics to a file
    with open('model_accuracies.json', 'w') as f:
        json.dump(accuracies, f)
    with open('model_f1_scores.json', 'w') as f:
        json.dump(f1_scores, f)
    with open('model_aucs.json', 'w') as f:
        json.dump(auc_rocs, f)