import argparse
import pandas as pd
import yaml

from pathlib import Path
from figures import *
from models import *
from utils import clean_data, _RANDOM_STATE
from validation import loocv, kfoldcv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify Chromatin Regulator Proteins as Readers, Writers, or Erasers."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="CSV datafile with features and labels."
    )
    parser.add_argument(
        "-n",
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for cross validation."
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="./",
        help="Output directory."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default="",
        help="YAML file for hyperparameters."
    )
    args = parser.parse_args()
    in_file = args.input
    n_folds = args.n_folds
    outdir = args.outdir
    params_path = args.params

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(in_file)
    X, y, class_labels = clean_data(raw)

    
    if params_path:
        with open(Path(params_path), "r") as f:
            params = yaml.safe_load(f)
        models = custom_models(params)
    else:
        models = default_models()

    metrics = ['Accuracy', 'F1 Score', 'AUC', 'MCC', 'Confusion Matrix']
    loocv_metrics = []
    kfold_metrics = []
    for name, model in models.items():
        print(f"Testing {name}...")
        loocv_metrics.append(loocv(model, X, y))
        kfold_metrics.append(kfoldcv(
            model, 
            X, 
            y, 
            k=n_folds,
            random_state=_RANDOM_STATE
        ))
    print("Tested all models.")
    loocv_df = pd.DataFrame(
        loocv_metrics, 
        index=list(models.keys()),
        columns=metrics
    )
    kfold_df = pd.DataFrame(
        kfold_metrics, 
        index=list(models.keys()),
        columns=metrics
    )

    loocv_df.to_json(out_path / "loocv_metrics.json", orient="index")
    kfold_df.to_json(out_path / "kfold_metrics.json", orient="index")

    # Generate plots
    create_cm_grid(
        loocv_df, 
        class_labels=class_labels,
        save_to=out_path / "loocv_cm.png"
    )
    create_cm_grid(
        kfold_df, 
        class_labels=class_labels,
        save_to=out_path / "kfold_cm.png"
    )

    all_metric_bar_chart(
        loocv_df, 
        ymin=0.7,
        ymax=1,
        yspacing=0.025,
        title="LOOCV",
        save_to=out_path / f"loocv_barchart.png"
    )
    all_metric_bar_chart(
        kfold_df, 
        ymin=0.7,
        ymax=1,
        yspacing=0.025,
        title="5-Fold",
        save_to=out_path / f"kfoldcv_barchart.png"
    )

    for metric in metrics[:-1]:
        metric_bar_chart_by_cv(
            loocv_df=loocv_df,
            kfold_df=kfold_df,
            metric=metric,
            ymin=0.7,
            ymax=1,
            title=metric,
            save_to=out_path / f"{metric.lower()}.png"
        )