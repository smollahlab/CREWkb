import warnings
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTEN

from utils import _RANDOM_STATE, _CATEGORICAL_COLS


def make_pipeline(
    clf,
    smote=True,
    onehot=True,
    var_filter__threshold=0.16,
    ft_filter__k=9,
    random_state=_RANDOM_STATE
):
    """Construct a `Pipeline` for REW classification with optional
    resampling, encoding, feature filtering, and model fitting.

    The pipeline optionally applies SMOTEN oversampling, one-hot encoding
    for categorical variables, variance thresholding, univariate feature
    selection, and a final classifier.

    Parameters
    ----------
    clf : estimator object
        A scikit-learn compatible classifier implementing `fit` and `predict`.

    smote : bool, default=True
        If True, applies SMOTEN oversampling to handle class imbalance.

    onehot : bool, default=True
        If True, applies one-hot encoding to predefined categorical columns.

    var_filter__threshold : float, default=0.16
        Features with variance below this threshold are removed via
        `VarianceThreshold`.

    ft_filter__k : int, default=9
        Number of top features to select using univariate ANOVA F-test
        (`SelectKBest` with `f_classif`).

    random_state : int or None, default=_RANDOM_STATE
        Random seed used for reproducibility in SMOTEN.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A configured pipeline consisting of optional SMOTEN sampling,
        optional one-hot encoding, variance filtering, feature selection,
        and the final classifier.
    """
    steps = []

    if smote:
        steps.append(("smote", SMOTEN(random_state=random_state)))

    if onehot:
        steps.append(("onehot", ColumnTransformer(
            [("onehot", OneHotEncoder(handle_unknown="ignore"), _CATEGORICAL_COLS)],
            remainder='passthrough'
        )))

    steps += [
        ("var_filter", VarianceThreshold(threshold=var_filter__threshold)),
        ("ft_filter", SelectKBest(score_func=f_classif, k=ft_filter__k)),
        ("clf", clf)
    ]
    return Pipeline(steps)


def custom_models(param_dict):
    """Initialize REW classifiers with user-specified model hyperparameters.

    Parameters
    ----------
    param_dict : dict
        Dictionary mapping model names (str) to parameter dictionaries (dict).
        Each parameter dictionary may contain:
        - valid sklearn estimator hyperparameters
        - pipeline-specific keys:
            - "smote"
            - "var_filter__threshold"
            - "ft_filter__k"

    Returns
    -------
    dict
        Dictionary mapping model names (str) to sklearn Pipeline objects.
    """
    _PIPE_PARAMS = ["smote", "var_filter__threshold", "ft_filter__k"]
    models = {}
    for model_type, params in param_dict["models"].items():
        # determine classifier type
        if model_type == "K-Nearest Neighbors":
            clf = KNeighborsClassifier()
        elif model_type == "Decision Tree":
            clf = DecisionTreeClassifier()
        elif model_type == "Random Forest":
            clf = RandomForestClassifier()
        elif model_type == "Feed-Forward Neural Network":
            clf = MLPClassifier()
        elif model_type == "Support Vector Classifier":
            clf = SVC()
        elif model_type == "Logistic Regression":
            clf = LogisticRegression()
        elif model_type == "Deep Neural Network":
            clf = MLPClassifier()
        elif model_type == "Naive Bayes":
            clf = BernoulliNB()
        else:
            warnings.warn(f"Model type {model_type} not recognized.")
            continue
        
        # filter invalid parameters for the given classifier
        clf_params = {}
        for p, v in params.items():
            if p in _PIPE_PARAMS:
                continue
            if not p in clf.get_params():
                warnings.warn(f"{p} is not recognized as a valid parameter for {model_type}.")
                continue
            clf_params[p] = v
        
        # update classifier, create pipeline, and append to list of models
        clf.set_params(**clf_params)
        pipe_params = {"clf": clf}
        if "smote" in params:
            pipe_params["smote"] = params["smote"]
        if "var_filter__threshold" in params:
            pipe_params["var_filter__threshold"] = params["var_filter__threshold"]
        if "ft_filter__k" in params:
            pipe_params["ft_filter__k"] = params["ft_filter__k"]
        pipe = make_pipeline(**pipe_params)
        models[model_type] = pipe

    return models


def default_models():
    return {
        "K-Nearest Neighbors": make_pipeline(KNeighborsClassifier()),
        "Decision Tree": make_pipeline(DecisionTreeClassifier()),
        "Random Forest": make_pipeline(RandomForestClassifier()),
        "Feed-Forward Neural Network": make_pipeline(MLPClassifier()),
        "Support Vector Machine": make_pipeline(SVC()),
        "Logistic Regression": make_pipeline(LogisticRegression()),
        "Deep Neural Network": make_pipeline(MLPClassifier()),
        "Naive Bayes": make_pipeline(BernoulliNB())
    }
