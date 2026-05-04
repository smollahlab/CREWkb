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


def make_rew_classifier(
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
        # delete
        from sklearn.naive_bayes import GaussianNB
        if isinstance(clf, GaussianNB):
            steps.append(("onehot", ColumnTransformer(
            [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATEGORICAL_COLS)],
            remainder='passthrough'
        )))
            
        else:
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


# DEFINE MODELS

####################### ORIGINAL HYPERPARAMS ###################################
# knn_classifier = make_rew_classifier(
#     clf=KNeighborsClassifier(
#         n_neighbors=9
#     ),
#     smote=True,    
#     var_filter__threshold=0.16,
#     ft_filter__k=9
# )

# dt_classifier = make_rew_classifier(
#     clf=DecisionTreeClassifier(
#         max_depth=10
#     ),
#     smote=True,
#     var_filter__threshold=0.16,
#     ft_filter__k=8
# )

# rf_classifier = make_rew_classifier(
#     clf=RandomForestClassifier(
#         max_depth=5
#     ),
#     smote=True,
#     var_filter__threshold=0.16,
#     ft_filter__k=9
# )

# fnn_classifier = make_rew_classifier(
#     clf=MLPClassifier(
#         random_state=0,
#         max_iter=5000
#     ),
#     smote=True,
#     var_filter__threshold=0.16,
#     ft_filter__k=9
# )

# svm_classifier = make_rew_classifier(
#     clf=SVC(
#         gamma="auto",
#         probability=True
#     ),
#     smote=True,
#     var_filter__threshold=0.16,
#     ft_filter__k=9
# )

# lr_classifier = make_rew_classifier(
#     clf=LogisticRegression(
#         l1_ratio=0
#     ),
#     smote=True,
#     var_filter__threshold=0.16,
#     ft_filter__k=9
# )

# ann_classifier = make_rew_classifier(
#     clf=MLPClassifier(
#         hidden_layer_sizes=(100,100),
#         max_iter=5000,
#         random_state=0
#     ),
#     smote=True,
#     var_filter__threshold=0.16,
#     ft_filter__k=9
# )

# from sklearn.naive_bayes import GaussianNB
# nb_classifier = make_rew_classifier(
#     clf=GaussianNB(),
#     smote=True,
#     var_filter__threshold=0.16,
#     ft_filter__k=8
# )
################################################################################

######################### TUNED HYPERPARAMS ###################################
# knn_classifier = make_rew_classifier(
#     clf=KNeighborsClassifier(
#         n_neighbors=9,
#         p=2,
#         weights='uniform'
#     ),
#     smote=False,    
#     var_filter__threshold=0.1,
#     ft_filter__k=7
# )

# dt_classifier = make_rew_classifier(
#     clf=DecisionTreeClassifier(
#         min_samples_split=5,
#         min_samples_leaf=2,
#         max_features="log2",
#         max_depth=10,
#         criterion="gini",
#         class_weight="balanced",
#         ccp_alpha=0.01
#     ),
#     smote=True,
#     var_filter__threshold=0.1,
#     ft_filter__k=9
# )

# rf_classifier = make_rew_classifier(
#     clf=RandomForestClassifier(
#         n_estimators=200,
#         max_depth=5,
#         min_samples_leaf=5,
#         min_samples_split=5,
#         max_features="sqrt",
#         class_weight="balanced",
#         ccp_alpha=1e-3
#     ),
#     smote=False,
#     var_filter__threshold=0,
#     ft_filter__k=9
# )

# fnn_classifier = make_rew_classifier(
#     clf=MLPClassifier(
#         activation="tanh",
#         max_iter=1000
#     ),
#     smote=False,
#     var_filter__threshold=0.16,
#     ft_filter__k=7
# )

# svm_classifier = make_rew_classifier(
#     clf=SVC(
#         kernel="rbf",
#         gamma="auto",
#         degree=2,
#         class_weight="balanced",
#         C=0.01,
#         probability=True
#     ),
#     smote=True,
#     var_filter__threshold=0.16,
#     ft_filter__k=7
# )

# lr_classifier = make_rew_classifier(
#     clf=LogisticRegression(
#         solver="lbfgs",
#         l1_ratio=0,
#         max_iter=200,
#         class_weight="balanced",
#         C=0.001
#     ),
#     smote=True,
#     var_filter__threshold=0.16,
#     ft_filter__k=7
# )

# ann_classifier = make_rew_classifier(
#     clf=MLPClassifier(
#         hidden_layer_sizes=(16,),
#         activation='relu',
#         max_iter=200,
#     ),
#     smote=False,
#     var_filter__threshold=0.05,
#     ft_filter__k=9
# )

# nb_classifier = make_rew_classifier(
#     clf=BernoulliNB(
#         alpha=0
#     ),
#     smote=False,
#     var_filter__threshold=0,
#     ft_filter__k=9
# )
################################################################################

############################## ALL SMOTE #######################################
knn_classifier = make_rew_classifier(
    clf=KNeighborsClassifier(
        n_neighbors=9,
        p=2,
        weights='uniform'
    ),
    smote=True,    
    var_filter__threshold=0.1,
    ft_filter__k=7
)

dt_classifier = make_rew_classifier(
    clf=DecisionTreeClassifier(
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="log2",
        max_depth=10,
        criterion="gini",
        class_weight="balanced",
        ccp_alpha=0.01
    ),
    smote=True,
    var_filter__threshold=0.1,
    ft_filter__k=9
)

rf_classifier = make_rew_classifier(
    clf=RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=5,
        max_features="sqrt",
        class_weight="balanced",
        ccp_alpha=1e-3
    ),
    smote=True,
    var_filter__threshold=0,
    ft_filter__k=9
)

fnn_classifier = make_rew_classifier(
    clf=MLPClassifier(
        activation="tanh",
        max_iter=1000
    ),
    smote=True,
    var_filter__threshold=0.16,
    ft_filter__k=7
)

svm_classifier = make_rew_classifier(
    clf=SVC(
        kernel="rbf",
        gamma="auto",
        degree=2,
        class_weight="balanced",
        C=0.01,
        probability=True
    ),
    smote=True,
    var_filter__threshold=0.16,
    ft_filter__k=7
)

lr_classifier = make_rew_classifier(
    clf=LogisticRegression(
        solver="lbfgs",
        l1_ratio=0,
        max_iter=200,
        class_weight="balanced",
        C=0.001
    ),
    smote=True,
    var_filter__threshold=0.16,
    ft_filter__k=7
)

ann_classifier = make_rew_classifier(
    clf=MLPClassifier(
        hidden_layer_sizes=(16,),
        activation='relu',
        max_iter=200,
    ),
    smote=True,
    var_filter__threshold=0.05,
    ft_filter__k=9
)

nb_classifier = make_rew_classifier(
    clf=BernoulliNB(
        alpha=0
    ),
    smote=True,
    var_filter__threshold=0,
    ft_filter__k=9
)
################################################################################

models = {
    "K-Nearest Neighbors": knn_classifier,
    "Decision Tree": dt_classifier,
    "Random Forest": rf_classifier,
    "Feed-Forward Neural Network": fnn_classifier,
    "Support Vector Machine": svm_classifier,
    "Logistic Regression": lr_classifier,
    "Deep Neural Network": ann_classifier,
    "Naive Bayes": nb_classifier
}