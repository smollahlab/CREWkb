from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
import numpy as np


def loocv(model, X, y):
    """Perform Leave-One-Out Cross Validation on a dataset using a given model.
    
    Parameters
    ----------
    model : estimator
        Classifier to validate. Must implement `fit` and `predict_proba`.
    X : ndarray of shape (n_samples, n_features)
        Feature matrix to validate on.
    y : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Ground truth labels to validate on.

    Returns
    -------
    acc : float
        Accuracy score.
    f1 : float
        Macro F1 score.
    auc : float
        AUC ROC.
    mcc : float
        Matthews Correlation coefficient.
    cm : ndarray of shape (n_classes, n_classes)
        Confusion matrix where the `i`th row and `j`th column is the number of 
        class `i` samples that were classified as class `j`. 
    """
    y_proba = cross_val_predict(
        model, 
        X, 
        y, 
        cv=LeaveOneOut(), 
        method="predict_proba"
    )
    y_preds = np.argmax(y_proba, axis=1)
    
    acc = accuracy_score(y, y_preds)
    f1 = f1_score(y, y_preds, average='macro')
    auc = roc_auc_score(y, y_proba, multi_class='ovr')
    cm = confusion_matrix(y, y_preds)
    mcc = matthews_corrcoef(y, y_preds)
    return acc, f1, auc, mcc, cm


def kfoldcv(model, X, y, k=5, random_state=None):
    """Perform k-fold stratified cross validation on a dataset using a given model.
    
    Parameters
    ----------
    model : estimator
        Classifier to validate. Must implement `fit` and `predict_proba`.
    X : ndarray of shape (n_samples, n_features)
        Feature matrix to validate on.
    y : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Ground truth labels to validate on.
    k : int
        Number of folds for cross validation.
    random_state : int, default=`None`
        Controls the shuffling order of samples. Pass in an int for reproducible
        outputs.

    Returns
    -------
    acc : float
        Accuracy score.
    f1 : float
        Macro F1 score.
    auc : float
        AUC ROC.
    mcc : float
        Matthews Correlation coefficient.
    cm : ndarray of shape (n_classes, n_classes)
        Confusion matrix where the `i`th row and `j`th column is the number of 
        class `i` samples that were classified as class `j`. 
    """
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    accs = []
    f1s = []
    aucs = []
    mccs = []
    cms = []

    for split, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_preds = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        accs.append(accuracy_score(y_test, y_preds))
        f1s.append(f1_score(y_test, y_preds, average="macro"))
        aucs.append(roc_auc_score(y_test, y_proba, multi_class="ovr"))
        mccs.append(matthews_corrcoef(y_test, y_preds))
        cms.append(confusion_matrix(y_test, y_preds))

    acc = np.mean(accs)
    f1 = np.mean(f1s)
    auc = np.mean(aucs)
    mcc = np.mean(mccs)
    cm = np.sum(cms, axis=0)
    return acc, f1, auc, mcc, cm