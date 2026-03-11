import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns



################################ CLASSIFIERS ###################################
class Model():
    """Generic class for performing Leave-One-Out Cross Validation using a given classifier.

    Parameters
    ----------
    name : str
        Name of classifier pipeline.
    classifier : sklearn-compatible estimator
        A classifier implementing the scikit-learn estimator interface. Must provide
        `fit`, `predict`, and `predict_proba` methods.
    feature_scoring_method : sklearn-compatible feature selection method
        Scoring function to rank features by predictive power (default is `f_classif`).
    num_features : int
        Number of features selected by `SelectKBest`.
    var_threshold = float
        Minimum variance for a given feature in the training data required to be kept 
        using `VarianceThreshold`.
    """

    def __init__(self, 
                 name,
                 classifier,
                 feature_scoring_method=f_classif,
                 num_features=8,
                 var_threshold=0.8 * (1-0.8)):
        self.name = name
        self.classifier = classifier
        self.variance_filter = VarianceThreshold(threshold=var_threshold)
        self.feature_filter = SelectKBest(score_func=feature_scoring_method, 
                                          k=num_features)
        self.pipeline = make_pipeline(
            self.variance_filter,
            self.feature_filter,
            self.classifier
        )

    def split_data(self, X, y, train_idxs, test_idxs):
        """Divide input data into train and test sets."""
        return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]
    
    def fit(self, X_train, y_train):
        """Fit pre-processing and classification pipeline onto training data."""
        self.pipeline.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Generate class label predictions from fitted pipeline on test data."""
        return self.pipeline.predict(X_test)
    
    def predict_proba(self, X_test):
        """Generate class label probabilities from fitted pipeline on test data."""
        return self.pipeline.predict_proba(X_test)
    
    def eval_accuracy(self, y_truth, y_preds):
        """Calculate accuracy from predicted class labels versus ground truth."""
        return accuracy_score(y_truth, y_preds)

    def eval_f1_score(self, y_truth, y_preds, average='weighted'):
        """Calculate F1 score from predicted class labels versus ground truth."""
        return f1_score(y_truth, y_preds, average=average)

    def eval_auc(self, y_truth, y_preds):
        """Calculate AUC for ROC from predicted class probabilities versus grouth truth."""
        return roc_auc_score(y_truth, y_preds, multi_class='ovr')
    
    def eval_confusion_matrix(self, y_truth, y_preds):
        """Compute confusion matrix based on ground truth labels and corresponding predictions."""
        return confusion_matrix(y_truth, y_preds)
    
    def get_confusion_matrix(self, confusion_matrix, class_labels, save_to=None, show=False):
        """Generate a heatmap visualization of a confusion matrix.
        
        Parameters
        ----------
        confusion_matrix : array-like of shape (n_classes, n_classes)
            Confusion matrix containing counts of predicted versus true labels.
        class_labels: list of str
            Labels corresponding to each class, used to annotate the axes of the
            confusion matrix.
        save_to : str or None, default=None
            File path to save the figure as a PNG image. If `None`, the figure
            is not saved.
        show : bool, default=False
            If `True`, display the figure using `plt.show()`.

        Returns
        -------
        plt : matplotplib.pyplot
            The matplotlib pyplot module containing the generated figure.
        """
        sns.set_theme(style="white", font_scale=1.2)
        plt.figure(figsize=(4, 4))
        ax = sns.heatmap(confusion_matrix, 
                        annot=True, 
                        square=True, 
                        xticklabels=class_labels, 
                        yticklabels=class_labels, 
                        cmap='RdPu', 
                        fmt="d",
                        cbar_kws={"shrink": 0.85})
        ax.xaxis.tick_top()
        ax.yaxis.tick_left()
        ax.set_xticklabels(class_labels, fontsize=14)
        ax.set_yticklabels(class_labels, fontsize=14)
        plt.xlabel(self.name, fontsize=14, labelpad=11)
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        if save_to:
            plt.savefig(save_to, 
                        format="png", 
                        bbox_inches='tight', 
                        dpi=600)
        if show:
            plt.show()
        return plt
    
    def loocv_eval(self, X, y):
        """Perform Leave-One-Out Cross Validation on the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Ground truth target labels.

        Returns
        -------
        accuracy : float
            Classification accuracy across all folds.
        f1_score : float
            Weighted F1 score across all folds.
        auc_roc : float
            One-vs-rest multiclass ROC AUC score.
        confusion_matrix : ndarray of shape (n_classes, n_classes)
            Confusion matrix summarizing predictions.        
        """

        preds = []
        probs = []
        y_truth = []
        for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(X=X)):
            X_train, y_train, X_test, y_test = self.split_data(X, y, train_idx, test_idx)
            self.fit(X_train, y_train)
            preds.append(self.predict(X_test))
            probs.append(self.predict_proba(X_test)[0])
            y_truth.append(y_test)
        preds = np.concatenate(preds, axis=0)       # shape: (n_samples,)
        probs = np.array(probs)                     # shape: (n_samples, n_classes)
        y_truth = np.concatenate(y_truth, axis=0)   # shape: (n_samples,)
        accuracy = self.eval_accuracy(y_truth, preds)
        f1_score = self.eval_f1_score(y_truth, preds)
        auc_roc = self.eval_auc(y_truth, probs)
        confusion_matrix = self.eval_confusion_matrix(y_truth, preds)
        return accuracy, f1_score, auc_roc, confusion_matrix
    
    def fold_cv_eval(self, X, y, n_folds):
        """Perform k-fold cross validation on the dataset.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Ground truth target labels.
        n_folds : int
            Number of folds to perform cross validation.

        Returns
        -------
        accuracy : float
            Classification accuracy across all folds.
        f1_score : float
            Weighted F1 score across all folds.
        auc_roc : float
            One-vs-rest multiclass ROC AUC score.
        confusion_matrix : ndarray of shape (n_classes, n_classes)
            Confusion matrix summarizing predictions.       
        """

        agg_preds_list = []
        agg_y_truths_list = []

        accuracies = []
        f1_scores = []
        auc_rocs = []
        kfolds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = 42)
        for fold, (train_idx, test_idx) in enumerate(kfolds.split(X=X, y=y)):
            X_train, y_train, X_test, y_test = self.split_data(X, y, train_idx, test_idx)
            self.fit(X_train, y_train)
            preds = self.predict(X_test)
            probs = self.predict_proba(X_test)

            accuracies.append(self.eval_accuracy(y_test, preds))
            f1_scores.append(self.eval_f1_score(y_test, preds))
            auc_rocs.append(self.eval_auc(y_test, probs))
            
            agg_preds_list.append(preds)
            agg_y_truths_list.append(y_test)

        agg_preds = np.concatenate(agg_preds_list, axis=0)
        agg_y_truths = np.concatenate(agg_y_truths_list, axis=0)
        confusion_matrix = self.eval_confusion_matrix(agg_y_truths, agg_preds)
        return np.mean(accuracies), np.mean(f1_scores), np.mean(auc_rocs), confusion_matrix

    
# Naive Bayes
nb_classifier = BernoulliNB()
nb_model = Model(name='Naive Bayes', classifier=nb_classifier)

# Multi-Layer Perceptron
mlp_classifier = MLPClassifier(hidden_layer_sizes = (100, 100),
                                    activation='relu', 
                                    solver='lbfgs',
                                    random_state=42,
                                    max_iter=5000)
mlp_model = Model(name='Multi-Layer Perceptron', classifier=mlp_classifier)

# Deep Neural Network
dnn_classifier = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                                    activation='relu', 
                                    solver='lbfgs',
                                    random_state=42,
                                    max_iter=5000)
dnn_model = Model(name='Deep Neural Network', classifier=dnn_classifier)


# K-Nearest Neighbor
knn_classifier = KNeighborsClassifier(n_neighbors=10,
                                    metric='hamming')
knn_model = Model(name='K-Nearest Neighbor', classifier=knn_classifier)
        

# Decision Tree
dt_classifier = DecisionTreeClassifier(criterion='gini',
                                    max_depth=6)
dt_model = Model(name='Decision Tree', classifier=dt_classifier)


# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100,
                                    criterion='gini',
                                    bootstrap=True,
                                    max_depth=6)
rf_model = Model(name='Random Forest', classifier=rf_classifier)


# Support Vector Machine
svc_classifier = SVC(probability=True)
svc_model = Model(name='Support Vector Machine', classifier=svc_classifier)


# Logistic Regression
lr_classifier = LogisticRegression(l1_ratio=0)
lr_model = Model(name='Logistic Regression', classifier=lr_classifier)


models = [nb_model, mlp_model, dnn_model, knn_model, dt_model, rf_model, svc_model, lr_model]