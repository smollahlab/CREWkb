import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import LabelEncoder
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
import json
import argparse


############################# DATA PRE-PROCESSING ##############################
def clean_data(df):

    # DROP NON-FEATURE COLUMNS
    df.drop(['ID (REMOVE)', 'SYMBOL', 'HGNC approved name'], axis=1, inplace=True)
    df.replace('#', np.nan, inplace=True)

    # REMOVE ROWS WITHOUT DATA
    to_check = df.columns[5:] # cols: 'Modification' , 'Protein complex', 'Target molecule', 'Target entity', 'Product'
    df.dropna(how='all', subset=to_check, inplace=True)
    df.replace(np.nan, '-', inplace = True)

    # SMOTEN CLASS REBALANCING
    X_ = df.iloc[:, 1:].astype('category')
    y_ = df.iloc[:, 0]
    le = LabelEncoder()
    y_categorical = le.fit_transform(y_)
    class_labels = list(le.classes_) # used for plotting
    X_res, y = SMOTEN(random_state=42).fit_resample(X_, y_categorical)

    # ONE-HOT ENCODING
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X = encoder.fit_transform(X_res)

    return X, y, class_labels

################################ CLASSIFIERS ###################################
class Model():
    """
    Generic template for performing leave one out cross validation for a given classifier.

    Attributes
    ----------
    name : str
    classifier : 

    """
    def __init__(self, 
                 name,
                 classifier,
                 cv_method=LeaveOneOut,
                 feature_scoring_method=f_classif,
                 num_features=8,
                 var_threshold=0.8 * (1-0.8)):
        self.name = name
        self.classifier = classifier
        self.loocv = cv_method()
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
        """Fit pre-processing and classificaiton pipeline onto training data."""
        self.pipeline.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Generate class label predictions from the fitted pipeline on test data."""
        return self.pipeline.predict(X_test)
    
    def predict_proba(self, X_test):
        """Generate class label probabilities from the fitted pipeline on test data."""
        return self.pipeline.predict_proba(X_test)
    
    def eval_accuracy(self, y_truth, y_preds):
        """Calculate accuracy from the predicted class labels versus ground truth."""
        return accuracy_score(y_truth, y_preds)

    def eval_f1_score(self, y_truth, y_preds, average='weighted'):
        return f1_score(y_truth, y_preds, average=average)

    def eval_auc(self, y_truth, y_preds):
        return roc_auc_score(y_truth, y_preds, multi_class='ovr')
    
    def eval_confusion_matrix(self, y_truth, y_preds):
        return confusion_matrix(y_truth, y_preds)
    
    def get_confusion_matrix(self, confusion_matrix, save_to=None, show=False):
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
        preds = []
        probs = []
        y_truth = []
        for fold, (train_idx, test_idx) in enumerate(self.loocv.split(X=X)):
            X_train, y_train, X_test, y_test = self.split_data(X, y, train_idx, test_idx)
            self.fit(X_train, y_train)
            preds.append(self.predict(X_test))
            probs.append(self.predict_proba(X_test).flatten())
            y_truth.append(y_test)
        preds = np.concatenate(preds, axis=0)       # shape: (n_samples,)
        probs = np.array(probs)                     # shape: (n_samples, n_classes)
        y_truth = np.concatenate(y_truth, axis=0)   # shape: (n_samples,)
        accuracy = self.eval_accuracy(y_truth, preds)
        f1_score = self.eval_f1_score(y_truth, preds)
        auc_roc = self.eval_auc(y_truth, probs)
        confusion_matrix = self.eval_confusion_matrix(y_truth, preds)
        return accuracy, f1_score, auc_roc, confusion_matrix
