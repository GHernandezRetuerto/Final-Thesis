import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


grid_lr = {
    # 'classifier': LogisticRegression(),
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': np.logspace(-4, 4, 20),
    'classifier__fit_intercept': [True, False],
    'classifier__solver': ['liblinear']
}

grid_knn = {
    # 'classifier': KNeighborsClassifier(),
    'classifier__n_neighbors': list(range(1, 21, 2)),
    'classifier__metric': ['euclidean', 'manhattan'],
    'classifier__weights': ['uniform', 'distance']
}

grid_svm = {
    # 'classifier': SVC(),
    'classifier__kernel': ['rbf', 'sigmoid'],
    'classifier__C': [10.0, 1.0, 0.1, 0.01],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__gamma': ['scale', 'auto'],
    # 'classifier__class_weight': ['balanced']
}

grid_nb = {
    # 'classifier': BernoulliNB(),
    'classifier__alpha': [x / 10.0 for x in range(0, 11)],
}

grid_rf = {
    # 'classifier': RandomForestClassifier(),
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__n_estimators': np.round(np.logspace(0, 3, 10)).astype(int),
    'classifier__max_leaf_nodes': [None],
    'classifier__class_weight': ['balanced', 'balanced_subsample']
}

grid_xgb = {
    # 'classifier': XGBClassifier(use_label_encoder=False)
    'classifier__n_estimators': np.round(np.logspace(0, 3, 10)).astype(int),
    'classifier__depth': np.round(np.logspace(0, 3, 10)).astype(int),
    'classifier__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
}

