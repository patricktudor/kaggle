# scripts/model_dispatcher.py

from sklearn import ensemble
from sklearn import linear_model
from sklearn import tree

models = {
    "decision_tree_gini": ensemble.BaggingClassifier(
        tree.DecisionTreeClassifier(max_depth=2), 
        n_estimators=50, max_samples=100, bootstrap=True, n_jobs=2
        ),
    "lr": linear_model.LogisticRegression(),
}