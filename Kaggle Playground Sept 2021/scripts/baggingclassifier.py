# %%
# apply logistic regression to pca variables
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import config as cf

# load data
df = pd.read_csv(cf.TRAINING_KFOLDS_FILE)

# %%

for fold_ in range(cf.KFOLDS):
    # temp train and test sets
    train_df = df[df.kfold != fold_].reset_index(drop=True)
    test_df = df[df.kfold == fold_].reset_index(drop=True)

    # summarise missing values in new column
    train_df['n_missing'] = train_df.isna().sum(axis=1)
    test_df['n_missing'] = test_df.isna().sum(axis=1)

    # columns for modelling
    train_X = train_df[train_df.columns.drop(["id", "claim", "kfold"])]
    train_y = train_df["claim"]

    test_X = test_df[test_df.columns.drop(["id", "claim", "kfold"])]
    test_y = test_df["claim"]

    # set up and fit imputer for missing values
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_X)

    # replace missing values with imputed values
    train_miss = imputer.transform(train_X)
    test_miss = imputer.transform(test_X)

    # convert to dataframe
    train_X = pd.DataFrame(train_miss, columns=train_X.columns, index=train_X.index)
    test_X = pd.DataFrame(test_miss, columns=test_X.columns, index=test_X.index)

    # scale data to mean = 0, var = 1
    scaler = StandardScaler()

    # fit on training data
    scaler.fit(train_X)

    # transform train and test
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # bagging classifier
    model = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=50, max_samples=100, bootstrap=True, n_jobs=2)
    model.fit(train_X, train_y)

    # predict on test data
    preds = model.predict_proba(test_X)[:, 1]

    # calculate accuracy
    auc = metrics.roc_auc_score(test_y, preds)

    print(f"Fold: {fold_}")
    print(f"AUC = {auc}")
    print('')


# Similar performance to logistic regression

# %%
# create test submission

# load data
sub_df = pd.read_csv(cf.TEST_FILE)

# processing
sub_df['n_missing'] = sub_df.isna().sum(axis=1)
sub = sub_df[sub_df.columns.drop(["id"])]
sub_miss = imputer.transform(sub)
sub = pd.DataFrame(sub_miss, columns=sub.columns, index=sub.index)
sub = scaler.transform(sub)
print("test set processed")

# %%
# submission predictions
sub_preds = model.predict_proba(sub)[:, 1]

# %%
submission = pd.DataFrame()
submission["id"] = sub_df["id"]
submission["claim"] = sub_preds

# %%
submission.head()

# %%
submission.to_csv(cf.SUBMISSION_PATH + "baggingclassifier.csv", index=False)
print("submission has been saved!")

# End

# %%
