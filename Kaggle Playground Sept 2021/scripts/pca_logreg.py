# %%
# apply logistic regression to pca variables
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import config as cf

# load data
df = pd.read_csv(cf.TRAINING_KFOLDS_FILE)

for fold_ in range(5):
    #temp train and test sets
    train_df = df[df.kfold != fold_].reset_index(drop=True)
    test_df = df[df.kfold == fold_].reset_index(drop=True)

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

    # initialise principal component analysis and fit to training data
    pca = PCA(n_components=10)
    pca.fit(train_X)
    #cumsum = np.cumsum(pca.explained_variance_ratio_)
    #d = np.argmax(cumsum >= 0.95) + 1

    # transform train and test
    train_X = pca.transform(train_X)
    test_X = pca.transform(test_X)

    # initialise and fit logistic regression model
    model = LogisticRegression()
    model.fit(train_X, train_y)

    # predict on test data
    preds = model.predict(test_X)

    # calculate accuracy
    auc = metrics.roc_auc_score(test_y, preds)

    print(f"Fold: {fold_}")
    print(f"AUC = {auc}")
    #print(f"components = {d}")
    print('')


"""
This was a great exercise for me but the results
were useless haha. The five folds all gave me AUC around 0.51
"""
# %%
