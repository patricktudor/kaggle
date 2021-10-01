# scripts/train.py

import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import config as cf
import model_dispatcher as md

# %%

def run(fold, model):
    # load data
    df = pd.read_csv(cf.TRAINING_KFOLDS_FILE)

    # temp train and test sets
    train_df = df[df.kfold != fold].reset_index(drop=True)
    test_df = df[df.kfold == fold].reset_index(drop=True)

    # summarise missing values in new column
    train_df['n_missing'] = train_df.isna().sum(axis=1)
    test_df['n_missing'] = test_df.isna().sum(axis=1)

    # columns for modelling
    train_X = train_df[train_df.columns.drop(["id", "claim", "kfold"])].values
    train_y = train_df["claim"]

    test_X = test_df[test_df.columns.drop(["id", "claim", "kfold"])].values
    test_y = test_df["claim"]

    # set up and fit imputer for missing values
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_X)

    # replace missing values with imputed values
    train_X = imputer.transform(train_X)
    test_X = imputer.transform(test_X)

    # convert to dataframe
    #train_X = pd.DataFrame(train_miss, columns=train_X.columns, index=train_X.index)
    #test_X = pd.DataFrame(test_miss, columns=test_X.columns, index=test_X.index)

    # scale data to mean = 0, var = 1
    scaler = StandardScaler()

    # fit on training data
    scaler.fit(train_X)

    # transform train and test
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # fit model
    clf = md.models[model]
    clf.fit(train_X, train_y)

    # predict on test data
    preds = clf.predict_proba(test_X)[:, 1]

    # calculate accuracy
    auc = metrics.roc_auc_score(test_y, preds)
    print(f"Fold = {fold}, AUC = {auc}")

    # save model
    joblib.dump(
        clf,
        Path(cf.MODEL_OUTPUT, f"{model}_{fold}.bin")
    )

if __name__ == "__main__":
    # argument parser class
    parser = argparse.ArgumentParser()

    # add arguments needed
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )

    # read arguments from command line
    args = parser.parse_args()

    # run fold specified in command line arguments
    run(fold=args.fold, model=args.model)