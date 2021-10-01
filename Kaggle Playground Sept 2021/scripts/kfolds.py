# create a new version of training data with stratified k folds column

# %%
import pandas as pd
from sklearn import model_selection

import config as cf

if __name__ == "__main__":
    df = pd.read_csv(cf.TRAINING_FILE)

    # new column filled with -1
    df["kfold"] = -1

    # randomise the rows
    df = df.sample(frac=1).reset_index(drop=True)

    # get target values
    y = df.claim.values

    # initiate kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=cf.KFOLDS)

    # populate new column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold

# %%

    # save file
    df.to_csv(cf.TRAINING_KFOLDS_FILE, index=False)
    print('Training Kfolds file saved!')

# %%

    # save a sample to view in an editor
    sample = df.sample(frac=0.02)
    sample.to_csv(cf.TRAINING_KFOLDS_FILE_SAMPLE, index=False)
    print('Training Kfolds file sample saved!')

# %%
