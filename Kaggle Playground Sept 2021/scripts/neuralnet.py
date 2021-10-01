# scripts/train.py

# %%
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras

import config as cf

# %%
# load data
df = pd.read_csv(cf.TRAINING_KFOLDS_FILE)

# %%
# set up neural network
model = keras.models.Sequential([
keras.layers.InputLayer(input_shape=[119]),
keras.layers.Dense(50, activation="relu"),
keras.layers.Dense(20, activation="relu"),
keras.layers.Dense(1, activation="sigmoid"),
])

model.summary()

model.compile(loss="binary_crossentropy",
optimizer=keras.optimizers.SGD(learning_rate=0.02),
metrics=["AUC"]
)

# %%
for fold_ in range(cf.KFOLDS):
    # temp train and test sets
    train_df = df[df.kfold != fold_].reset_index(drop=True)
    test_df = df[df.kfold == fold_].reset_index(drop=True)

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

    # scale data to mean = 0, var = 1
    scaler = StandardScaler()

    # fit on training data
    scaler.fit(train_X)

    # transform train and test
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # fit model
    model.fit(train_X, train_y, epochs=2,
    validation_data=(test_X, test_y))

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
sub_preds = model.predict(sub)

# %%
submission = pd.DataFrame()
submission["id"] = sub_df["id"]
submission["claim"] = sub_preds

# %%
submission.head()

# %%
submission.to_csv(cf.SUBMISSION_PATH + "neuralnetwork.csv", index=False)
print("submission has been saved!")

# End

# %%
