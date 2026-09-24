# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:27:03 2026

@author: B317980
"""

import os
import pandas as pd
import numpy
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import gzip
import pickle
import json
from scipy.stats import randint, uniform


# %%

load_dir = "."
save_dir = "."
# %%

with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    agg_df_train = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    agg_df_val = pickle.load(f)

# %%

X_train = agg_df_train.drop(columns=["label", "seq_id"])
y_train = agg_df_train["label"]

X_val = agg_df_val.drop(columns=["label", "seq_id"])
y_val = agg_df_val["label"]

# %%

num_pos = (agg_df_train["label"] == 1).sum()
num_neg = (agg_df_train["label"] == 0).sum()
pos_weight = (num_neg / num_pos) / 3

# %%

model = xgb.XGBClassifier(
    objective="binary:logistic",
    booster="gbtree",
    tree_method="hist",
    eval_metric="aucpr",
    scale_pos_weight=pos_weight,
    random_state=123,
    device="cuda",
    n_estimators=500,
    early_stopping_rounds=10
    )

param_dist = {
    "max_depth": randint(3,9),
    "learning_rate": uniform(0.02, 0.18),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 10),
    "reg_alpha": uniform(0.0, 1.0),
    "reg_lamda": uniform(0.5, 2.0)
    }

search = RandomSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=25,
    scoring="average_precision",
    cv=3,
    verbose=1,
    random_state=123,
    )

search.fit(X_train, y_train,
           eval_set=[(X_val, y_val)],
           verbose=False
           )

# %%

with open(os.path.join(save_dir, "."), "w") as f:
    json.dump(search.best_params_, f, indent=2)

# %%


model = xgb.XGBClassifier(
    objective="binary:logistic",
    booster="gbtree",
    tree_method="hist",
    eval_metric="aucpr",
    scale_pos_weight=pos_weight,
    random_state=123,
    device="cuda",
    n_estimators=10000,
    early_stopping_rounds=30,
    **search.best_params_
    )

model.fit(X_train, y_train,
           eval_set=[(X_val, y_val)],
           verbose=True
           )

# %%

model.save_model(os.path.join(save_dir, "."))