# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:26:12 2026

@author: B317980
"""


import os
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split, RandomSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import gzip
import pickle
import StandardScaler
from sklearn.linear_model import LogisticRegression


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

numeric_cols = X_train.select_dtypes(include="number").columns

cont_cols = [
    col for col in numeric_cols
    if X_train[col].nunique(dropna=True) > 2
    ]

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

scaler = StandardScaler()
X_train_scaled[cont_cols] = scaler.fit_transform(X_train[cont_cols])
X_val[cont_cols] = scaler.transform(X_val[cont_cols])

# %%

lr_model = LogisticRegression(
    penalty="l2",
    random_state=123,
    class_weight={0:1, 1:pos_weight}
    )

lr_model.fit(X_train_scaled, y_train)

