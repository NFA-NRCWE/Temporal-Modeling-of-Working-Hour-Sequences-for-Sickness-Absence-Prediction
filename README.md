# Temporal Modeling of Working Hour Sequences for Sickness Absence Prediction

This repository contains example model code used in the study:

**Temporal Modeling of Working Hour Sequences for Sickness Absence Prediction**

The study investigates how to represent temporally ordered sequences of working-hour
and health-related events and whether it improves prediction of sickness absence compared
with models based on aggregated features.

**Data availability**

The data used are based on individual-level administrative register data from Statistics Denmark and are not available to the public. Access to Danish register data requires approval from Statistics Denmark and must be conducted through collaboration with a Danish research institution. Further information on access procedures is available at: https://www.dst.dk/en/TilSalg/data-til-forskning

## Repository structure

```text
.
├── BERT_pretrainig/
│   ├── BERT_pretraining.py
│   └── BERT_pretraining_functions.py
│
├── BERT_finetune/
│   ├── BERT_finetuning.py
│   └── BERT_finetuning_functions.py
│
├── GRU/
│   ├── GRU.py
│   └── GRU_functions.py
│
├── MLP/
│   ├── MLP.py
│   └── MLP_functions.py
│
├── LR/
│   └── Logistic_regression.py
│
├── XGBoost/
│   └── XGB.py
│
└── README.md
