\# Fault Detection - ML Challenge (IEEE SB GEHU)



\## Problem Statement



The objective is to classify whether a device is operating normally (Class = 0) or is faulty (Class = 1) using 47 numerical features.



The evaluation metrics are:

\- Accuracy

\- F1 Score



Due to slight class imbalance (~60-40), F1 Score was prioritized.



---



\## Approach



\- Performed exploratory data analysis (EDA)

\- Used Stratified 5-Fold Cross Validation to preserve class distribution

\- Trained a tuned XGBoost classifier

\- Performed out-of-fold threshold optimization

\- Final threshold selected: \*\*0.44\*\*

\- Final OOF F1 Score: ~0.9809



---



\## Model Details



\- Model: XGBoost

\- n\_estimators: 600

\- max\_depth: 7

\- learning\_rate: 0.03

\- subsample: 0.9

\- colsample\_bytree: 0.9

\- gamma: 0.1

\- regularization applied



---



\## Setup Instructions



Install dependencies:



```bash

pip install -r requirements.txt

