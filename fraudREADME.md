
# Bank Account Fraud Classifier

**My name is Terry Lee Harris, III. I'm a graduate student at University of Denver, completing an M.S. in Applied Data Science and AI. As a former intelligence specialist and investigative analyst, I processed tens of thousands of cases annually in which applicants pursuing certain credentials were suspected, and oftentimes convicted, of fraudulence. In addition to continuously observing the rapidly-evolving technological landscape, I understand that our reliance upon online banking systems is always growing. Thus, I found myself compelled to build this binary classification pipeline which employs four machine-learning models (logistic regression, DecisionTree, RandomForest, and XGBoost) to assess the indicators of fraud and its prevalence in society at scale.**

---

## Pipeline Architecture

Stage: EDA

Description: Class imbalance analysis, sentinel value detection, feature correlation analysis

---

Stage: Preprocessing

Description: Median imputation with missing-value indicator columns; OHE + StandardScaler via ColumnTransformer

---

Stage: Training

Description: 5-fold stratified cross-validation across 4 classifiers (default configuration)

---

Stage: Evaluation

Description: Confusion matrix, classification report, ROC-AUC, Precision-Recall curve, optimal threshold selection

---

Stage: Tuning

Description: GridSearchCV with ROC-AUC scoring across full hyperparameter grids

---

Stage: Test Set Evaluation

Description: Champion model evaluated on a 20% stratified holdout set

---

## Model Selection Rationale

`Logistic Regression` — selected as the linear baseline. Start simple and scale with intention. Not susceptible to overfitting; classifies via maximum likelihood estimation of an optimal decision boundary. Helps organizations evaluate complexity tradeoffs responsibly before committing resources to more complex architectures.

`Decision Tree` — selected as a non-linear comparator. Susceptible to overfitting on high-cardinality data; included deliberately to demonstrate the limitation that bagging and boosting were each designed to address.

`RandomForest` — selected to assess the effect of *bagging* - simultaneously building an ensemble of strong estimators, each trained on a bootstrapped resample of the dataset.
Capitalizes on the DecisionTree's tendency to overfit by aggregating strong, diverse, and independently-trained trees - colloquially, an ensemble of subject-matter experts, each specialized on a perturbation of the data.

`XGBoost` — selected to assess the effect of *boosting*: sequentially building an ensemble of weak estimators, where each tree corrects the residual error of its predecessor.
Tuned with `scale_pos_weight=89` to directly compensate for the 89:1 class imbalance.

---

## Key Engineering Decisions

### Sentinel Value Handling

Three features encode missing values as `-1` rather than `NaN`: `prev_address_months_count`, `current_address_months_count`, and `bank_months_count`. Standard `isnull().sum()` reports zero missing values — a silent inaccuracy. Binary indicator columns were created *before* median imputation to preserve the missingness signal as a predictive feature. A fraudulent applicant is unlikely to disclose prior address history or existing banking relationships; their absence is itself informative.

### Single Cross-Validation Call per Model

An initial implementation called `cross_val_predict()` twice per model — once for label predictions, once for probabilities — resulting in 10 model fits per classifier over 800,000 rows. Refactored to a single call using `method='predict_proba'`, with binary labels extracted via index slicing on the probability output:

```python
probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
preds = (probs[:, 1] >= 0.5).astype(int)  # col index 1 = P(fraud)
```

This halved compute time with no change in output. Results are stored preds-first per model key: `(preds, probs)`.

### Persistence with joblib

Cross-validation results and GridSearch objects are serialized as `.pkl` files to eliminate recomputation on kernel restarts. Given a fixed `random_state`, outputs are fully reproducible — rerunning from scratch produces identical results at significant time cost (LR grid search: ~17 min; RF grid search: ~1 hr).

### XGBoost Compatibility

Upgraded XGBoost 2.1.3 → 3.2.0 via conda to resolve `AttributeError: 'super' object has no attribute '__sklearn_tags__'`. XGBoost 3.x implements the `__sklearn_tags__` protocol required by scikit-learn 1.6+.

---

## Evaluation Method Considerations

With a 98.9% legitimate / 1.1% fraud split (89:1 imbalance), standard accuracy is misleading — a model that flags every application as legitimate achieves 98.9% accuracy while detecting zero fraud. Three threshold-selection methods were evaluated:

### Precision-Recall Curve (selected)

Focuses evaluation on the minority class.
Identifies the optimal decision threshold by maximizing the F1 harmonic mean across all threshold values. Superior to ROC-AUC for imbalanced problems because it excludes true negatives, which inflate ROC performance when the negative class dominates.

**Limitation:** Does not account for true negatives. If specificity is operationally critical, the PR curve alone is insufficient.

### Youden's J Statistic (considered, rejected)

Maximizes TPR − FPR (sensitivity + specificity − 1).
Treats sensitivity and specificity as equally important.
In fraud detection, sensitivity (recall of the fraud class) substantially outweighs specificity — a missed fraudster carries a categorically higher cost than a flagged legitimate application.
Youden's J does not model this asymmetry.

### Business Cost Threshold (considered, out of scope)

Assigns a dollar cost to each misclassification type and minimizes total expected loss.
Operationally the most realistic approach, but requires institution-specific cost data not available in this synthetic dataset.

---

## Hyperparameter Tuning Rationale

```python
log_regress_param_grid = {
    'penalty': ['l1', 'l2'],        # l1 crushes weights for natural feature selection; l2 penalizes less aggressively while respecting all weights
    'C': np.logspace(-3, 3, 10),    # regularization strength sweep from 0.001 to 1000
    'solver': ['liblinear']         # supports both l1 and l2 penalties
}

decision_tree_param_grid = {
    'max_depth': [3, 5, 10, 20],        # controls tree growth to prevent pure-node memorization
    'min_samples_leaf': [1, 10, 50],    # enforces minimum support at leaf nodes
    'min_samples_split': [2, 10, 50]    # controls split granularity
}

rf_param_grid = {
    'n_estimators': [100, 200],         # ensemble size - more trees yields more stable predictions
    'max_depth': [5, 10, 20, None],     # same pruning principles as DecisionTree
    'max_features': ['sqrt', 'log2']    # features considered per split - controls diversity across trees
}

xgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.3], # step size during gradient descent - small, controlled steps down into the valley
    'max_depth': [3, 6, 9],            # shallow trees = weak learners = less overfitting per boosting round
    'n_estimators': [100, 200]         # boosting rounds — more rounds = finer residual correction
}
```

---

## Results

`Logistic Regression` — Default ROC-AUC: 0.8831 | Tuned ROC-AUC: 0.8831 | Δ: 0.0000

`Decision Tree` — Default ROC-AUC: 0.5318 | Tuned ROC-AUC: 0.8008 | Δ: +0.2690

`Random Forest` — Default ROC-AUC: 0.8199 | Tuned ROC-AUC: 0.8737 | Δ: +0.0538

`XGBoost` — Default ROC-AUC: 0.8775 | Tuned ROC-AUC: **0.8957** | Δ: +0.0182

**Champion model: XGBoost Tuned** — `learning_rate=0.1`, `max_depth=3`, `n_estimators=200`, `scale_pos_weight=89`

### Test Set Evaluation (20% holdout, n=200,000)

Precision — class_0 (Legitimate): 1.00 | class_1 (Fraud): 0.05

Recall — class_0: 0.83 | class_1: 0.80

F1-Score — class_0: 0.90 | class_1: 0.09

ROC-AUC: **0.8938**

Cross-validated ROC-AUC: **0.8957** → Test set ROC-AUC: **0.8938** (Δ = 0.0019). The negligible performance delta confirms the model generalized to unseen data without meaningful degradation.

---

## Interactive Dashboard

`bank_fraud_signal_analysis_app.py` — a companion Dash application for exploratory analysis of the raw fraud signal indicators across the full 1,000,000-application dataset.

**Device Signal Explorer** — histogram and boxplot of device/session features, filterable by class (All / Fraud Only / Legitimate Only)

**Risk Profiles** — violin plots across three feature groups: Velocity, Device/Session, and Identity/Contact

**Bivariate Risk Analysis** — scatter plot of any two features colored by fraud class, with an adjustable sample size slider (1k–50k)

```bash
pip install -r requirements.txt
python bank_fraud_signal_analysis_app.py
```

---

## Dataset

[Bank Account Fraud Dataset — NeurIPS 2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)

1,000,000 synthetic bank account applications | 32 features | 1.1% fraud prevalence (89:1 imbalance)
