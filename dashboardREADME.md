# Bank Account Fraud: A Signal Analysis

An interactive Dash application for exploring fraud signal indicators in bank account applications. Built using the NeurIPS 2022 Bank Account Fraud dataset (1,000,000 applications).

---

## Dataset

The dataset is sourced from the 2022 NeurIPS conference and is publicly available on Kaggle.

> [Bank Account Fraud Dataset — NeurIPS 2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)

Download `Base.csv` from Kaggle and place it in the same directory as `bank_fraud_signal_analysis_app.py` before running.

---

## Setup

**1. Clone or download the repository.**

**2. Install dependencies.**

```bash
pip install -r requirements.txt
```

**3. Ensure `Base.csv` is in the project directory.**

---

## Running the Application

**Option A — via the menu launcher:**

```bash
python signal_analysis_main.py
```

Select `Option 1` to launch the Dash app. The menu will block until the server is stopped (`Ctrl+C`), then return to the prompt.

**Option B — directly:**

```bash
python bank_fraud_signal_analysis_app.py
```

Once running, open a browser and navigate to:

```
http://127.0.0.1:8050
```

---

## Application Sections

### KPI Badges
Displays three top-line metrics computed from the full dataset: total fraudulent applications, total legitimate applications, and overall fraud rate.

### Account Dashboard
A paginated data table showing the first 1,000 rows of the dataset, illustrating the real-world class imbalance between fraudulent and legitimate accounts.

### Device Signal Explorer
A histogram and box plot for analyzing device and session features — including `foreign_request`, `email_is_free`, `device_os`, `device_distinct_emails_8w`, `device_fraud_count`, `session_length_in_minutes`, and `keep_alive_session` — against the fraud label. Users can filter by application class via the dropdown and radio buttons.

### Risk Profiles
Violin plots displaying feature distributions by fraud class across three risk groups — Velocity, Device & Session, and Identity & Contact. Users select a group via radio buttons; each feature in the group receives its own subplot.

### Bivariate Risk Analysis
A scatter plot for exploring pairwise relationships between any two features, colored by fraud class. Users select X and Y axis features via dropdowns and control sample size via a slider (1k–50k).

---

## Project Structure

```
Bank Account Fraud Signal Analysis/
├── bank_fraud_signal_analysis_app.py   # Main Dash application
├── signal_analysis_main.py             # Menu launcher
├── Base.csv                            # Dataset (download from Kaggle)
├── requirements.txt                    # Python dependencies
├── assets/
│   └── style.css                       # Dark-mode theme
└── README.md
```

---

## References

- Cheng, D., et al. (2023). *Anti-Money Laundering by Group-Aware Deep Graph Learning.* IEEE. [Link](https://ieeexplore-ieee-org.du.idm.oclc.org/document/10114503)
- Ofoeda, J., et al. (2020). *Anti-money laundering regulations and financial sector development.* IJFE. [Link](https://onlinelibrary-wiley-com.du.idm.oclc.org/doi/full/10.1002/ijfe.2360)
- Jesus, S., et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation.* NeurIPS 2022.
