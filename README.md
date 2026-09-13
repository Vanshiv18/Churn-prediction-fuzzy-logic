# Intelligent Customer Churn Prediction & Retention Decision Support System

An end-to-end machine learning + deep learning + fuzzy logic system that predicts customer churn and turns that prediction into a business-usable retention priority — built on the IBM Telco Customer Churn dataset.

**[Try the live Fuzzy Logic demo →](https://YOUR_GITHUB_USERNAME.github.io/YOUR_REPO_NAME/)**

![Overfitting analysis](assets/images/plot8b_four_panel_overfitting.png)

## What this project does

Most churn projects stop at "will this customer churn?" This one goes one step further and answers the question a retention manager actually needs answered: **which customers should we prioritise, and how aggressively?**

The pipeline has three layers:

1. **Exploratory data analysis** — uncovers *why* customers churn (contract type, tenure, payment method, internet service).
2. **Comparative machine learning** — Logistic Regression, Decision Tree, Random Forest, and an Artificial Neural Network (TensorFlow/Keras), fully benchmarked against each other (accuracy, precision, recall, F1, ROC-AUC, confusion matrices).
3. **Fuzzy Logic decision engine** — a 9-rule Mamdani inference system, built from scratch (no external fuzzy library), that converts a raw ANN churn probability, plus customer value and tenure, into a **Low / Medium / High retention priority** a manager can act on.

The fuzzy system is validated against real outcomes: customers it flags as **High priority actually churn at 53.9%**, versus **20.3% for Low priority** — a clean, monotonic result that confirms the system works.

## Live demo

The `docs/` folder contains a fully client-side (vanilla JS, no backend) reimplementation of the fuzzy logic engine, deployable for free on **GitHub Pages**. Move the sliders for churn risk, customer value, and tenure, and watch the retention priority gauge update live.

To enable it on your own fork: **Settings → Pages → Deploy from branch → `main` / `docs`**.

## Results at a glance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| Decision Tree | 0.801 | 0.667 | 0.497 | 0.570 | 0.822 |
| Random Forest | 0.804 | 0.667 | 0.524 | 0.587 | 0.843 |
| ANN (Keras, base config) | 0.779 | 0.592 | 0.543 | 0.566 | 0.772 |

**Key findings:**
- Overall churn rate: **26.5%**
- Month-to-month contracts churn at **42.7%** vs. **2.8%** for two-year contracts
- Electronic-check payers churn at **45.3%** vs. **~15–17%** for automatic payment methods
- A controlled overfitting experiment shows validation loss diverging from training loss almost immediately — regularisation + early stopping improved every metric
- An overly aggressive learning rate (0.1) caused the ANN to collapse entirely, predicting only the majority class — a real, reproducible cautionary finding, not a synthetic one

Full methodology, all 20 figures, real code screenshots, and console output are in the [project report](reports/Churn_Prediction_Fuzzy_Retention_Report.pdf).

## Fuzzy Logic engine

![Fuzzy membership functions](assets/images/plotX9_fuzzy_membership.png)

Three inputs — **Churn Risk** (ANN probability), **Customer Value** (monthly charges), and **Tenure** — are each converted into degrees of Low/Medium/High membership using triangular and trapezoidal functions, combined via 9 hand-justified Mamdani rules, and defuzzified with the centroid method into a single 0–100 retention priority score.

![Fuzzy decision surface](assets/images/plotX10_fuzzy_surface.png)

## Repository structure

```
.
├── data/
│   └── telco_customer_churn.csv        # Raw IBM Telco Customer Churn dataset
├── src/
│   ├── 01_data_prep.py                 # Cleaning, encoding, feature engineering
│   ├── 02_eda.py                       # Exploratory data analysis + charts
│   ├── 03_ml_models.py                 # Logistic Regression / Decision Tree / Random Forest
│   ├── 04a_ann_architecture.py         # ANN architecture experiment (Keras)
│   ├── 04b_ann_hyperparameters.py      # Learning rate / epochs / batch size experiments
│   ├── 04c_ann_overfitting_regularisation.py  # Overfitting analysis + L2/early stopping
│   ├── 05_fuzzy_logic.py               # Fuzzy Logic retention decision system
│   └── 06_extra_charts.py              # ROC/PR curves, membership plots, decision surface
├── reports/
│   └── Churn_Prediction_Fuzzy_Retention_Report.pdf   # Full write-up (47 pages)
├── assets/images/                      # Key charts used in this README
├── docs/
│   └── index.html                      # Live client-side fuzzy logic demo (GitHub Pages)
├── requirements.txt
└── LICENSE
```

## Running it yourself

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt

cd src
python 01_data_prep.py
python 02_eda.py
python 03_ml_models.py
python 04a_ann_architecture.py
python 04b_ann_hyperparameters.py
python 04c_ann_overfitting_regularisation.py
python 05_fuzzy_logic.py
python 06_extra_charts.py
```

Each script reads the CSV outputs of the previous one, so run them in order the first time. Scripts `01`–`03` regenerate the cleaned dataset, train/test split, and baseline models; `04a`–`04c` train and evaluate the Keras ANN; `05` applies the fuzzy logic engine on top of the trained ANN; `06` produces the additional ROC/PR/membership/surface charts.

## Tech stack

- **Data & ML:** pandas, NumPy, scikit-learn
- **Deep Learning:** TensorFlow / Keras
- **Fuzzy Logic:** implemented from first principles (no external fuzzy library)
- **Visualisation:** Matplotlib, Seaborn
- **Live demo:** vanilla HTML/CSS/JS (no framework, no backend)

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 variables, distributed via Kaggle.

## Author

**Shivam** — MBA (Data Science & AI), Mittal School of Business, Lovely Professional University.
Built as an academic project for MGNM–525 (Intelligent Customer Churn Prediction and Retention Decision Support System using ANN and Fuzzy Logic).

## License

MIT — see [LICENSE](LICENSE).
