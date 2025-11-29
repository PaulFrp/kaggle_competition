# Marketing Conversion Prediction Pipeline

This project is a machine learning pipeline designed to predict high-probability buyers for a marketing campaign and maximize ROI. The pipeline cleans raw datasets, trains an XGBoost model, optimizes hyperparameters with Optuna, selects important features, and generates a marketing report highlighting the top users to target.

---

## Project Structure

```
kaggle_competition/
├── data/
│ ├── train_dataset_M1_with_id.csv
│ └── test_dataset_M1_with_id.csv
├── cleaning/
│ ├── knn_imputation.py
│ ├── cat_cleaning.py
│ ├── one_hot.py
│ ├── minmax.py
│ ├── seasonality_features.py
│ └── cleaning_pipeline.py
├── model/
│ ├── model_training.py
│ └── analysis.py
├── outputs/
│ ├── submission_final.csv
│ └── submission_final_with_probabilities.csv
├── main_pipeline.py
├── requirements.txt
└── README.md
```

---

## Installation

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the full pipeline:

```bash
python pipeline.py
```

This will:

1. Load and clean the training and test datasets  
2. Prepare features for modeling  
3. Train a baseline **XGBoost** model  
4. Optimize hyperparameters using **Optuna**  
5. Compute feature importance and select top features  
6. Train the final model using selected features  
7. Generate predictions and probabilities for the test set  
8. Produce a marketing report and select the **top 2,000 users** expected to provide the highest ROI  

---

## Marketing Report

The marketing report includes:

- **targeted_users**: Number of users to target (default: 2000)  
- **expected_revenue (€)**: Sum of expected revenue for top users  
- **ROI**: Expected return on investment based on budget  
- **expected_purchases**: Sum of predicted purchase probabilities for top users  

The top users are saved to:

```
outputs/top_2000_users.csv
```

---

## How It Works

### 1. Data Cleaning

- Impute missing numeric values using **KNN**  
- Encode categorical variables with **one-hot encoding**  
- Normalize numeric features with **MinMax scaling**  
- Add **seasonality features** (time-of-day, day-of-week, etc.)

### 2. Feature Preparation

- Drop ID columns and target from training features  
- Ensure the test set has the **same feature columns**

### 3. Model Training

- Use **XGBoost classifier** with a **time-series split** for cross-validation  
- Optimize hyperparameters using **Optuna**  
- Calculate feature importance and optionally reduce feature set  

### 4. Marketing Optimization

Compute expected revenue per user:

```python
expected_revenue = purchase_proba * (Price * (1 - Discount))
```

Rank users by expected revenue and select the top N users.  

Compute ROI:

```python
ROI = total_expected_revenue / budget
```

---

## Requirements

Key packages (see `requirements.txt` for full list):

- `numpy`, `pandas`  
- `scikit-learn`  
- `xgboost`  
- `optuna`  
- `SQLAlchemy` (optional, for database interaction)

