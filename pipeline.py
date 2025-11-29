# main_pipeline.py

import pandas as pd
from cleaning.cleaning_pipeline import full_cleaning_pipeline
from model.analysis import generate_marketing_report
from model.model_training import (
    prepare_features,
    baseline_xgb,
    xgb_optuna_search,
    get_feature_importance,
    feature_subset_testing,
    train_final_model,
)

# Load data 
train_raw = pd.read_csv("data/train_dataset_M1_with_id.csv")
test_raw = pd.read_csv("data/test_dataset_M1_with_id.csv")

# Clean data
train_clean = full_cleaning_pipeline(train_raw)
test_clean  = full_cleaning_pipeline(test_raw)

# Feature preparation
X, y, X_test = prepare_features(train_clean, test_clean, target="Purchase")

# Create the base model
model_base, cv_scores, train_proba, scale_pos_weight, tscv = baseline_xgb(X, y)

# Tune the parameters
best_params, best_value, study = xgb_optuna_search(
    X, y, tscv, scale_pos_weight, n_trials=40
)

# Feature importance 
feat_imp = get_feature_importance(model_base, X)
best_features, feature_results = feature_subset_testing(
    X, y, feat_imp, tscv, scale_pos_weight
)

# Final model
final_model, test_pred, test_proba, thresh, f1 = train_final_model(
    X, y, X_test, best_features, best_params, scale_pos_weight
)

print("Final F1:", f1)
print("Optimal threshold:", thresh)

# ---------------------------
#      MARKETING REPORT
# ---------------------------

# Wrap the NumPy array into a DataFrame with IDs
test_predictions_df = pd.DataFrame({
    "id": test_clean["id"],       # ensure test_clean has the 'id' column
    "purchase_proba": test_proba  # NumPy array from the model
})

# Generate ROI / marketing report
report, top_users = generate_marketing_report(
    test_df=test_clean,       # your cleaned test dataframe
    pred_df=test_predictions_df,
    top_n=2000
)

# Print the marketing report
print("=== MARKETING REPORT ===")
for k, v in report.items():
    print(f"{k}: {v}")

# Save the top 2000 users
top_users.to_csv("top_2000_users.csv", index=False)
