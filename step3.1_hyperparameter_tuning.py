import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
import time

print("="*80)
print("STEP 6: HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*80)

# Load data
X_train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/X_train_engineered.csv')
y_train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/y_train.csv').squeeze()
X_test = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/X_test_engineered.csv')
train_ids = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/train_ids.csv').squeeze()
test_ids = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/test_ids.csv').squeeze()

print(f"\n📥 Loaded datasets: X_train {X_train.shape}, X_test {X_test.shape}")

tscv = TimeSeriesSplit(n_splits=5)
f1_scorer = make_scorer(f1_score)

# ============================================================================
# PART 1: LIGHTGBM HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("PART 1: LIGHTGBM TUNING")
print("="*80)

print("\n📋 Baseline: CV F1=0.7944")
print("   Params: n_estimators=200, max_depth=6, learning_rate=0.1, num_leaves=31")

# Focused grid (most impactful parameters)
lgbm_param_grid = {
    'n_estimators': [200, 250, 300],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.08, 0.1, 0.12],
    'num_leaves': [31, 40, 50],
    'min_child_samples': [15, 20, 25]
}

print(f"\n🔧 Focused parameter grid (most impactful):")
for param, values in lgbm_param_grid.items():
    print(f"   {param:20s}: {values}")

n_comb = np.prod([len(v) for v in lgbm_param_grid.values()])
print(f"\n📊 Combinations: {n_comb} × 5 folds = {n_comb*5} fits")
print(f"   Estimated time: ~{n_comb*5*2/60:.0f} minutes")

lgbm_base = LGBMClassifier(
    class_weight='balanced',
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print(f"\n⏳ Running GridSearchCV for LightGBM...")
start = time.time()

grid_lgbm = GridSearchCV(
    lgbm_base, lgbm_param_grid,
    cv=tscv, scoring=f1_scorer,
    n_jobs=-1, verbose=2
)
grid_lgbm.fit(X_train, y_train)

elapsed = time.time() - start
print(f"\n✅ LightGBM tuning completed in {elapsed/60:.1f} minutes")

print(f"\n🏆 Best LightGBM Parameters:")
for param, value in grid_lgbm.best_params_.items():
    print(f"   {param:20s}: {value}")

lgbm_best_score = grid_lgbm.best_score_
improvement_lgbm = lgbm_best_score - 0.7944

print(f"\n📊 Performance:")
print(f"   Best CV F1: {lgbm_best_score:.4f}")
print(f"   Baseline F1: 0.7944")
print(f"   Improvement: {improvement_lgbm:+.4f} ({improvement_lgbm/0.7944*100:+.2f}%)")

best_lgbm = grid_lgbm.best_estimator_

# Save detailed results
lgbm_results = pd.DataFrame(grid_lgbm.cv_results_)
lgbm_results = lgbm_results.sort_values('rank_test_score')
lgbm_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].head(10).to_csv(
    'lgbm_top10_results.csv', index=False
)
print(f"\n💾 Saved top 10 configurations: lgbm_top10_results.csv")

# ============================================================================
# PART 2: XGBOOST HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("PART 2: XGBOOST TUNING")
print("="*80)

print("\n📋 Baseline: CV F1=0.7917")
print("   Params: n_estimators=200, max_depth=6, learning_rate=0.1")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

# Focused grid
xgb_param_grid = {
    'n_estimators': [200, 250, 300],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.08, 0.1, 0.12],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}

print(f"\n🔧 Focused parameter grid:")
for param, values in xgb_param_grid.items():
    print(f"   {param:20s}: {values}")

n_comb = np.prod([len(v) for v in xgb_param_grid.values()])
print(f"\n📊 Combinations: {n_comb} × 5 folds = {n_comb*5} fits")
print(f"   Estimated time: ~{n_comb*5*2/60:.0f} minutes")

xgb_base = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

print(f"\n⏳ Running GridSearchCV for XGBoost...")
start = time.time()

grid_xgb = GridSearchCV(
    xgb_base, xgb_param_grid,
    cv=tscv, scoring=f1_scorer,
    n_jobs=-1, verbose=2
)
grid_xgb.fit(X_train, y_train)

elapsed = time.time() - start
print(f"\n✅ XGBoost tuning completed in {elapsed/60:.1f} minutes")

print(f"\n🏆 Best XGBoost Parameters:")
for param, value in grid_xgb.best_params_.items():
    print(f"   {param:20s}: {value}")

xgb_best_score = grid_xgb.best_score_
improvement_xgb = xgb_best_score - 0.7917

print(f"\n📊 Performance:")
print(f"   Best CV F1: {xgb_best_score:.4f}")
print(f"   Baseline F1: 0.7917")
print(f"   Improvement: {improvement_xgb:+.4f} ({improvement_xgb/0.7917*100:+.2f}%)")

best_xgb = grid_xgb.best_estimator_

# Save detailed results
xgb_results = pd.DataFrame(grid_xgb.cv_results_)
xgb_results = xgb_results.sort_values('rank_test_score')
xgb_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].head(10).to_csv(
    'xgb_top10_results.csv', index=False
)
print(f"\n💾 Saved top 10 configurations: xgb_top10_results.csv")

# ============================================================================
# PART 3: COMPARISON & MODEL SELECTION
# ============================================================================
print("\n" + "="*80)
print("PART 3: FINAL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Model': [
        'LightGBM (Baseline)',
        'LightGBM (Tuned)',
        'XGBoost (Baseline)',
        'XGBoost (Tuned)'
    ],
    'CV_F1': [
        0.7944,
        lgbm_best_score,
        0.7917,
        xgb_best_score
    ],
    'Improvement': [
        0.0000,
        improvement_lgbm,
        0.0000,
        improvement_xgb
    ],
    'Improvement_Pct': [
        0.00,
        improvement_lgbm/0.7944*100,
        0.00,
        improvement_xgb/0.7917*100
    ]
})

comparison = comparison.sort_values('CV_F1', ascending=False).reset_index(drop=True)

print("\n📊 Complete Model Comparison:")
print(comparison.to_string(index=False))

# Select winner
best_model_name = comparison.iloc[0]['Model']
best_cv_f1 = comparison.iloc[0]['CV_F1']

print(f"\n🏆 WINNER: {best_model_name}")
print(f"   CV F1 Score: {best_cv_f1:.4f}")

# Determine which model to use
if 'LightGBM (Tuned)' in best_model_name:
    best_model = best_lgbm
    print(f"   Using tuned LightGBM")
elif 'XGBoost (Tuned)' in best_model_name:
    best_model = best_xgb
    print(f"   Using tuned XGBoost")
elif best_cv_f1 == 0.7944:
    print(f"   Note: Tuning did not improve. Using baseline LightGBM.")
    best_model = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
    )
    best_model.fit(X_train, y_train)
else:
    print(f"   Note: Tuning did not improve. Using baseline XGBoost.")
    best_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1
    )
    best_model.fit(X_train, y_train)

# ============================================================================
# PART 4: THRESHOLD OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("PART 4: THRESHOLD OPTIMIZATION")
print("="*80)

print(f"\n🎯 Finding optimal threshold for {best_model_name}...")

train_proba = best_model.predict_proba(X_train)[:, 1]

best_f1_val = 0
best_thresh = 0.5

for threshold in np.arange(0.35, 0.70, 0.01):
    pred = (train_proba >= threshold).astype(int)
    f1_val = f1_score(y_train, pred)
    if f1_val > best_f1_val:
        best_f1_val = f1_val
        best_thresh = threshold

print(f"\n🏆 Optimal threshold: {best_thresh:.2f}")
print(f"   Training F1 at threshold: {best_f1_val:.4f}")

# Show key thresholds
print(f"\n📊 F1 scores at different thresholds:")
for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    pred = (train_proba >= t).astype(int)
    f1_val = f1_score(y_train, pred)
    marker = " ← OPTIMAL" if abs(t - best_thresh) < 0.01 else ""
    print(f"   Threshold {t:.2f}: F1 = {f1_val:.4f}{marker}")

# ============================================================================
# PART 5: GENERATE TEST PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("PART 5: TEST PREDICTIONS")
print("="*80)

print(f"\n🔮 Generating predictions...")

test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_thresh).astype(int)

print(f"\n📊 Test Predictions:")
print(f"   Threshold used: {best_thresh:.2f}")
print(f"   Predicted purchases: {test_pred.sum()} ({test_pred.mean():.2%})")
print(f"   Mean probability: {test_proba.mean():.4f}")
print(f"   Median probability: {np.median(test_proba):.4f}")
print(f"   Min probability: {test_proba.min():.4f}")
print(f"   Max probability: {test_proba.max():.4f}")

print(f"\n📊 Comparison to Original (Step 4):")
print(f"   Original: 2,325 purchases (37.11%)")
print(f"   Tuned: {test_pred.sum()} purchases ({test_pred.mean():.2%})")
print(f"   Difference: {test_pred.sum() - 2325:+d} predictions")

# ============================================================================
# PART 6: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("PART 6: SAVING RESULTS")
print("="*80)

# Save submission
submission = pd.DataFrame({
    'id': test_ids,
    'Purchase': test_pred
})
submission.to_csv('submission_tuned.csv', index=False)

# Save probabilities
submission_proba = pd.DataFrame({
    'id': test_ids,
    'Purchase_Probability': test_proba,
    'Purchase_Prediction': test_pred
})
submission_proba.to_csv('submission_tuned_with_probabilities.csv', index=False)

# Save summary
summary_data = {
    'best_model': best_model_name,
    'cv_f1_score': best_cv_f1,
    'baseline_best_f1': 0.7944,
    'improvement': best_cv_f1 - 0.7944,
    'improvement_pct': (best_cv_f1 - 0.7944) / 0.7944 * 100,
    'optimal_threshold': best_thresh,
    'predicted_purchases': int(test_pred.sum()),
    'predicted_rate': float(test_pred.mean())
}

if 'Tuned' in best_model_name:
    if 'LightGBM' in best_model_name:
        summary_data.update(grid_lgbm.best_params_)
    else:
        summary_data.update(grid_xgb.best_params_)

summary_df = pd.DataFrame([summary_data])
summary_df.to_csv('tuning_summary.csv', index=False)

print("\n✅ Files saved:")
print("   📁 submission_tuned.csv - For Kaggle submission")
print("   📁 submission_tuned_with_probabilities.csv - With probabilities")
print("   📁 lgbm_top10_results.csv - Top 10 LightGBM configs")
print("   📁 xgb_top10_results.csv - Top 10 XGBoost configs")
print("   📁 tuning_summary.csv - Complete summary")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("HYPERPARAMETER TUNING COMPLETE!")
print("="*80)

print(f"\n🏆 FINAL RESULTS:")
print(f"   Best Model: {best_model_name}")
print(f"   CV F1 Score: {best_cv_f1:.4f}")
print(f"   Baseline F1: 0.7944 (LightGBM Step 4)")
print(f"   Overall Improvement: {best_cv_f1 - 0.7944:+.4f} ({(best_cv_f1-0.7944)/0.7944*100:+.2f}%)")
print(f"   Optimal Threshold: {best_thresh:.2f}")

if 'Tuned' in best_model_name:
    print(f"\n🔧 Tuned Parameters:")
    if 'LightGBM' in best_model_name:
        for param, value in grid_lgbm.best_params_.items():
            print(f"   {param:20s}: {value}")
    else:
        for param, value in grid_xgb.best_params_.items():
            print(f"   {param:20s}: {value}")

print(f"\n📊 Test Predictions:")
print(f"   Predicted purchases: {test_pred.sum()} ({test_pred.mean():.2%})")

if best_cv_f1 > 0.7950:  # Meaningful improvement
    print(f"\n✅ TUNING SUCCESSFUL! Improvement achieved.")
    print(f"   → Submit: submission_tuned.csv")
    print(f"   → Expected F1: ~{best_cv_f1:.2f}")
elif best_cv_f1 > 0.7944:  # Small improvement
    print(f"\n✅ SLIGHT IMPROVEMENT from tuning")
    print(f"   → Can use either submission_tuned.csv or original")
    print(f"   → Expected F1: ~{best_cv_f1:.2f}")
else:
    print(f"\n⚠️  Tuning did not improve performance")
    print(f"   → Original submission.csv (F1=0.7944) is still best")
    print(f"   → But submission_tuned.csv is also competitive")

print("\n" + "="*80)