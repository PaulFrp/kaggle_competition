import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 7: FEATURE SELECTION FOR IMPROVED GENERALIZATION")
print("="*80)

# Load data
X_train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/X_train_engineered.csv')
y_train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/y_train.csv').squeeze()
X_test = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/X_test_engineered.csv')
test_ids = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/test_ids.csv').squeeze()

print(f"\n📥 Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
print(f"\n📊 Status: CV 0.8006 | Kaggle 0.7828 | Gap -0.0178 (overfitting!)")

feature_importance = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/feature_importance.csv')

# ============================================================================
# PART 1: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 1: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
feature_importance['cumulative_pct'] = (
    feature_importance['cumulative_importance'] / feature_importance['importance'].sum() * 100
)

print(f"\n📊 Importance Coverage:")
for pct in [80, 85, 90, 95]:
    n = (feature_importance['cumulative_pct'] <= pct).sum()
    print(f"   Top {n} features = {pct}% importance")

print(f"\n🔝 Top 15 Features:")
for i in range(15):
    row = feature_importance.iloc[i]
    print(f"   {i+1:2d}. {row['feature']:30s} {row['importance']:6.0f}")

# ============================================================================
# PART 2: TEST FEATURE SUBSETS
# ============================================================================
print("\n" + "="*80)
print("PART 2: TESTING FEATURE SUBSETS")
print("="*80)

tscv = TimeSeriesSplit(n_splits=5)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

best_xgb_params = {
    'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.08,
    'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 3,
    'scale_pos_weight': scale_pos_weight, 'random_state': 42,
    'n_jobs': -1, 'eval_metric': 'logloss'
}

feature_counts = [20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 108]

print(f"\n🔬 Testing {len(feature_counts)} configurations with 5-fold CV...\n")

results = []
for n in feature_counts:
    print(f"Testing {n:3d} features...", end=" ")
    top_feats = feature_importance.head(n)['feature'].tolist()
    X_subset = X_train[top_feats]
    
    model = XGBClassifier(**best_xgb_params)
    cv_scores = cross_val_score(model, X_subset, y_train, cv=tscv, scoring='f1', n_jobs=-1)
    
    results.append({
        'n_features': n,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std()
    })
    print(f"CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

results_df = pd.DataFrame(results).sort_values('cv_f1_mean', ascending=False)

print("\n" + "="*80)
print("RESULTS (Sorted by CV F1)")
print("="*80)
print(results_df.to_string(index=False))

best = results_df.iloc[0]
best_n = int(best['n_features'])
best_cv = best['cv_f1_mean']
baseline_cv = results_df[results_df['n_features']==108]['cv_f1_mean'].values[0]

print(f"\n🏆 BEST: {best_n} features | CV F1: {best_cv:.4f}")
print(f"   vs 108 features: {best_cv - baseline_cv:+.4f} ({(best_cv-baseline_cv)/baseline_cv*100:+.2f}%)")

# ============================================================================
# PART 3: PROPER THRESHOLD OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("PART 3: THRESHOLD OPTIMIZATION (ON VALIDATION)")
print("="*80)

best_features = feature_importance.head(best_n)['feature'].tolist()
X_train_best = X_train[best_features]
X_test_best = X_test[best_features]

# Train/val split
split = int(len(X_train_best) * 0.8)
X_tr, X_val = X_train_best.iloc[:split], X_train_best.iloc[split:]
y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

print(f"\n   Train: {len(X_tr)} | Val: {len(X_val)}")

val_model = XGBClassifier(**best_xgb_params)
val_model.fit(X_tr, y_tr)
val_proba = val_model.predict_proba(X_val)[:, 1]

best_f1, best_thresh = 0, 0.5
for t in np.arange(0.30, 0.71, 0.01):
    f1 = f1_score(y_val, (val_proba >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"\n🏆 Optimal threshold: {best_thresh:.2f} (Val F1: {best_f1:.4f})")

# ============================================================================
# PART 4: FINAL MODEL & PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("PART 4: FINAL MODEL & PREDICTIONS")
print("="*80)

final_model = XGBClassifier(**best_xgb_params)
final_model.fit(X_train_best, y_train)

test_proba = final_model.predict_proba(X_test_best)[:, 1]
test_pred = (test_proba >= best_thresh).astype(int)

print(f"\n📊 Test Predictions:")
print(f"   Features: {best_n} | Threshold: {best_thresh:.2f}")
print(f"   Purchases: {test_pred.sum()} ({test_pred.mean():.2%})")
print(f"   vs Previous: {test_pred.sum() - 2544:+d} predictions")

# ============================================================================
# PART 5: SAVE
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

pd.DataFrame({'id': test_ids, 'Purchase': test_pred}).to_csv(
    'submission_feature_selected.csv', index=False)
pd.DataFrame({'id': test_ids, 'Purchase_Probability': test_proba, 
              'Purchase_Prediction': test_pred}).to_csv(
    'submission_feature_selected_with_probs.csv', index=False)
pd.DataFrame({'feature': best_features}).to_csv('selected_features.csv', index=False)
results_df.to_csv('feature_selection_results.csv', index=False)

print("\n✅ Saved:")
print("   📁 submission_feature_selected.csv")
print("   📁 submission_feature_selected_with_probs.csv")
print("   📁 selected_features.csv")
print("   📁 feature_selection_results.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPLETE!")
print("="*80)

print(f"\n🏆 Configuration: {best_n} features | Threshold {best_thresh:.2f}")
print(f"   CV F1: {best_cv:.4f} | Improvement: {best_cv-baseline_cv:+.4f}")

expected_kaggle = 0.7828 + (best_cv - baseline_cv)
print(f"\n📊 Expected Kaggle: ~{expected_kaggle:.4f}")
if expected_kaggle > 0.798:
    print(f"   ✅ SHOULD BEAT LEADER (0.79780)! 🎉")
else:
    print(f"   ⚠️  Gap: {0.79780 - expected_kaggle:.4f} | Try stacking next")

print(f"\n🎯 Submit: submission_feature_selected.csv")
print("="*80)