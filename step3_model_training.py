import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 4: MODEL SELECTION & TRAINING")
print("="*80)

# Load engineered datasets
X_train = pd.read_csv('X_train_engineered.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
X_test = pd.read_csv('X_test_engineered.csv')
train_ids = pd.read_csv('train_ids.csv').squeeze()
test_ids = pd.read_csv('test_ids.csv').squeeze()

print(f"\n📥 Loaded engineered datasets:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   Purchase rate: {y_train.mean():.2%}")

# ============================================================================
# PART 1: BASELINE MODEL - LOGISTIC REGRESSION
# ============================================================================
print("\n" + "="*80)
print("PART 1: BASELINE MODEL - LOGISTIC REGRESSION")
print("="*80)

# Temporal cross-validation setup (respect time ordering)
# We'll use 5 folds for time series split
tscv = TimeSeriesSplit(n_splits=5)

print("\n🔄 Using Temporal Cross-Validation (5 folds)")
print("   This respects the time-based nature of the data")

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\n⚖️  Class weights: {class_weight_dict}")

# Train Logistic Regression
print("\n🔨 Training Logistic Regression (baseline)...")
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

# Cross-validation with F1 score
lr_cv_scores = cross_val_score(
    lr_model, X_train, y_train, 
    cv=tscv, 
    scoring='f1',
    n_jobs=-1
)

print(f"   CV F1 Scores: {lr_cv_scores}")
print(f"   Mean F1: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std():.4f})")

# Train on full training set
lr_model.fit(X_train, y_train)

# Predictions
y_train_pred_lr = lr_model.predict(X_train)
y_train_proba_lr = lr_model.predict_proba(X_train)[:, 1]

# Evaluation
train_f1_lr = f1_score(y_train, y_train_pred_lr)
train_auc_lr = roc_auc_score(y_train, y_train_proba_lr)

print(f"\n📊 Logistic Regression Results:")
print(f"   Training F1 Score: {train_f1_lr:.4f}")
print(f"   Training AUC-ROC: {train_auc_lr:.4f}")
print(f"   Mean CV F1: {lr_cv_scores.mean():.4f}")

# ============================================================================
# PART 2: RANDOM FOREST
# ============================================================================
print("\n" + "="*80)
print("PART 2: RANDOM FOREST")
print("="*80)

print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# Cross-validation
rf_cv_scores = cross_val_score(
    rf_model, X_train, y_train,
    cv=tscv,
    scoring='f1',
    n_jobs=-1
)

print(f"   CV F1 Scores: {rf_cv_scores}")
print(f"   Mean F1: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

# Train on full training set
rf_model.fit(X_train, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train)
y_train_proba_rf = rf_model.predict_proba(X_train)[:, 1]

# Evaluation
train_f1_rf = f1_score(y_train, y_train_pred_rf)
train_auc_rf = roc_auc_score(y_train, y_train_proba_rf)

print(f"\n📊 Random Forest Results:")
print(f"   Training F1 Score: {train_f1_rf:.4f}")
print(f"   Training AUC-ROC: {train_auc_rf:.4f}")
print(f"   Mean CV F1: {rf_cv_scores.mean():.4f}")

# ============================================================================
# PART 3: XGBOOST
# ============================================================================
print("\n" + "="*80)
print("PART 3: XGBOOST")
print("="*80)

print("\n🚀 Training XGBoost...")
# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"   Scale pos weight: {scale_pos_weight:.2f}")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# Cross-validation
xgb_cv_scores = cross_val_score(
    xgb_model, X_train, y_train,
    cv=tscv,
    scoring='f1',
    n_jobs=-1
)

print(f"   CV F1 Scores: {xgb_cv_scores}")
print(f"   Mean F1: {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std():.4f})")

# Train on full training set
xgb_model.fit(X_train, y_train)

# Predictions
y_train_pred_xgb = xgb_model.predict(X_train)
y_train_proba_xgb = xgb_model.predict_proba(X_train)[:, 1]

# Evaluation
train_f1_xgb = f1_score(y_train, y_train_pred_xgb)
train_auc_xgb = roc_auc_score(y_train, y_train_proba_xgb)

print(f"\n📊 XGBoost Results:")
print(f"   Training F1 Score: {train_f1_xgb:.4f}")
print(f"   Training AUC-ROC: {train_auc_xgb:.4f}")
print(f"   Mean CV F1: {xgb_cv_scores.mean():.4f}")

# ============================================================================
# PART 4: LIGHTGBM
# ============================================================================
print("\n" + "="*80)
print("PART 4: LIGHTGBM")
print("="*80)

print("\n⚡ Training LightGBM...")
lgbm_model = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Cross-validation
lgbm_cv_scores = cross_val_score(
    lgbm_model, X_train, y_train,
    cv=tscv,
    scoring='f1',
    n_jobs=-1
)

print(f"   CV F1 Scores: {lgbm_cv_scores}")
print(f"   Mean F1: {lgbm_cv_scores.mean():.4f} (+/- {lgbm_cv_scores.std():.4f})")

# Train on full training set
lgbm_model.fit(X_train, y_train)

# Predictions
y_train_pred_lgbm = lgbm_model.predict(X_train)
y_train_proba_lgbm = lgbm_model.predict_proba(X_train)[:, 1]

# Evaluation
train_f1_lgbm = f1_score(y_train, y_train_pred_lgbm)
train_auc_lgbm = roc_auc_score(y_train, y_train_proba_lgbm)

print(f"\n📊 LightGBM Results:")
print(f"   Training F1 Score: {train_f1_lgbm:.4f}")
print(f"   Training AUC-ROC: {train_auc_lgbm:.4f}")
print(f"   Mean CV F1: {lgbm_cv_scores.mean():.4f}")

# ============================================================================
# PART 5: MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PART 5: MODEL COMPARISON")
print("="*80)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM'],
    'CV_F1_Mean': [lr_cv_scores.mean(), rf_cv_scores.mean(), xgb_cv_scores.mean(), lgbm_cv_scores.mean()],
    'CV_F1_Std': [lr_cv_scores.std(), rf_cv_scores.std(), xgb_cv_scores.std(), lgbm_cv_scores.std()],
    'Train_F1': [train_f1_lr, train_f1_rf, train_f1_xgb, train_f1_lgbm],
    'Train_AUC': [train_auc_lr, train_auc_rf, train_auc_xgb, train_auc_lgbm]
})

results = results.sort_values('CV_F1_Mean', ascending=False).reset_index(drop=True)

print("\n📊 Model Comparison (sorted by CV F1 Score):")
print(results.to_string(index=False))

# Select best model based on CV F1
best_model_name = results.iloc[0]['Model']
best_cv_f1 = results.iloc[0]['CV_F1_Mean']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   CV F1 Score: {best_cv_f1:.4f}")

# Get the best model object
model_map = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgbm_model
}
best_model = model_map[best_model_name]

# ============================================================================
# PART 6: THRESHOLD OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("PART 6: THRESHOLD OPTIMIZATION FOR BEST MODEL")
print("="*80)

# Get probabilities from best model
if best_model_name == 'Logistic Regression':
    y_train_proba_best = y_train_proba_lr
elif best_model_name == 'Random Forest':
    y_train_proba_best = y_train_proba_rf
elif best_model_name == 'XGBoost':
    y_train_proba_best = y_train_proba_xgb
else:
    y_train_proba_best = y_train_proba_lgbm

# Try different thresholds
print("\n🎯 Testing different probability thresholds:")
thresholds = np.arange(0.3, 0.7, 0.05)
threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_train_proba_best >= threshold).astype(int)
    f1 = f1_score(y_train, y_pred_threshold)
    threshold_results.append({'threshold': threshold, 'f1_score': f1})
    print(f"   Threshold {threshold:.2f}: F1 = {f1:.4f}")

threshold_df = pd.DataFrame(threshold_results)
best_threshold = threshold_df.loc[threshold_df['f1_score'].idxmax(), 'threshold']
best_threshold_f1 = threshold_df['f1_score'].max()

print(f"\n🎯 Optimal Threshold: {best_threshold:.2f}")
print(f"   F1 Score at optimal threshold: {best_threshold_f1:.4f}")

# ============================================================================
# PART 7: FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n" + "="*80)
print("PART 7: FEATURE IMPORTANCE")
print("="*80)

if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    print(f"\n📊 Top 20 Most Important Features ({best_model_name}):")
    
    feature_names = X_train.columns
    if best_model_name == 'Random Forest':
        importances = rf_model.feature_importances_
    elif best_model_name == 'XGBoost':
        importances = xgb_model.feature_importances_
    else:
        importances = lgbm_model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print(feature_importance.head(20).to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(f"\n   💾 Saved to feature_importance.csv")

# ============================================================================
# PART 8: PREDICTIONS ON TEST SET
# ============================================================================
print("\n" + "="*80)
print("PART 8: PREDICTIONS ON TEST SET")
print("="*80)

print(f"\n🔮 Making predictions with {best_model_name}...")

# Get predictions with optimal threshold
test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

print(f"   Test predictions completed")
print(f"   Predicted positives: {test_pred.sum()} ({test_pred.mean():.2%})")
print(f"   Predicted negatives: {(1-test_pred).sum()} ({(1-test_pred).mean():.2%})")

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Purchase': test_pred
})

submission.to_csv('submission.csv', index=False)
print(f"\n✅ Submission file created: submission.csv")

# Also save probabilities for analysis
submission_proba = pd.DataFrame({
    'id': test_ids,
    'Purchase_Probability': test_proba,
    'Purchase_Prediction': test_pred
})
submission_proba.to_csv('submission_with_probabilities.csv', index=False)
print(f"✅ Detailed predictions saved: submission_with_probabilities.csv")

# ============================================================================
# PART 9: FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Cross-Validation F1: {best_cv_f1:.4f}")
print(f"   Training F1: {results[results['Model'] == best_model_name]['Train_F1'].values[0]:.4f}")
print(f"   Optimal Threshold: {best_threshold:.2f}")
print(f"   F1 at Optimal Threshold: {best_threshold_f1:.4f}")

print(f"\n📊 Test Set Predictions:")
print(f"   Total predictions: {len(test_pred)}")
print(f"   Predicted purchases: {test_pred.sum()} ({test_pred.mean():.2%})")
print(f"   Mean probability: {test_proba.mean():.4f}")

print("\n📁 Files Created:")
print("   ✓ submission.csv - Ready for Kaggle submission")
print("   ✓ submission_with_probabilities.csv - Detailed predictions")
if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    print("   ✓ feature_importance.csv - Feature importance rankings")

print("\n" + "="*80)
print("STEP 4: MODEL TRAINING COMPLETE!")
print("="*80)
print("\n🎯 Next: Review results and submit to Kaggle!")