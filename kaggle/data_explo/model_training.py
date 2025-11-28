import numpy as np
import pandas as pd
import time
import warnings
import optuna

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def load_and_process_data(train_path, test_path):
    df = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Interaction / ratio features
    for d in [df, df_test]:
        d['price_discount_ratio'] = d['Price'] / (1 + d['Discount'])
        d['engagement_x_socio'] = d['Engagement_Score'] * d['Socioeconomic_Status_Score']
        d['email_x_engagement'] = d['Email_Interaction'] * d['Engagement_Score']

    return df, df_test


def prepare_features(df, df_test, target="Purchase", id_cols=["id", "Session_ID"]):
    y = df[target]
    X = df.drop(columns=[target] + id_cols)
    X_test = df_test.drop(columns=id_cols)
    return X, y, X_test


def baseline_xgb(X, y, tscv_splits=5):
    tscv = TimeSeriesSplit(n_splits=tscv_splits)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    
    xgb_base = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    
    cv_scores = cross_val_score(
        xgb_base, X, y, cv=tscv, scoring="f1", n_jobs=-1
    )
    xgb_base.fit(X, y)
    train_proba = xgb_base.predict_proba(X)[:, 1]
    
    return xgb_base, cv_scores, train_proba, scale_pos_weight, tscv


def optimize_threshold(y_true, y_proba, start=0.30, stop=0.70, step=0.01):
    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(start, stop, step):
        f = f1_score(y_true, (y_proba >= t).astype(int))
        if f > best_f1:
            best_f1, best_thresh = f, t
    return best_thresh, best_f1


def xgb_optuna_search(X, y, tscv, scale_pos_weight, n_trials=50, save_path="xgb_optuna_results.csv"):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            "tree_method": "hist",
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring=make_scorer(f1_score))
        return scores.mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Save full trial history
    df_results = pd.DataFrame([t.params | {"value": t.value} for t in study.trials])
    df_results.to_csv(save_path, index=False)
    
    best_params = study.best_params
    best_value = study.best_value
    return best_params, best_value, study


def get_feature_importance(model, X, save_path="feature_importance.csv"):
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)
    feature_importance.to_csv(save_path, index=False)
    return feature_importance


def feature_subset_testing(X, y, feature_importance, tscv, scale_pos_weight, top_n_options=[20,30,40,50,60,80,100], save_path="feature_selection_results.csv"):
    results = []
    for n in top_n_options + [len(feature_importance)]:
        top_feats = feature_importance.head(n)["feature"].tolist()
        model = XGBClassifier(scale_pos_weight=scale_pos_weight, n_jobs=-1)
        scores = cross_val_score(model, X[top_feats], y, cv=tscv, scoring="f1", n_jobs=-1)
        results.append({"n_features": n, "f1_mean": scores.mean(), "f1_std": scores.std()})
    results_df = pd.DataFrame(results).sort_values("f1_mean", ascending=False)
    results_df.to_csv(save_path, index=False)
    best_n = int(results_df.iloc[0]["n_features"])
    best_features = feature_importance.head(best_n)["feature"].tolist()
    return best_features, results_df


def train_final_model(X_train, y_train, X_test, best_features, best_params, scale_pos_weight,
                      save_pred_csv="submission_final.csv", save_pred_with_prob_csv="submission_final_with_probabilities.csv"):
    X_train_sel = X_train[best_features]
    X_test_sel = X_test[best_features]
    
    final_model = XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight, n_jobs=-1)
    final_model.fit(X_train_sel, y_train)
    
    train_proba = final_model.predict_proba(X_train_sel)[:, 1]
    best_thresh, best_f1 = optimize_threshold(y_train, train_proba)
    
    test_proba = final_model.predict_proba(X_test_sel)[:, 1]
    test_pred = (test_proba >= best_thresh).astype(int)
    
    # Save predictions
    pd.DataFrame({"id": X_test.index, "Purchase": test_pred}).to_csv(save_pred_csv, index=False)
    pd.DataFrame({
        "id": X_test.index,
        "Purchase_Probability": test_proba,
        "Purchase_Prediction": test_pred
    }).to_csv(save_pred_with_prob_csv, index=False)
    
    return final_model, best_thresh, best_f1, test_pred, test_proba
