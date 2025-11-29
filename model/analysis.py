import pandas as pd

def merge_predictions_with_test(test_df, pred_df):
    return test_df.merge(pred_df, on="id", how="left")


def select_top_buyers(df: pd.DataFrame, top_n: int = 2000):
    df["price_after_discount"] = df["Price"] * (1 - df["Discount"] * 0.01)
    df["expected_value"] = df["purchase_proba"] * df["price_after_discount"]
    return df.sort_values("expected_value", ascending=False).head(top_n)

def compute_expected_revenue(df_top: pd.DataFrame):
    return df_top["expected_value"].sum()

def generate_marketing_report(test_df, pred_df, top_n=2000, budget=200):
    merged = merge_predictions_with_test(test_df, pred_df)
    top_users = select_top_buyers(merged, top_n)

    # Compute expected revenue and ROI
    expected_revenue = compute_expected_revenue(top_users)
    roi = expected_revenue / budget

    report = {
        "targeted_users": top_n,
        "expected_revenue_€": expected_revenue,
        "ROI": roi,
        "expected_purchases": top_users["purchase_proba"].sum(),
    }

    return report, top_users
