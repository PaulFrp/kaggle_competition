import pandas as pd

def merge_predictions_with_test(test_df, pred_df):
    """
    Merge predictions with the original test dataframe using ID.
    """
    return test_df.merge(pred_df, on="id", how="left")


def select_top_buyers(df: pd.DataFrame, prob_col: str, top_n: int = 2000):
    """
    Select the top N users with highest purchase probability.
    """
    return df.sort_values(prob_col, ascending=False).head(top_n)


def compute_expected_revenue(df_top: pd.DataFrame):
    """
    Expected revenue = sum(probability * (price - discount)).
    """
    df_top["price_after_discount"] = df_top["Price"] * (1 - df_top["Discount"]* 0.01)
    df_top["expected_revenue"] = df_top["price_after_discount"]
    return df_top["expected_revenue"].sum()


def generate_marketing_report(test_df, pred_df, top_n=2000):
    # merge price + discount with predicted probabilities
    merged = merge_predictions_with_test(test_df, pred_df)

    # select top probable buyers
    top_users = select_top_buyers(merged, "purchase_proba", top_n)

    # compute expected revenue
    expected_revenue = compute_expected_revenue(top_users)

    return {
        "targeted_users": top_n,
        "expected_revenue_€": expected_revenue,
        "ROI": expected_revenue / 200,  # if 200€ budget
        "expected_purchases": top_users["purchase_proba"].sum(),
    }, top_users
