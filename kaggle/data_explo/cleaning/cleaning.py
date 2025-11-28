import numpy as np

def clean_categorical(df):
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
        df[col] = df[col].replace("nan", np.nan)
        col_lower = col.lower()

        if col_lower == "time_of_day":
            df[col] = df[col].replace({
                r'^m.*':'morning',
                r'^a.*':'afternoon',
                r'^e.*':'evening',
                r'^m[0o].*rning$': 'morning',
                r'^aftern?[0o].*n$': 'afternoon',
                r'^even.*g$': 'evening',
                }, regex=True)

        elif col_lower == 'payment_method':
            df[col] = df[col].replace({
                r'^cred.*$': 'credit',
                r'^cash$': 'cash',
                r'^pay[\s_]?pal$': 'paypal',
                r'^bank.*$': 'bank',
            }, regex=True)

        # Normalize Referral_Source
        elif col_lower == 'referral_source':
            df[col] = df[col].replace({
                r'^s[0o].*cial.*media$': 'social_media',
                r'^search.*engine$': 'search_engine',
                r'^ads$': 'ads',
                r'^email$': 'email',
                r'^direct$': 'direct',
            }, regex=True)
    return df
