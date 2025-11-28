from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def cat_encoding(df):
    
    #Encode categorical variable to allow knn imputation. 
    cat_columns = df.select_dtypes(exclude=["number"]).columns
    cat_columns = cat_columns.drop("Session_ID")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_columns] = cat_imputer.fit_transform(df[cat_columns])
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[cat_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat_columns), index=df.index)
    df_sklearn_encoded = pd.concat([df.drop(columns = cat_columns), one_hot_df], axis=1)
    
    return df_sklearn_encoded
