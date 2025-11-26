import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 2: DATA CLEANING & PREPROCESSING")
print("="*80)

# Load datasets
train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/train_dataset_M1_with_id.csv')
test = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/test_dataset_M1_with_id.csv')

print(f"\nLoaded datasets:")
print(f"   Train: {len(train)} rows")
print(f"   Test: {len(test)} rows")

# Store original sizes
original_train_size = len(train)
original_test_size = len(test)

# ============================================================================
# PART 1: STANDARDIZE CATEGORICAL VARIABLES
# ============================================================================
print("\n" + "="*80)
print("PART 1: STANDARDIZING CATEGORICAL VARIABLES")
print("="*80)

def clean_time_of_day(value):
    """Standardize Time_of_Day to: morning, afternoon, evening"""
    if pd.isna(value):
        return np.nan
    value_lower = str(value).lower().strip()
    # Remove numbers and special chars
    value_clean = re.sub(r'[0-9]', '', value_lower)
    
    if 'morning' in value_clean or 'morn' in value_clean:
        return 'morning'
    elif 'afternoon' in value_clean or 'aftern' in value_clean:
        return 'afternoon'
    elif 'evening' in value_clean or 'even' in value_clean:
        return 'evening'
    else:
        return np.nan

def clean_payment_method(value):
    """Standardize Payment_Method to: Bank, Cash, Credit, PayPal"""
    if pd.isna(value):
        return np.nan
    value_lower = str(value).lower().strip()
    value_lower = value_lower.replace(' ', '').replace('_', '')
    
    if 'paypal' in value_lower or 'pay' in value_lower:
        return 'PayPal'
    elif 'credit' in value_lower:
        return 'Credit'
    elif 'cash' in value_lower:
        return 'Cash'
    elif 'bank' in value_lower:
        return 'Bank'
    else:
        return np.nan

def clean_referral_source(value):
    """Standardize Referral_Source to: Search_engine, Direct, Ads, Social_media, Email"""
    if pd.isna(value):
        return np.nan
    value_lower = str(value).lower().strip()
    value_lower = value_lower.replace(' ', '').replace('_', '').replace('-', '')
    
    if 'search' in value_lower or 'engine' in value_lower:
        return 'Search_engine'
    elif 'direct' in value_lower:
        return 'Direct'
    elif 'ad' in value_lower:
        return 'Ads'
    elif 'social' in value_lower or 'media' in value_lower:
        return 'Social_media'
    elif 'email' in value_lower or 'mail' in value_lower:
        return 'Email'
    else:
        return np.nan

# Apply cleaning to both datasets
print("\nCleaning Time_of_Day...")
train['Time_of_Day'] = train['Time_of_Day'].apply(clean_time_of_day)
test['Time_of_Day'] = test['Time_of_Day'].apply(clean_time_of_day)
print(f"   Train unique values: {train['Time_of_Day'].unique()}")
print(f"   Test unique values: {test['Time_of_Day'].unique()}")
print(f"   Train value counts:\n{train['Time_of_Day'].value_counts()}")

print("\nCleaning Payment_Method...")
train['Payment_Method'] = train['Payment_Method'].apply(clean_payment_method)
test['Payment_Method'] = test['Payment_Method'].apply(clean_payment_method)
print(f"   Train unique values: {train['Payment_Method'].unique()}")
print(f"   Test unique values: {test['Payment_Method'].unique()}")
print(f"   Train value counts:\n{train['Payment_Method'].value_counts()}")

print("\nCleaning Referral_Source...")
train['Referral_Source'] = train['Referral_Source'].apply(clean_referral_source)
test['Referral_Source'] = test['Referral_Source'].apply(clean_referral_source)
print(f"   Train unique values: {train['Referral_Source'].unique()}")
print(f"   Test unique values: {test['Referral_Source'].unique()}")
print(f"   Train value counts:\n{train['Referral_Source'].value_counts()}")

print("\nCleaning Device_Type (removing extra spaces)...")
train['Device_Type'] = train['Device_Type'].str.strip() if train['Device_Type'].dtype == 'object' else train['Device_Type']
test['Device_Type'] = test['Device_Type'].str.strip() if test['Device_Type'].dtype == 'object' else test['Device_Type']

# ============================================================================
# PART 2: HANDLE MISSING VALUES
# ============================================================================
print("\n" + "="*80)
print("PART 2: HANDLING MISSING VALUES")
print("="*80)

print("\nMissing values before handling:")
print("\nTrain:")
print(train.isnull().sum()[train.isnull().sum() > 0].sort_values(ascending=False))
print("\nTest:")
print(test.isnull().sum()[test.isnull().sum() > 0].sort_values(ascending=False))

# Strategy:
# 1. Categorical: Fill with 'Unknown' or mode
# 2. Numerical: Use median or KNN imputation
# 3. Session_ID, Day: Special handling

# Fill Day missing values in test (forward fill based on id order)
print("\nHandling missing 'Day' in test set...")
if test['Day'].isnull().any():
    # For test set, missing days are likely sequential
    # Fill with median day from test set
    test['Day'] = test['Day'].fillna(test['Day'].median())
    print(f"   Filled {test['Day'].isnull().sum()} missing Day values with median: {test['Day'].median()}")

# Fill Session_ID missing values - these might be data errors, we'll create placeholder IDs
print("\nHandling missing Session_IDs...")
train_missing_sid = train['Session_ID'].isnull().sum()
test_missing_sid = test['Session_ID'].isnull().sum()
if train_missing_sid > 0:
    train.loc[train['Session_ID'].isnull(), 'Session_ID'] = [f"MISSING_TRAIN_{i}" for i in range(train_missing_sid)]
    print(f"   Created {train_missing_sid} placeholder Session_IDs in train")
if test_missing_sid > 0:
    test.loc[test['Session_ID'].isnull(), 'Session_ID'] = [f"MISSING_TEST_{i}" for i in range(test_missing_sid)]
    print(f"   Created {test_missing_sid} placeholder Session_IDs in test")

# Categorical features: Fill with 'Unknown'
print("\nFilling categorical missing values with 'Unknown'...")
categorical_cols = ['Time_of_Day', 'Device_Type', 'Payment_Method', 'Referral_Source', 'PM_RS_Combo']
for col in categorical_cols:
    if col in train.columns:
        train_missing = train[col].isnull().sum()
        test_missing = test[col].isnull().sum()
        train[col] = train[col].fillna('Unknown')
        test[col] = test[col].fillna('Unknown')
        if train_missing > 0 or test_missing > 0:
            print(f"   {col}: filled {train_missing} (train), {test_missing} (test)")

# Campaign_Period: Fill with False (assume not in campaign if missing)
print("\nFilling Campaign_Period missing values...")
train['Campaign_Period'] = train['Campaign_Period'].fillna(False)
test['Campaign_Period'] = test['Campaign_Period'].fillna(False)
# Convert to boolean properly
train['Campaign_Period'] = train['Campaign_Period'].map({'True': True, 'False': False, True: True, False: False})
test['Campaign_Period'] = test['Campaign_Period'].map({'True': True, 'False': False, True: True, False: False})

# Numerical features: Median imputation for simple features
print("\nFilling numerical missing values with median...")
simple_numerical = ['Gender', 'Discount', 'Category', 'Email_Interaction', 'AB_Bucket']
for col in simple_numerical:
    if col in train.columns:
        train_missing = train[col].isnull().sum()
        test_missing = test[col].isnull().sum()
        if train_missing > 0:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            print(f"   {col} (train): filled {train_missing} with median {median_val}")
        if test_missing > 0:
            # Use training median for test
            median_val = train[col].median()
            test[col] = test[col].fillna(median_val)
            print(f"   {col} (test): filled {test_missing} with median {median_val}")

# Price: Fill with median by Category
print("\nFilling Price missing values with median by Category...")
for dataset_name, dataset in [('train', train), ('test', test)]:
    price_missing = dataset['Price'].isnull().sum()
    if price_missing > 0:
        # Fill with category median from training set
        category_price_median = train.groupby('Category')['Price'].median().to_dict()
        dataset['Price'] = dataset.apply(
            lambda row: category_price_median.get(row['Category'], train['Price'].median()) 
            if pd.isna(row['Price']) else row['Price'], 
            axis=1
        )
        print(f"   {dataset_name}: filled {price_missing} missing Price values")

# Price_Sine: Recalculate from Price or fill with median
print("\nHandling Price_Sine missing values...")
for dataset_name, dataset in [('train', train), ('test', test)]:
    price_sine_missing = dataset['Price_Sine'].isnull().sum()
    if price_sine_missing > 0:
        # Fill with median
        median_val = train['Price_Sine'].median()
        dataset['Price_Sine'] = dataset['Price_Sine'].fillna(median_val)
        print(f"   {dataset_name}: filled {price_sine_missing} with median {median_val}")

# Remaining numerical features: KNN Imputation
print("\nApplying KNN imputation for remaining numerical features...")
remaining_numerical = ['Age', 'Reviews_Read', 'Items_In_Cart', 'Socioeconomic_Status_Score', 'Engagement_Score']
features_to_impute = [col for col in remaining_numerical if train[col].isnull().sum() > 0 or test[col].isnull().sum() > 0]

if features_to_impute:
    print(f"   Features for KNN imputation: {features_to_impute}")
    
    # Prepare data for KNN imputation
    all_numerical = ['Age', 'Gender', 'Reviews_Read', 'Price', 'Discount', 'Category', 
                     'Items_In_Cart', 'Email_Interaction', 'Socioeconomic_Status_Score', 
                     'Engagement_Score', 'AB_Bucket', 'Price_Sine', 'Day']
    
    # Apply KNN imputer
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    train_numerical = train[all_numerical].copy()
    test_numerical = test[all_numerical].copy()
    
    train[all_numerical] = imputer.fit_transform(train_numerical)
    test[all_numerical] = imputer.transform(test_numerical)
    
    print(f"KNN imputation completed")

print("\nMissing values AFTER handling:")
print("\nTrain:")
train_missing_after = train.isnull().sum()[train.isnull().sum() > 0]
if len(train_missing_after) == 0:
    print("No missing values!")
else:
    print(train_missing_after)

print("\nTest:")
test_missing_after = test.isnull().sum()[test.isnull().sum() > 0]
if len(test_missing_after) == 0:
    print("No missing values!")
else:
    print(test_missing_after)

# ============================================================================
# PART 3: OUTLIER DETECTION & HANDLING
# ============================================================================
print("\n" + "="*80)
print("PART 3: OUTLIER DETECTION & ANALYSIS")
print("="*80)

def detect_outliers_iqr(data, column, multiplier=3):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers, lower_bound, upper_bound

# Check for outliers in key numerical features
outlier_features = ['Price', 'Items_In_Cart', 'Age', 'Discount', 'Engagement_Score']

for feature in outlier_features:
    if feature in train.columns:
        outliers, lower, upper = detect_outliers_iqr(train, feature, multiplier=3)
        outlier_count = outliers.sum()
        outlier_pct = (outlier_count / len(train)) * 100
        
        print(f"\n{feature}:")
        print(f"   Outliers: {outlier_count} ({outlier_pct:.2f}%)")
        print(f"   Bounds: [{lower:.2f}, {upper:.2f}]")
        print(f"   Min: {train[feature].min():.2f}, Max: {train[feature].max():.2f}")
        print(f"   Median: {train[feature].median():.2f}, Mean: {train[feature].mean():.2f}")
        
        if outlier_count > 0:
            # Show some outlier values
            outlier_values = train.loc[outliers, feature].head(10).values
            print(f"   Sample outlier values: {outlier_values}")

# Decision on outliers: For now, we'll keep them but flag them
# We can create outlier flags as features
print("\nCreating outlier flags as features...")
for feature in ['Price', 'Items_In_Cart']:
    if feature in train.columns:
        outliers_train, _, _ = detect_outliers_iqr(train, feature, multiplier=3)
        outliers_test, _, _ = detect_outliers_iqr(test, feature, multiplier=3)
        
        train[f'{feature}_is_outlier'] = outliers_train.astype(int)
        test[f'{feature}_is_outlier'] = outliers_test.astype(int)
        print(f"   Created {feature}_is_outlier flag")

# ============================================================================
# PART 4: DATA VALIDATION & SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PART 4: DATA VALIDATION & SUMMARY")
print("="*80)

print("\nFinal Data Quality Checks:")
print(f"\n1. Shape:")
print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")

print(f"\n2. Missing values:")
print(f"   Train: {train.isnull().sum().sum()}")
print(f"   Test: {test.isnull().sum().sum()}")

print(f"\n3. Categorical standardization:")
print(f"   Time_of_Day unique: {train['Time_of_Day'].nunique()} (expected: 4 including Unknown)")
print(f"   Payment_Method unique: {train['Payment_Method'].nunique()} (expected: 5 including Unknown)")
print(f"   Referral_Source unique: {train['Referral_Source'].nunique()} (expected: 6 including Unknown)")
print(f"   Device_Type unique: {train['Device_Type'].nunique()}")

print(f"\n4. Data types:")
print(train.dtypes)

print(f"\n5. Purchase distribution in train:")
print(train['Purchase'].value_counts())
print(f"   Purchase rate: {train['Purchase'].mean():.2%}")

# ============================================================================
# PART 5: SAVE CLEANED DATA
# ============================================================================
print("\n" + "="*80)
print("PART 5: SAVING CLEANED DATA")
print("="*80)

train.to_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/train_cleaned.csv', index=False)
test.to_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/test_cleaned.csv', index=False)

print("\nCleaned datasets saved")

print("\n" + "="*80)
print("STEP 2: DATA CLEANING COMPLETE!")
print("="*80)

print("="*80)
print("STEP 3: FEATURE ENGINEERING")
print("="*80)

# # Load cleaned datasets
# train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/train_cleaned.csv')
# test = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/test_cleaned.csv')

# print(f"\nLoaded cleaned datasets:")
# print(f"   Train: {train.shape}")
# print(f"   Test: {test.shape}")

# Store target and IDs
y_train = train['Purchase'].copy()
train_ids = train['id'].copy()
test_ids = test['id'].copy()

print(f"\nTarget distribution: {y_train.value_counts().to_dict()}")
print(f"   Purchase rate: {y_train.mean():.2%}")

# ============================================================================
# PART 1: BASIC FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("PART 1: BASIC FEATURE ENGINEERING")
print("="*80)

def create_basic_features(df):
    """Create basic engineered features"""
    df = df.copy()
    
    # 1. Price-related features
    print("\nCreating price-related features...")
    df['Price_per_Item'] = df['Price'] / (df['Items_In_Cart'] + 1)  # +1 to avoid div by 0
    df['Discount_Pct'] = (df['Discount'] / 100)
    df['Effective_Price'] = df['Price'] * (1 - df['Discount_Pct'])
    df['Has_Discount'] = (df['Discount'] > 0).astype(int)
    df['High_Discount'] = (df['Discount'] >= 30).astype(int)
    
    # Price bins
    df['Price_Bin'] = pd.cut(df['Price'], 
                              bins=[0, 200, 500, 1000, np.inf],
                              labels=['low', 'medium', 'high', 'premium'])
    
    # 2. Cart-related features
    print("Creating cart-related features...")
    df['Has_Items_In_Cart'] = (df['Items_In_Cart'] > 0).astype(int)
    df['Cart_Empty'] = (df['Items_In_Cart'] == 0).astype(int)
    df['Cart_Size'] = pd.cut(df['Items_In_Cart'],
                              bins=[-1, 0, 2, 5, np.inf],
                              labels=['empty', 'small', 'medium', 'large'])
    
    # 3. Engagement features
    print("Creating engagement features...")
    df['High_Engagement'] = (df['Engagement_Score'] > df['Engagement_Score'].median()).astype(int)
    df['Read_Reviews'] = (df['Reviews_Read'] > 0).astype(int)
    df['Heavy_Researcher'] = (df['Reviews_Read'] >= 4).astype(int)
    
    # 4. Temporal features
    print("Creating temporal features...")
    df['Day_Normalized'] = df['Day'] / 100  # Normalize to [0,1]
    df['Campaign_Active'] = df['Campaign_Period'].astype(int)
    
    # Distance to campaign periods (Days 25-50, 75-90)
    df['Days_Since_Campaign1_Start'] = np.abs(df['Day'] - 25)
    df['Days_To_Campaign2_Start'] = np.maximum(75 - df['Day'], 0)
    df['In_Campaign_Window'] = (
        ((df['Day'] >= 25) & (df['Day'] <= 50)) | 
        ((df['Day'] >= 75) & (df['Day'] <= 90))
    ).astype(int)
    
    # 5. Demographic features
    print("Creating demographic features...")
    df['Age_Group'] = pd.cut(df['Age'],
                              bins=[0, 25, 35, 50, 100],
                              labels=['young', 'adult', 'middle', 'senior'])
    df['High_SES'] = (df['Socioeconomic_Status_Score'] > df['Socioeconomic_Status_Score'].median()).astype(int)
    
    return df

train = create_basic_features(train)
test = create_basic_features(test)

print(f"\nBasic features created. New shape:")
print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")

# ============================================================================
# PART 2: INTERACTION FEATURES
# ============================================================================
print("\n" + "="*80)
print("PART 2: INTERACTION FEATURES")
print("="*80)

def create_interaction_features(df):
    """Create interaction features between key variables"""
    df = df.copy()
    
    print("\nCreating interaction features...")
    
    # Device × Time interactions
    df['Device_Time'] = df['Device_Type'] + '_' + df['Time_of_Day']
    
    # Device × Campaign
    df['Device_Campaign'] = df['Device_Type'] + '_' + df['Campaign_Period'].astype(str)
    
    # Category × Price_Bin
    df['Category_PriceBin'] = df['Category'].astype(str) + '_' + df['Price_Bin'].astype(str)
    
    # Email × Device
    df['Email_Device'] = df['Email_Interaction'].astype(str) + '_' + df['Device_Type']
    
    # Payment × Referral (already exists as PM_RS_Combo, but let's verify)
    # Already have PM_RS_Combo
    
    # High value sessions (high price + high engagement)
    df['High_Value_Session'] = ((df['Price'] > df['Price'].median()) & 
                                 (df['Engagement_Score'] > df['Engagement_Score'].median())).astype(int)
    
    # Cart abandonment risk (high items but no email interaction)
    df['Cart_Abandon_Risk'] = ((df['Items_In_Cart'] > 3) & 
                                (df['Email_Interaction'] == 0)).astype(int)
    
    # Mobile evening shopper
    df['Mobile_Evening'] = ((df['Device_Type'] == 'Mobile') & 
                             (df['Time_of_Day'] == 'evening')).astype(int)
    
    # Campaign responsive (high engagement during campaign)
    df['Campaign_Engaged'] = ((df['Campaign_Period'] == True) & 
                               (df['Engagement_Score'] > df['Engagement_Score'].median())).astype(int)
    
    return df

train = create_interaction_features(train)
test = create_interaction_features(test)

print(f"\nInteraction features created. New shape:")
print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")

# ============================================================================
# PART 3: CATEGORICAL ENCODING
# ============================================================================
print("\n" + "="*80)
print("PART 3: CATEGORICAL ENCODING")
print("="*80)

# Identify categorical columns
categorical_cols = [
    'Time_of_Day', 'Device_Type', 'Payment_Method', 'Referral_Source',
    'Price_Bin', 'Cart_Size', 'Age_Group', 'Device_Time', 'Device_Campaign',
    'Category_PriceBin', 'Email_Device'
]

# Remove PM_RS_Combo as it has too many unique values
# We'll use one-hot encoding for most categories

print(f"\nEncoding categorical features...")
print(f"   Features to encode: {len(categorical_cols)}")

# One-hot encode categorical features
train_encoded = pd.get_dummies(train, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test, columns=categorical_cols, prefix=categorical_cols, drop_first=True)

print(f"\nAfter encoding:")
print(f"   Train: {train_encoded.shape}")
print(f"   Test: {test_encoded.shape}")

# Align train and test columns
train_cols = set(train_encoded.columns)
test_cols = set(test_encoded.columns)

# Find columns only in train
only_in_train = train_cols - test_cols - {'Purchase'}
if only_in_train:
    print(f"\nColumns only in train: {len(only_in_train)}")
    # Add missing columns to test with 0s
    for col in only_in_train:
        test_encoded[col] = 0

# Find columns only in test
only_in_test = test_cols - train_cols
if only_in_test:
    print(f"Columns only in test: {len(only_in_test)}")
    # Add missing columns to train with 0s
    for col in only_in_test:
        train_encoded[col] = 0

# Ensure same column order
if 'Purchase' in train_encoded.columns:
    feature_cols = [col for col in train_encoded.columns if col not in ['id', 'Session_ID', 'Purchase', 'PM_RS_Combo']]
else:
    feature_cols = [col for col in train_encoded.columns if col not in ['id', 'Session_ID', 'PM_RS_Combo']]

# Also remove any remaining object columns
feature_cols = [col for col in feature_cols if train_encoded[col].dtype != 'object']

print(f"\nFinal feature set: {len(feature_cols)} features")

# ============================================================================
# PART 4: FEATURE STATISTICS & CORRELATIONS
# ============================================================================
print("\n" + "="*80)
print("PART 4: FEATURE STATISTICS")
print("="*80)

# Get top correlations with target
print("\nTop 20 features correlated with Purchase:")
correlations = train_encoded[feature_cols + ['Purchase']].corr()['Purchase'].abs().sort_values(ascending=False)
print(correlations.head(21))  # 21 to include Purchase itself

print("\nNewly created features correlation with Purchase:")
new_features = [
    'Price_per_Item', 'Discount_Pct', 'Effective_Price', 'Has_Discount', 'High_Discount',
    'Has_Items_In_Cart', 'Cart_Empty', 'High_Engagement', 'Read_Reviews', 'Heavy_Researcher',
    'Campaign_Active', 'Days_Since_Campaign1_Start', 'Days_To_Campaign2_Start', 'In_Campaign_Window',
    'High_SES', 'High_Value_Session', 'Cart_Abandon_Risk', 'Mobile_Evening', 'Campaign_Engaged'
]
new_features_existing = [f for f in new_features if f in train_encoded.columns]
if new_features_existing:
    new_feature_corr = train_encoded[new_features_existing + ['Purchase']].corr()['Purchase'].abs().sort_values(ascending=False)
    print(new_feature_corr)

# ============================================================================
# PART 5: PREPARE FINAL DATASETS
# ============================================================================
print("\n" + "="*80)
print("PART 5: PREPARING FINAL DATASETS")
print("="*80)

# Create final feature matrices
X_train = train_encoded[feature_cols].copy()
X_test = test_encoded[feature_cols].copy()

print(f"\nFinal dataset shapes:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}")

print(f"\nData types check:")
print(f"   Object columns in X_train: {X_train.select_dtypes(include=['object']).columns.tolist()}")
print(f"   Object columns in X_test: {X_test.select_dtypes(include=['object']).columns.tolist()}")

# Check for any remaining NaN or inf
print(f"\nData quality check:")
print(f"   NaN in X_train: {X_train.isna().sum().sum()}")
print(f"   NaN in X_test: {X_test.isna().sum().sum()}")
# Check inf only on numeric columns
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
print(f"   Inf in X_train: {np.isinf(X_train[numeric_cols].values).sum()}")
print(f"   Inf in X_test: {np.isinf(X_test[numeric_cols].values).sum()}")

# Replace any inf values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Fill any remaining NaN with median
if X_train.isna().sum().sum() > 0:
    print(f"\nFilling {X_train.isna().sum().sum()} remaining NaN values...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

# ============================================================================
# PART 6: SAVE ENGINEERED DATASETS
# ============================================================================
print("\n" + "="*80)
print("PART 6: SAVING ENGINEERED DATASETS")
print("="*80)

# Save feature matrices
X_train.to_csv('X_train_engineered.csv', index=False)
X_test.to_csv('X_test_engineered.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=True)

# Save feature names
with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_cols))

# Save ID mappings
train_ids.to_csv('train_ids.csv', index=False, header=True)
test_ids.to_csv('test_ids.csv', index=False, header=True)

print(f"\nEngineered datasets saved")

print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING COMPLETE!")
print("="*80)

print("="*80)
print("STEP 4: MODEL SELECTION & TRAINING")
print("="*80)

# # Load engineered datasets
# X_train = pd.read_csv('X_train_engineered.csv')
# y_train = pd.read_csv('y_train.csv').squeeze()
# X_test = pd.read_csv('X_test_engineered.csv')
# train_ids = pd.read_csv('train_ids.csv').squeeze()
# test_ids = pd.read_csv('test_ids.csv').squeeze()

# print(f"\nLoaded engineered datasets:")
# print(f"   X_train: {X_train.shape}")
# print(f"   y_train: {y_train.shape}")
# print(f"   X_test: {X_test.shape}")
# print(f"   Purchase rate: {y_train.mean():.2%}")

# ============================================================================
# PART 1: BASELINE MODEL - LOGISTIC REGRESSION
# ============================================================================
print("\n" + "="*80)
print("PART 1: BASELINE MODEL - LOGISTIC REGRESSION")
print("="*80)

# Temporal cross-validation setup (respect time ordering)
# We'll use 5 folds for time series split
tscv = TimeSeriesSplit(n_splits=5)

print("\nUsing Temporal Cross-Validation (5 folds)")

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights: {class_weight_dict}")

# Train Logistic Regression
print("\nTraining Logistic Regression (baseline)...")
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

print(f"\nLogistic Regression Results:")
print(f"   Training F1 Score: {train_f1_lr:.4f}")
print(f"   Training AUC-ROC: {train_auc_lr:.4f}")
print(f"   Mean CV F1: {lr_cv_scores.mean():.4f}")

# ============================================================================
# PART 2: RANDOM FOREST
# ============================================================================
print("\n" + "="*80)
print("PART 2: RANDOM FOREST")
print("="*80)

print("\nTraining Random Forest...")
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

print(f"\nRandom Forest Results:")
print(f"   Training F1 Score: {train_f1_rf:.4f}")
print(f"   Training AUC-ROC: {train_auc_rf:.4f}")
print(f"   Mean CV F1: {rf_cv_scores.mean():.4f}")

# ============================================================================
# PART 3: XGBOOST
# ============================================================================
print("\n" + "="*80)
print("PART 3: XGBOOST")
print("="*80)

print("\nTraining XGBoost...")
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

print(f"\nXGBoost Results:")
print(f"   Training F1 Score: {train_f1_xgb:.4f}")
print(f"   Training AUC-ROC: {train_auc_xgb:.4f}")
print(f"   Mean CV F1: {xgb_cv_scores.mean():.4f}")

# ============================================================================
# PART 4: LIGHTGBM
# ============================================================================
print("\n" + "="*80)
print("PART 4: LIGHTGBM")
print("="*80)

print("\nTraining LightGBM...")
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

print(f"\nLightGBM Results:")
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

print("\nModel Comparison (sorted by CV F1 Score):")
print(results.to_string(index=False))

# Select best model based on CV F1
best_model_name = results.iloc[0]['Model']
best_cv_f1 = results.iloc[0]['CV_F1_Mean']

print(f"\nBest Model: {best_model_name}")
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
print("\nTesting different probability thresholds:")
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

print(f"\nOptimal Threshold: {best_threshold:.2f}")
print(f"   F1 Score at optimal threshold: {best_threshold_f1:.4f}")

# ============================================================================
# PART 7: FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n" + "="*80)
print("PART 7: FEATURE IMPORTANCE")
print("="*80)

if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    print(f"\nTop 20 Most Important Features ({best_model_name}):")
    
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

# ============================================================================
# PART 8: PREDICTIONS ON TEST SET
# ============================================================================
print("\n" + "="*80)
print("PART 8: PREDICTIONS ON TEST SET")
print("="*80)

print(f"\nMaking predictions with {best_model_name}...")

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

# Also save probabilities for analysis
submission_proba = pd.DataFrame({
    'id': test_ids,
    'Purchase_Probability': test_proba,
    'Purchase_Prediction': test_pred
})
submission_proba.to_csv('submission_with_probabilities.csv', index=False)

print("\n" + "="*80)
print("STEP 4: MODEL TRAINING COMPLETE!")
print("="*80)

print("="*80)
print("STEP 6: HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*80)

# # Load data
# X_train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/X_train_engineered.csv')
# y_train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/y_train.csv').squeeze()
# X_test = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/X_test_engineered.csv')
# train_ids = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/train_ids.csv').squeeze()
# test_ids = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/test_ids.csv').squeeze()

# print(f"\nLoaded datasets: X_train {X_train.shape}, X_test {X_test.shape}")

tscv = TimeSeriesSplit(n_splits=5)
f1_scorer = make_scorer(f1_score)

# ============================================================================
# PART 1: LIGHTGBM HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("PART 1: LIGHTGBM TUNING")
print("="*80)

print("\nBaseline: CV F1=0.7944")
print("   Params: n_estimators=200, max_depth=6, learning_rate=0.1, num_leaves=31")

# Focused grid (most impactful parameters)
lgbm_param_grid = {
    'n_estimators': [200, 250, 300],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.08, 0.1, 0.12],
    'num_leaves': [31, 40, 50],
    'min_child_samples': [15, 20, 25]
}

print(f"\nFocused parameter grid (most impactful):")
for param, values in lgbm_param_grid.items():
    print(f"   {param:20s}: {values}")

n_comb = np.prod([len(v) for v in lgbm_param_grid.values()])
print(f"\nCombinations: {n_comb} × 5 folds = {n_comb*5} fits")
print(f"   Estimated time: ~{n_comb*5*2/60:.0f} minutes")

lgbm_base = LGBMClassifier(
    class_weight='balanced',
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print(f"\nRunning GridSearchCV for LightGBM...")
start = time.time()

grid_lgbm = GridSearchCV(
    lgbm_base, lgbm_param_grid,
    cv=tscv, scoring=f1_scorer,
    n_jobs=-1, verbose=2
)
grid_lgbm.fit(X_train, y_train)

elapsed = time.time() - start
print(f"\nLightGBM tuning completed in {elapsed/60:.1f} minutes")

print(f"\nBest LightGBM Parameters:")
for param, value in grid_lgbm.best_params_.items():
    print(f"   {param:20s}: {value}")

lgbm_best_score = grid_lgbm.best_score_
improvement_lgbm = lgbm_best_score - 0.7944

print(f"\nPerformance:")
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

# ============================================================================
# PART 2: XGBOOST HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("PART 2: XGBOOST TUNING")
print("="*80)

print("\nBaseline: CV F1=0.7917")
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

print(f"\nFocused parameter grid:")
for param, values in xgb_param_grid.items():
    print(f"   {param:20s}: {values}")

n_comb = np.prod([len(v) for v in xgb_param_grid.values()])
print(f"\nCombinations: {n_comb} × 5 folds = {n_comb*5} fits")
print(f"   Estimated time: ~{n_comb*5*2/60:.0f} minutes")

xgb_base = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

print(f"\nRunning GridSearchCV for XGBoost...")
start = time.time()

grid_xgb = GridSearchCV(
    xgb_base, xgb_param_grid,
    cv=tscv, scoring=f1_scorer,
    n_jobs=-1, verbose=2
)
grid_xgb.fit(X_train, y_train)

elapsed = time.time() - start
print(f"\nXGBoost tuning completed in {elapsed/60:.1f} minutes")

print(f"\nBest XGBoost Parameters:")
for param, value in grid_xgb.best_params_.items():
    print(f"   {param:20s}: {value}")

xgb_best_score = grid_xgb.best_score_
improvement_xgb = xgb_best_score - 0.7917

print(f"\nPerformance:")
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

print("\nComplete Model Comparison:")
print(comparison.to_string(index=False))

# Select winner
best_model_name = comparison.iloc[0]['Model']
best_cv_f1 = comparison.iloc[0]['CV_F1']

print(f"\nWINNER: {best_model_name}")
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

print(f"\nFinding optimal threshold for {best_model_name}...")

train_proba = best_model.predict_proba(X_train)[:, 1]

best_f1_val = 0
best_thresh = 0.5

for threshold in np.arange(0.35, 0.70, 0.01):
    pred = (train_proba >= threshold).astype(int)
    f1_val = f1_score(y_train, pred)
    if f1_val > best_f1_val:
        best_f1_val = f1_val
        best_thresh = threshold

print(f"\nOptimal threshold: {best_thresh:.2f}")
print(f"   Training F1 at threshold: {best_f1_val:.4f}")

# Show key thresholds
print(f"\nF1 scores at different thresholds:")
for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    pred = (train_proba >= t).astype(int)
    f1_val = f1_score(y_train, pred)
    marker = " OPTIMAL" if abs(t - best_thresh) < 0.01 else ""
    print(f"   Threshold {t:.2f}: F1 = {f1_val:.4f}{marker}")

# ============================================================================
# PART 5: GENERATE TEST PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("PART 5: TEST PREDICTIONS")
print("="*80)

print(f"\nGenerating predictions...")

test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_thresh).astype(int)

print(f"\nTest Predictions:")
print(f"   Threshold used: {best_thresh:.2f}")
print(f"   Predicted purchases: {test_pred.sum()} ({test_pred.mean():.2%})")
print(f"   Mean probability: {test_proba.mean():.4f}")
print(f"   Median probability: {np.median(test_proba):.4f}")
print(f"   Min probability: {test_proba.min():.4f}")
print(f"   Max probability: {test_proba.max():.4f}")

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

print("\nFiles saved")

print("\n" + "="*80)
print("HYPERPARAMETER TUNING COMPLETE!")
print("="*80)

print("="*80)
print("STEP 7: FEATURE SELECTION FOR IMPROVED GENERALIZATION")
print("="*80)

# # Load data
# X_train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/X_train_engineered.csv')
# y_train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/y_train.csv').squeeze()
# X_test = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/X_test_engineered.csv')
# test_ids = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/test_ids.csv').squeeze()

# print(f"\nLoaded: X_train {X_train.shape}, X_test {X_test.shape}")

# feature_importance = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/kaggle_competition/feature_importance.csv')

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

print(f"\nImportance Coverage:")
for pct in [80, 85, 90, 95]:
    n = (feature_importance['cumulative_pct'] <= pct).sum()
    print(f"   Top {n} features = {pct}% importance")

print(f"\nTop 15 Features:")
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

print(f"\nTesting {len(feature_counts)} configurations with 5-fold CV...\n")

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

print(f"\nBEST: {best_n} features | CV F1: {best_cv:.4f}")
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

print(f"\nOptimal threshold: {best_thresh:.2f} (Val F1: {best_f1:.4f})")

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

print(f"\nTest Predictions:")
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

print("\nSaved")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)