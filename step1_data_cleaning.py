import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import re
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 2: DATA CLEANING & PREPROCESSING")
print("="*80)

# Load datasets
train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/train_dataset_M1_with_id.csv')
test = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/test_dataset_M1_with_id.csv')

print(f"\n📥 Loaded datasets:")
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
print("\n🧹 Cleaning Time_of_Day...")
train['Time_of_Day'] = train['Time_of_Day'].apply(clean_time_of_day)
test['Time_of_Day'] = test['Time_of_Day'].apply(clean_time_of_day)
print(f"   Train unique values: {train['Time_of_Day'].unique()}")
print(f"   Test unique values: {test['Time_of_Day'].unique()}")
print(f"   Train value counts:\n{train['Time_of_Day'].value_counts()}")

print("\n🧹 Cleaning Payment_Method...")
train['Payment_Method'] = train['Payment_Method'].apply(clean_payment_method)
test['Payment_Method'] = test['Payment_Method'].apply(clean_payment_method)
print(f"   Train unique values: {train['Payment_Method'].unique()}")
print(f"   Test unique values: {test['Payment_Method'].unique()}")
print(f"   Train value counts:\n{train['Payment_Method'].value_counts()}")

print("\n🧹 Cleaning Referral_Source...")
train['Referral_Source'] = train['Referral_Source'].apply(clean_referral_source)
test['Referral_Source'] = test['Referral_Source'].apply(clean_referral_source)
print(f"   Train unique values: {train['Referral_Source'].unique()}")
print(f"   Test unique values: {test['Referral_Source'].unique()}")
print(f"   Train value counts:\n{train['Referral_Source'].value_counts()}")

print("\n🧹 Cleaning Device_Type (removing extra spaces)...")
train['Device_Type'] = train['Device_Type'].str.strip() if train['Device_Type'].dtype == 'object' else train['Device_Type']
test['Device_Type'] = test['Device_Type'].str.strip() if test['Device_Type'].dtype == 'object' else test['Device_Type']

# ============================================================================
# PART 2: HANDLE MISSING VALUES
# ============================================================================
print("\n" + "="*80)
print("PART 2: HANDLING MISSING VALUES")
print("="*80)

print("\n📊 Missing values BEFORE handling:")
print("\nTrain:")
print(train.isnull().sum()[train.isnull().sum() > 0].sort_values(ascending=False))
print("\nTest:")
print(test.isnull().sum()[test.isnull().sum() > 0].sort_values(ascending=False))

# Strategy:
# 1. Categorical: Fill with 'Unknown' or mode
# 2. Numerical: Use median or KNN imputation
# 3. Session_ID, Day: Special handling

# Fill Day missing values in test (forward fill based on id order)
print("\n🔧 Handling missing 'Day' in test set...")
if test['Day'].isnull().any():
    # For test set, missing days are likely sequential
    # Fill with median day from test set
    test['Day'] = test['Day'].fillna(test['Day'].median())
    print(f"   Filled {test['Day'].isnull().sum()} missing Day values with median: {test['Day'].median()}")

# Fill Session_ID missing values - these might be data errors, we'll create placeholder IDs
print("\n🔧 Handling missing Session_IDs...")
train_missing_sid = train['Session_ID'].isnull().sum()
test_missing_sid = test['Session_ID'].isnull().sum()
if train_missing_sid > 0:
    train.loc[train['Session_ID'].isnull(), 'Session_ID'] = [f"MISSING_TRAIN_{i}" for i in range(train_missing_sid)]
    print(f"   Created {train_missing_sid} placeholder Session_IDs in train")
if test_missing_sid > 0:
    test.loc[test['Session_ID'].isnull(), 'Session_ID'] = [f"MISSING_TEST_{i}" for i in range(test_missing_sid)]
    print(f"   Created {test_missing_sid} placeholder Session_IDs in test")

# Categorical features: Fill with 'Unknown'
print("\n🔧 Filling categorical missing values with 'Unknown'...")
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
print("\n🔧 Filling Campaign_Period missing values...")
train['Campaign_Period'] = train['Campaign_Period'].fillna(False)
test['Campaign_Period'] = test['Campaign_Period'].fillna(False)
# Convert to boolean properly
train['Campaign_Period'] = train['Campaign_Period'].map({'True': True, 'False': False, True: True, False: False})
test['Campaign_Period'] = test['Campaign_Period'].map({'True': True, 'False': False, True: True, False: False})

# Numerical features: Median imputation for simple features
print("\n🔧 Filling numerical missing values with median...")
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
print("\n🔧 Filling Price missing values with median by Category...")
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
print("\n🔧 Handling Price_Sine missing values...")
for dataset_name, dataset in [('train', train), ('test', test)]:
    price_sine_missing = dataset['Price_Sine'].isnull().sum()
    if price_sine_missing > 0:
        # Fill with median
        median_val = train['Price_Sine'].median()
        dataset['Price_Sine'] = dataset['Price_Sine'].fillna(median_val)
        print(f"   {dataset_name}: filled {price_sine_missing} with median {median_val}")

# Remaining numerical features: KNN Imputation
print("\n🔧 Applying KNN imputation for remaining numerical features...")
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
    
    print(f"   ✓ KNN imputation completed")

print("\n📊 Missing values AFTER handling:")
print("\nTrain:")
train_missing_after = train.isnull().sum()[train.isnull().sum() > 0]
if len(train_missing_after) == 0:
    print("   ✓ No missing values!")
else:
    print(train_missing_after)

print("\nTest:")
test_missing_after = test.isnull().sum()[test.isnull().sum() > 0]
if len(test_missing_after) == 0:
    print("   ✓ No missing values!")
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
        
        print(f"\n📊 {feature}:")
        print(f"   Outliers: {outlier_count} ({outlier_pct:.2f}%)")
        print(f"   Bounds: [{lower:.2f}, {upper:.2f}]")
        print(f"   Min: {train[feature].min():.2f}, Max: {train[feature].max():.2f}")
        print(f"   Median: {train[feature].median():.2f}, Mean: {train[feature].mean():.2f}")
        
        if outlier_count > 0:
            # Show some outlier values
            outlier_values = train.loc[outliers, feature].head(10).values
            print(f"   Sample outlier values: {outlier_values}")

# Decision on outliers: For now, we'll KEEP them but flag them
# We can create outlier flags as features
print("\n🏷️ Creating outlier flags as features...")
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

print("\n✅ Final Data Quality Checks:")
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

print("\n✅ Cleaned datasets saved:")
print("   📁 /Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/train_cleaned.csv")
print("   📁 /Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/test_cleaned.csv")

print("\n" + "="*80)
print("STEP 2: DATA CLEANING COMPLETE!")
print("="*80)
print("\n📋 Summary of changes:")
print(f"   • Standardized 3 categorical features (Time_of_Day, Payment_Method, Referral_Source)")
print(f"   • Filled all missing values using appropriate strategies")
print(f"   • Created outlier flags for Price and Items_In_Cart")
print(f"   • No rows were dropped (maintained {len(train)} train, {len(test)} test)")
print(f"   • Datasets are ready for feature engineering!")