import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
train_df = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/train_dataset_M1_with_id.csv')
test_df = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/test_dataset_M1_with_id.csv')

print(f"Training: {train_df.shape} (Days {train_df['Day'].min()}-{train_df['Day'].max()})")
print(f"Test: {test_df.shape} (Days {test_df['Day'].min()}-{test_df['Day'].max()})")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
def create_features(df, train_stats=None, is_train=True):
    """
    Create features with NO data leakage.
    If training: compute statistics from training data.
    If testing: use provided training statistics.
    """
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # STEP A: Compute/Use Statistics
    # -------------------------------------------------------------------------
    if is_train:
        # Compute all statistics from TRAINING DATA ONLY
        stats = {
            'Age_median': df['Age'].median(),
            'Price_median': df['Price'].median(),
            'Items_median': df['Items_In_Cart'].median(),
            'SES_median': df['Socioeconomic_Status_Score'].median(),
            'Engagement_median': df['Engagement_Score'].median(),
            'Reviews_median': df['Reviews_Read'].median(),
        }
    else:
        # Use provided training statistics
        assert train_stats is not None, "Must provide train_stats for test data!"
        stats = train_stats
    
    # -------------------------------------------------------------------------
    # STEP B: Handle Missing Values (using training statistics)
    # -------------------------------------------------------------------------
    df['Age'] = df['Age'].fillna(stats['Age_median'])
    df['Price'] = df['Price'].fillna(stats['Price_median'])
    df['Items_In_Cart'] = df['Items_In_Cart'].fillna(stats['Items_median'])
    df['Socioeconomic_Status_Score'] = df['Socioeconomic_Status_Score'].fillna(stats['SES_median'])
    df['Engagement_Score'] = df['Engagement_Score'].fillna(stats['Engagement_median'])
    df['Reviews_Read'] = df['Reviews_Read'].fillna(stats['Reviews_median'])
    df['Discount'] = df['Discount'].fillna(0)
    
    # Categorical missing values
    df['Payment_Method'] = df['Payment_Method'].fillna('Unknown')
    df['Referral_Source'] = df['Referral_Source'].fillna('Unknown')
    df['Device_Type'] = df['Device_Type'].fillna('Unknown')
    df['Time_of_Day'] = df['Time_of_Day'].fillna('Unknown')
    
    # -------------------------------------------------------------------------
    # STEP C: Clean String Columns
    # -------------------------------------------------------------------------
    # Fix typos in categorical variables
    # Standardize categorical columns using regex patterns
    categorical_mappings = {
        'payment_method': {
            r'^cred.*$': 'credit',
            r'^cash$': 'cash',
            r'^pay[\s_]?pal$': 'paypal',
            r'^bank.*$': 'bank',
        },
        'time_of_day': {
            r'^m[0o].*rning$': 'morning',
            r'^aftern?[0o].*n$': 'afternoon',
            r'^even.*g$': 'evening',
        },
        'referral_source': {
            r'^s[0o].*cial.*media$': 'social_media',
            r'^search.*engine$': 'search_engine',
            r'^ads$': 'ads',
            r'^email$': 'email',
            r'^direct$': 'direct',
        },
        'device_type': {
            r'^mob.*$': 'mobile',
            r'^desk.*$': 'desktop',
            r'^tab.*$': 'tablet',
        }
    }

    # Apply standardization
    for col in ['Payment_Method', 'Time_of_Day', 'Referral_Source', 'Device_Type']:
        if col in df.columns:
            # Convert to lowercase and strip whitespace
            df[col] = df[col].str.lower().str.strip()
            # Replace string 'nan' with actual NaN
            df[col] = df[col].replace("nan", np.nan)
            # Apply regex mappings
            col_lower = col.lower()
            if col_lower in categorical_mappings:
                df[col] = df[col].replace(categorical_mappings[col_lower], regex=True)
    
    # -------------------------------------------------------------------------
    # STEP D: Create Features
    # -------------------------------------------------------------------------
    
    # Price features
    df['Final_Price'] = df['Price'] * (1 - df['Discount'] / 100)
    df['Cart_Value'] = df['Final_Price'] * df['Items_In_Cart']
    df['Price_Log'] = np.log1p(df['Price'])
    df['High_Discount'] = (df['Discount'] >= 30).astype(int)
    
    # Cart features
    df['Has_Items_In_Cart'] = (df['Items_In_Cart'] > 0).astype(int)
    df['Multiple_Items'] = (df['Items_In_Cart'] > 1).astype(int)
    
    # Engagement features (using training median)
    df['High_Engagement'] = (df['Engagement_Score'] > stats['Engagement_median']).astype(int)
    df['Engagement_Log'] = np.log1p(df['Engagement_Score'])
    
    # Review features
    df['Reviews_Per_Item'] = df['Reviews_Read'] / (df['Items_In_Cart'] + 1)
    df['Many_Reviews'] = (df['Reviews_Read'] >= 3).astype(int)
    
    # SES features (using training median)
    df['High_SES'] = (df['Socioeconomic_Status_Score'] > stats['SES_median']).astype(int)
    
    # Campaign features
    df['Campaign_Period'] = df['Campaign_Period'].fillna(False).astype(int)
    df['In_Campaign_1'] = ((df['Day'] >= 25) & (df['Day'] <= 50)).astype(int)
    df['In_Campaign_2'] = ((df['Day'] >= 75) & (df['Day'] <= 90)).astype(int)
    
    # Time features
    df['Day_Of_Week'] = df['Day'] % 7
    df['Is_Weekend'] = df['Day_Of_Week'].isin([5, 6]).astype(int)
    
    # Category features
    df['Premium_Category'] = df['Category'].isin([3, 4]).astype(int)
    
    # INTERACTION FEATURES
    df['Cart_X_Engagement'] = df['Items_In_Cart'] * df['Engagement_Score']
    df['Email_X_Engagement'] = df['Email_Interaction'] * df['Engagement_Score']
    df['Email_X_HighEng'] = df['Email_Interaction'] * df['High_Engagement']
    df['Campaign_X_Discount'] = df['Campaign_Period'] * df['Discount']
    df['HasCart_X_HighEng'] = df['Has_Items_In_Cart'] * df['High_Engagement']
    
    return df, stats

# Apply to train (compute statistics)
train_processed, train_stats = create_features(train_df, is_train=True)

# Apply to test (use training statistics)
test_processed, _ = create_features(test_df, train_stats=train_stats, is_train=False)

print(f"Features created: {train_processed.shape[1]} columns")

# =============================================================================
# 3. PREPARE DATA FOR MODELING
# =============================================================================
# Define columns to drop
drop_cols = ['id', 'Session_ID', 'Purchase', 'AB_Bucket', 'Price_Sine', 'PM_RS_Combo']

# Get target and features
y_train = train_processed['Purchase'].values
feature_cols = [col for col in train_processed.columns if col not in drop_cols]
X_train = train_processed[feature_cols].copy()
X_test = test_processed[feature_cols].copy()

print(f"Features: {len(feature_cols)}")

# Encode categorical variables
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"Encoding {len(categorical_cols)} categorical features")

for col in categorical_cols:
    le = LabelEncoder()
    # Fit on training data only
    le.fit(X_train[col].astype(str))
    X_train[col] = le.transform(X_train[col].astype(str))
    
    # Transform test data, handling unseen categories
    test_values = X_test[col].astype(str)
    X_test[col] = [le.transform([v])[0] if v in le.classes_ else 0 for v in test_values]

# Fill any remaining NaNs
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 4. TRAIN MODEL
# =============================================================================
# Use ensemble of 3 models for better performance
model1 = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    min_samples_split=15,
    min_samples_leaf=8,
    subsample=0.85,
    random_state=42
)

model2 = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model3 = GradientBoostingClassifier(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.08,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=123
)

# Combine models with voting
ensemble = VotingClassifier(
    estimators=[
        ('gb1', model1),
        ('rf', model2),
        ('gb2', model3)
    ],
    voting='soft',
    weights=[2, 1, 2]
)

# Train ensemble
ensemble.fit(X_train_scaled, y_train)

# Check training accuracy
train_pred = ensemble.predict(X_train_scaled)
train_acc = (train_pred == y_train).mean()
print(f"Training accuracy: {train_acc:.4f}")

# =============================================================================
# 5. MAKE PREDICTIONS
# =============================================================================
# Get probabilities
y_test_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

# Convert to binary predictions (threshold = 0.5)
y_test_pred = (y_test_proba >= 0.5).astype(int)

print(f"Predicted purchases: {y_test_pred.sum()} / {len(y_test_pred)} ({y_test_pred.mean()*100:.1f}%)")

# =============================================================================
# 6. CREATE SUBMISSION FILE
# =============================================================================
import os
os.makedirs('outputs', exist_ok=True)

submission = pd.DataFrame({
    'id': test_df['id'],
    'Purchase': y_test_pred
})

submission.to_csv('outputs/submission_final.csv', index=False)

# Also save with probabilities for analysis
submission_with_proba = pd.DataFrame({
    'id': test_df['id'],
    'Purchase_Probability': y_test_proba,
    'Purchase': y_test_pred
})
submission_with_proba.to_csv('outputs/submission_with_probabilities.csv', index=False)