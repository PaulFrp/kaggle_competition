import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 3: FEATURE ENGINEERING")
print("="*80)

# Load cleaned datasets
train = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/train_cleaned.csv')
test = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/test_cleaned.csv')

print(f"\n📥 Loaded cleaned datasets:")
print(f"   Train: {train.shape}")
print(f"   Test: {test.shape}")

# Store target and IDs
y_train = train['Purchase'].copy()
train_ids = train['id'].copy()
test_ids = test['id'].copy()

print(f"\n🎯 Target distribution: {y_train.value_counts().to_dict()}")
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
    print("\n💰 Creating price-related features...")
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
    print("🛒 Creating cart-related features...")
    df['Has_Items_In_Cart'] = (df['Items_In_Cart'] > 0).astype(int)
    df['Cart_Empty'] = (df['Items_In_Cart'] == 0).astype(int)
    df['Cart_Size'] = pd.cut(df['Items_In_Cart'],
                              bins=[-1, 0, 2, 5, np.inf],
                              labels=['empty', 'small', 'medium', 'large'])
    
    # 3. Engagement features
    print("📊 Creating engagement features...")
    df['High_Engagement'] = (df['Engagement_Score'] > df['Engagement_Score'].median()).astype(int)
    df['Read_Reviews'] = (df['Reviews_Read'] > 0).astype(int)
    df['Heavy_Researcher'] = (df['Reviews_Read'] >= 4).astype(int)
    
    # 4. Temporal features
    print("📅 Creating temporal features...")
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
    print("👤 Creating demographic features...")
    df['Age_Group'] = pd.cut(df['Age'],
                              bins=[0, 25, 35, 50, 100],
                              labels=['young', 'adult', 'middle', 'senior'])
    df['High_SES'] = (df['Socioeconomic_Status_Score'] > df['Socioeconomic_Status_Score'].median()).astype(int)
    
    return df

train = create_basic_features(train)
test = create_basic_features(test)

print(f"\n✅ Basic features created. New shape:")
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
    
    print("\n🔗 Creating interaction features...")
    
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

print(f"\n✅ Interaction features created. New shape:")
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

print(f"\n🏷️ Encoding categorical features...")
print(f"   Features to encode: {len(categorical_cols)}")

# One-hot encode categorical features
train_encoded = pd.get_dummies(train, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test, columns=categorical_cols, prefix=categorical_cols, drop_first=True)

print(f"\n✅ After encoding:")
print(f"   Train: {train_encoded.shape}")
print(f"   Test: {test_encoded.shape}")

# Align train and test columns
train_cols = set(train_encoded.columns)
test_cols = set(test_encoded.columns)

# Find columns only in train
only_in_train = train_cols - test_cols - {'Purchase'}
if only_in_train:
    print(f"\n⚠️  Columns only in train: {len(only_in_train)}")
    # Add missing columns to test with 0s
    for col in only_in_train:
        test_encoded[col] = 0

# Find columns only in test
only_in_test = test_cols - train_cols
if only_in_test:
    print(f"⚠️  Columns only in test: {len(only_in_test)}")
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

print(f"\n📊 Final feature set: {len(feature_cols)} features")

# ============================================================================
# PART 4: FEATURE STATISTICS & CORRELATIONS
# ============================================================================
print("\n" + "="*80)
print("PART 4: FEATURE STATISTICS")
print("="*80)

# Get top correlations with target
print("\n🎯 Top 20 features correlated with Purchase:")
correlations = train_encoded[feature_cols + ['Purchase']].corr()['Purchase'].abs().sort_values(ascending=False)
print(correlations.head(21))  # 21 to include Purchase itself

print("\n📊 Newly created features correlation with Purchase:")
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

print(f"\n📊 Final dataset shapes:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}")

print(f"\n✅ Data types check:")
print(f"   Object columns in X_train: {X_train.select_dtypes(include=['object']).columns.tolist()}")
print(f"   Object columns in X_test: {X_test.select_dtypes(include=['object']).columns.tolist()}")

# Check for any remaining NaN or inf
print(f"\n🔍 Data quality check:")
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
    print(f"\n⚠️  Filling {X_train.isna().sum().sum()} remaining NaN values...")
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

print(f"\n✅ Engineered datasets saved:")
print(f"   📁 X_train_engineered.csv ({X_train.shape[0]} × {X_train.shape[1]})")
print(f"   📁 X_test_engineered.csv ({X_test.shape[0]} × {X_test.shape[1]})")
print(f"   📁 y_train.csv ({len(y_train)} samples)")
print(f"   📁 feature_names.txt ({len(feature_cols)} features)")
print(f"   📁 train_ids.csv, test_ids.csv")

# ============================================================================
# PART 7: FEATURE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING SUMMARY")
print("="*80)

print("\n📋 Feature categories created:")
print(f"   • Basic features: Price-related, Cart-related, Engagement, Temporal, Demographic")
print(f"   • Interaction features: Device×Time, Device×Campaign, Category×Price, etc.")
print(f"   • Encoded features: One-hot encoding of {len(categorical_cols)} categorical variables")
print(f"   • Total features: {len(feature_cols)}")

print("\n🎯 Key engineered features:")
key_features = [
    'Price_per_Item', 'Discount_Pct', 'Effective_Price', 'Has_Discount',
    'Cart_Empty', 'Has_Items_In_Cart', 'Cart_Abandon_Risk',
    'High_Engagement', 'Read_Reviews', 'Campaign_Active',
    'High_Value_Session', 'Mobile_Evening'
]
key_features_existing = [f for f in key_features if f in X_train.columns]
for feat in key_features_existing[:10]:  # Show first 10
    print(f"   • {feat}")

print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING COMPLETE!")
print("="*80)