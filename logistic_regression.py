import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
train_df = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/train_dataset_M1_with_id.csv')
test_df = pd.read_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/test_dataset_M1_with_id.csv')

print(f"   Training: {train_df.shape[0]:,} sessions, {train_df.shape[1]} columns")
print(f"   Test: {test_df.shape[0]:,} sessions, {test_df.shape[1]} columns")
print(f"   Days - Train: {train_df['Day'].min()}-{train_df['Day'].max()}, Test: {test_df['Day'].min()}-{test_df['Day'].max()}")
print(f"   Purchase rate (train): {train_df['Purchase'].mean():.1%}")

# ============================================================================
# 2. DATA CLEANING
# ============================================================================
def clean_data(df, is_train=True, cleaning_params=None):
    """Comprehensive data cleaning"""
    df = df.copy()
    
    if cleaning_params is None:
        cleaning_params = {}
    
    # Identify numeric and categorical columns
    numeric_cols = ['Age', 'Reviews_Read', 'Price', 'Discount', 
                    'Items_In_Cart', 'Socioeconomic_Status_Score', 
                    'Engagement_Score', 'Price_Sine']
    
    categorical_cols = ['Gender', 'Category', 'Time_of_Day', 
                       'Email_Interaction', 'Device_Type', 
                       'Payment_Method', 'Referral_Source', 
                       'Campaign_Period']
    
    print(f"   {'Training' if is_train else 'Test'} set:")
    
    # Check for missing values
    missing_before = df[numeric_cols + categorical_cols].isnull().sum().sum()
    print(f"     Missing values before cleaning: {missing_before:,}")
    
    # Specific data standardization
    cat_cols_to_standardize = ['Time_of_Day', 'Payment_Method', 'Referral_Source', 'Device_Type']
    
    for col in cat_cols_to_standardize:
        if col in df.columns:
            # Convert to lowercase string and strip whitespace
            df[col] = df[col].astype(str).str.lower().str.strip()
            # Replace string 'nan' with actual NaN
            df[col] = df[col].replace("nan", np.nan)
            
            col_lower = col.lower()
            
            # Normalize Time_of_Day
            if col_lower == "time_of_day":
                df[col] = df[col].replace({
                    r'^m[0o].*rning$': 'morning',
                    r'^aftern?[0o].*n$': 'afternoon',
                    r'^even.*g$': 'evening',
                }, regex=True)
            
            # Normalize Payment_Method
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
            
            # Normalize Device_Type
            elif col_lower == 'device_type':
                df[col] = df[col].replace({
                    r'^mob.*$': 'mobile',
                    r'^desk.*$': 'desktop',
                    r'^tab.*$': 'tablet',
                }, regex=True)
    
    # Clean numeric features
    for col in numeric_cols:
        if col in df.columns:
            # Remove extreme outliers (beyond 3 std from mean)
            if is_train:
                mean_val = df[col].mean()
                std_val = df[col].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Store bounds for test set
                cleaning_params[f'{col}_bounds'] = (lower_bound, upper_bound)
                
                # Fill missing with median
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                cleaning_params[f'{col}_median'] = median_val
            else:
                # Apply same bounds from training
                if f'{col}_bounds' in cleaning_params:
                    lower_bound, upper_bound = cleaning_params[f'{col}_bounds']
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Fill with training median
                median_val = cleaning_params.get(f'{col}_median', df[col].median())
                df[col].fillna(median_val, inplace=True)
    
    # Clean categorical features
    for col in categorical_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                # Replace rare categories with 'other'
                if is_train:
                    value_counts = df[col].value_counts()
                    rare_threshold = 0.01 * len(df)  # Categories with < 1% frequency
                    rare_categories = value_counts[value_counts < rare_threshold].index.tolist()
                    cleaning_params[f'{col}_rare'] = rare_categories
                else:
                    rare_categories = cleaning_params.get(f'{col}_rare', [])
                
                df[col] = df[col].apply(lambda x: 'other' if x in rare_categories else x)
                
                # Fill missing with mode or 'unknown'
                if is_train:
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
                    cleaning_params[f'{col}_mode'] = mode_val
                else:
                    mode_val = cleaning_params.get(f'{col}_mode', 'unknown')
                
                df[col].fillna(mode_val, inplace=True)
            else:
                # Numeric categorical (like Gender, Category)
                if is_train:
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                    cleaning_params[f'{col}_mode'] = mode_val
                else:
                    mode_val = cleaning_params.get(f'{col}_mode', 0)
                
                df[col].fillna(mode_val, inplace=True)
    
    missing_after = df[numeric_cols + categorical_cols].isnull().sum().sum()
    print(f"     Missing values after cleaning: {missing_after:,}")
    
    return df, cleaning_params

# Apply cleaning
train_cleaned, cleaning_params = clean_data(train_df, is_train=True)
test_cleaned, _ = clean_data(test_df, is_train=False, cleaning_params=cleaning_params)

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
def engineer_features(df, is_train=True, engineering_params=None):
    """Create advanced engineered features"""
    df = df.copy()
    
    if engineering_params is None:
        engineering_params = {}
    
    print(f"   {'Training' if is_train else 'Test'} set:")
    features_created = 0
    
    # 1. Price-related features
    if 'Price' in df.columns and 'Discount' in df.columns:
        # Final price after discount
        df['Final_Price'] = df['Price'] * (1 - df['Discount'] / 100)
        features_created += 1
        
        # Discount amount in euros
        df['Discount_Amount'] = df['Price'] * (df['Discount'] / 100)
        features_created += 1
        
        # Price bins (low, medium, high)
        if is_train:
            price_quantiles = df['Price'].quantile([0.33, 0.67]).values
            engineering_params['price_quantiles'] = price_quantiles
        else:
            price_quantiles = engineering_params.get('price_quantiles', [0, 1000])
        
        df['Price_Tier'] = pd.cut(df['Price'], 
                                   bins=[0] + list(price_quantiles) + [np.inf], 
                                   labels=[0, 1, 2]).astype(float)
        features_created += 1
    
    # 2. Engagement features
    if 'Engagement_Score' in df.columns:
        if is_train:
            engagement_median = df['Engagement_Score'].median()
            engineering_params['engagement_median'] = engagement_median
        else:
            engagement_median = engineering_params.get('engagement_median', 1.0)
        
        df['High_Engagement'] = (df['Engagement_Score'] > engagement_median).astype(int)
        features_created += 1
    
    if 'Email_Interaction' in df.columns and 'Engagement_Score' in df.columns:
        df['Email_Engaged'] = df['Email_Interaction'] * df['Engagement_Score']
        features_created += 1
    
    # 3. Cart-related features
    if 'Items_In_Cart' in df.columns and 'Reviews_Read' in df.columns:
        # Total activity
        df['Total_Activity'] = df['Items_In_Cart'] + df['Reviews_Read']
        features_created += 1
        
        # Cart to review ratio
        df['Cart_Review_Ratio'] = df['Items_In_Cart'] / (df['Reviews_Read'] + 1)
        features_created += 1
    
    if 'Items_In_Cart' in df.columns:
        df['Has_Items'] = (df['Items_In_Cart'] > 0).astype(int)
        features_created += 1
        
        # Cart size category
        if is_train:
            cart_median = df['Items_In_Cart'].median()
            engineering_params['cart_median'] = cart_median
        else:
            cart_median = engineering_params.get('cart_median', 0)
        
        df['Large_Cart'] = (df['Items_In_Cart'] > cart_median).astype(int)
        features_created += 1
    
    # 4. Campaign features
    if 'Campaign_Period' in df.columns:
        # Convert boolean/string to int
        if df['Campaign_Period'].dtype == 'object':
            df['Campaign_Period'] = df['Campaign_Period'].str.lower().map({'true': 1, 'false': 0, 'yes': 1, 'no': 0}).fillna(0).astype(int)
        else:
            df['Campaign_Period'] = df['Campaign_Period'].astype(int)
        
        if 'Discount' in df.columns:
            df['Campaign_Discount_Interaction'] = df['Campaign_Period'] * df['Discount']
            features_created += 1
        
        if 'Email_Interaction' in df.columns:
            df['Campaign_Email'] = df['Campaign_Period'] * df['Email_Interaction']
            features_created += 1
    
    # 5. Socioeconomic features
    if 'Socioeconomic_Status_Score' in df.columns and 'Price' in df.columns:
        # Affordability ratio
        df['Affordability'] = df['Socioeconomic_Status_Score'] / (df['Price'] + 1)
        features_created += 1
        
        if is_train:
            ses_median = df['Socioeconomic_Status_Score'].median()
            engineering_params['ses_median'] = ses_median
        else:
            ses_median = engineering_params.get('ses_median', 5.0)
        
        df['High_SES'] = (df['Socioeconomic_Status_Score'] > ses_median).astype(int)
        features_created += 1
    
    # 6. Review engagement features
    if 'Reviews_Read' in df.columns:
        df['Reads_Reviews'] = (df['Reviews_Read'] > 0).astype(int)
        features_created += 1
        
        if is_train:
            reviews_median = df['Reviews_Read'].median()
            engineering_params['reviews_median'] = reviews_median
        else:
            reviews_median = engineering_params.get('reviews_median', 2)
        
        df['Heavy_Researcher'] = (df['Reviews_Read'] > reviews_median).astype(int)
        features_created += 1
    
    # 7. Time-based features (if Time_of_Day exists)
    if 'Time_of_Day' in df.columns:
        # Standardize time values
        time_map = {'morning': 0, 'afternoon': 1, 'evening': 2}
        df['Time_of_Day_Numeric'] = df['Time_of_Day'].map(time_map).fillna(1)
        features_created += 1
    
    # 8. Discount intensity features
    if 'Discount' in df.columns:
        if is_train:
            discount_median = df['Discount'].median()
            engineering_params['discount_median'] = discount_median
        else:
            discount_median = engineering_params.get('discount_median', 15)
        
        df['High_Discount'] = (df['Discount'] > discount_median).astype(int)
        features_created += 1
        
        # Discount categories
        df['Discount_Category'] = pd.cut(df['Discount'], 
                                         bins=[0, 10, 25, 50, 100], 
                                         labels=[0, 1, 2, 3]).astype(float)
        features_created += 1
    
    # 9. Category-specific features
    if 'Category' in df.columns and 'Price' in df.columns:
        if is_train:
            category_avg_price = df.groupby('Category')['Price'].mean().to_dict()
            engineering_params['category_avg_price'] = category_avg_price
        else:
            category_avg_price = engineering_params.get('category_avg_price', {})
        
        df['Price_vs_Category_Avg'] = df.apply(
            lambda row: row['Price'] / category_avg_price.get(row['Category'], row['Price']) 
            if row['Category'] in category_avg_price else 1.0, 
            axis=1
        )
        features_created += 1
    
    # 10. Device and time interaction
    if 'Device_Type' in df.columns and 'Time_of_Day' in df.columns:
        df['Device_Time'] = df['Device_Type'].astype(str) + '_' + df['Time_of_Day'].astype(str)
        features_created += 1
    
    print(f"     Created {features_created} new features")
    
    return df, engineering_params

# Apply feature engineering
train_engineered, engineering_params = engineer_features(train_cleaned, is_train=True)
test_engineered, _ = engineer_features(test_cleaned, is_train=False, engineering_params=engineering_params)

print(f"     Total features before: {train_cleaned.shape[1]}")
print(f"     Total features after: {train_engineered.shape[1]}")

# ============================================================================
# 4. ENCODE AND PREPARE FEATURES
# ============================================================================
def prepare_features(df, is_train=True, encoders=None):
    """Encode categorical variables and prepare feature matrix"""
    df = df.copy()
    
    if encoders is None:
        encoders = {}
    
    # Columns to exclude
    exclude_cols = ['id', 'Session_ID', 'Purchase', 'Day', 
                   'AB_Bucket', 'PM_RS_Combo']  # Non-predictive identifiers
    
    # Identify categorical columns (object type or known categoricals)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    # Add known categorical columns even if numeric
    known_categoricals = ['Gender', 'Category', 'Device_Type', 'Payment_Method', 
                         'Referral_Source', 'Time_of_Day', 'Email_Interaction', 
                         'Campaign_Period', 'Device_Time']
    categorical_cols.extend([col for col in known_categoricals 
                            if col in df.columns and col not in categorical_cols])
    
    # Encode categorical variables
    if is_train:
        encoders['label_encoders'] = {}
    
    for col in categorical_cols:
        if col in df.columns:
            if is_train:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders['label_encoders'][col] = le
            else:
                le = encoders['label_encoders'].get(col)
                if le:
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    df[col] = -1
    
    # Select feature columns (exclude target and identifiers)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Fill any remaining NaN values
    X = df[feature_cols].copy()
    X = X.fillna(0)  # Fill any remaining NaNs with 0
    
    return X, feature_cols, encoders

# Prepare features
X_train_raw, feature_cols, encoders = prepare_features(train_engineered, is_train=True)
X_test_raw, _, _ = prepare_features(test_engineered, is_train=False, encoders=encoders)
y_train = train_engineered['Purchase']

print(f"   Total features prepared: {len(feature_cols)}")
print(f"   Training samples: {X_train_raw.shape[0]:,}")
print(f"   Test samples: {X_test_raw.shape[0]:,}")

# ============================================================================
# 5. FEATURE SELECTION
# ============================================================================
# Scale features first (required for proper feature selection)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Method 1: SelectKBest (univariate statistical tests)
k_best = min(30, len(feature_cols))  # Select top 30 features or all if less
selector_kbest = SelectKBest(score_func=f_classif, k=k_best)
selector_kbest.fit(X_train_scaled, y_train)

# Get feature scores
feature_scores = pd.DataFrame({
    'feature': feature_cols,
    'score': selector_kbest.scores_
}).sort_values('score', ascending=False)

print(f"     Top 15 features by F-statistic:")
for idx, row in feature_scores.head(15).iterrows():
    print(f"       {row['feature']:35s}: {row['score']:8.2f}")

# Method 2: Recursive Feature Elimination (RFE)
n_features_to_select = min(25, len(feature_cols))  # Select top 25 features
lr_for_rfe = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', 
                                 max_iter=500, random_state=42)
selector_rfe = RFE(estimator=lr_for_rfe, n_features_to_select=n_features_to_select, 
                   step=1, verbose=0)
selector_rfe.fit(X_train_scaled, y_train)

# Get selected features
selected_features_rfe = [feature_cols[i] for i in range(len(feature_cols)) 
                         if selector_rfe.support_[i]]
print(f"     Selected {len(selected_features_rfe)} features via RFE")

# Combine both methods: features selected by either method
selected_features_kbest = feature_scores.head(k_best)['feature'].tolist()
selected_features_combined = list(set(selected_features_kbest) | set(selected_features_rfe))

print(f"\n   Combined selection: {len(selected_features_combined)} features")

# Create final feature matrices with selected features
selected_indices = [i for i, col in enumerate(feature_cols) if col in selected_features_combined]
X_train_selected = X_train_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]
selected_feature_names = [feature_cols[i] for i in selected_indices]

print(f"   Final feature set: {len(selected_feature_names)} features")

# ============================================================================
# 6. TRAIN LOGISTIC REGRESSION
# ============================================================================
# Try different C values and select best
best_f1 = 0
best_model = None
best_C = 1.0

C_values = [0.1, 0.5, 1.0, 2.0, 5.0]

for C in C_values:
    lr = LogisticRegression(
        penalty='l2',
        C=C,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train_selected, y_train)
    y_pred = lr.predict(X_train_selected)
    f1 = f1_score(y_train, y_pred)
    
    print(f"     C={C:4.1f}: F1-Score = {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = lr
        best_C = C

print(f"\n   Best C: {best_C} (F1-Score: {best_f1:.4f})")

# Use best model
lr_model = best_model

# ============================================================================
# 7. EVALUATE ON TRAINING SET
# ============================================================================
y_train_pred = lr_model.predict(X_train_selected)
y_train_proba = lr_model.predict_proba(X_train_selected)[:, 1]

train_f1 = f1_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)

print(f"   F1-Score:  {train_f1:.4f}")
print(f"   Precision: {train_precision:.4f}")
print(f"   Recall:    {train_recall:.4f}")

print("\n   Confusion Matrix:")
cm = confusion_matrix(y_train, y_train_pred)
print(f"   TN: {cm[0,0]:5,}  FP: {cm[0,1]:5,}")
print(f"   FN: {cm[1,0]:5,}  TP: {cm[1,1]:5,}")

# ============================================================================
# 8. FEATURE IMPORTANCE FROM FINAL MODEL
# ============================================================================
feature_importance = pd.DataFrame({
    'feature': selected_feature_names,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("\n   Top 15 features by coefficient magnitude:")
for idx, row in feature_importance.head(15).iterrows():
    direction = "↑ Increases" if row['coefficient'] > 0 else "↓ Decreases"
    print(f"     {row['feature']:35s}: {row['coefficient']:7.4f} {direction} purchase probability")

# ============================================================================
# 9. PREDICT ON TEST SET
# ============================================================================
y_test_pred = lr_model.predict(X_test_selected)
y_test_proba = lr_model.predict_proba(X_test_selected)[:, 1]

n_purchases = y_test_pred.sum()
avg_probability = y_test_proba[y_test_pred == 1].mean() if n_purchases > 0 else 0

print(f"   Predicted purchases: {n_purchases:,} ({n_purchases/len(y_test_pred)*100:.1f}%)")
print(f"   Avg probability (predicted purchases): {avg_probability:.3f}")

# Probability distribution
print("\n   Probability distribution:")
bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']
prob_bins = pd.cut(y_test_proba, bins=bins, labels=labels)
distribution = prob_bins.value_counts().sort_index()

for bin_name, count in distribution.items():
    pct = count / len(y_test_proba) * 100
    bar = '█' * int(pct / 2)
    print(f"     {bin_name:10s}: {count:4,} ({pct:5.1f}%) {bar}")

# ============================================================================
# 10. CREATE SUBMISSION
# ============================================================================
submission = pd.DataFrame({
    'id': test_df['id'],
    'Purchase': y_test_pred
})

submission.to_csv('/Users/paolacassinelli/Desktop/Foundation of Machine Learning/dsba-m-1-challenge-purchase-prediction/logistic_regression_enhanced_submission.csv', index=False)