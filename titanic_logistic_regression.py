"""
LOGISTIC REGRESSION: TITANIC SURVIVAL PREDICTION
=================================================

Objective: Predict passenger survival on the Titanic using Logistic Regression

Dataset: Titanic passenger data (train and test sets)
Target: Survived (0 = No, 1 = Yes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 80)
print("LOGISTIC REGRESSION: TITANIC SURVIVAL PREDICTION")
print("=" * 80)

# ============================================================================
# 1. DATA EXPLORATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: DATA EXPLORATION AND EDA")
print("=" * 80)

# Load datasets
train_df = pd.read_csv('Titanic_train.csv')
test_df = pd.read_csv('Titanic_test.csv')

print(f"\nTraining Data Shape: {train_df.shape}")
print(f"Testing Data Shape: {test_df.shape}")

print("\n\nFirst 5 rows of training data:")
print(train_df.head())

print("\n\nDataset Info:")
print(train_df.info())

print("\n\nFeature Descriptions:")
feature_desc = {
    'PassengerId': 'Unique ID for each passenger',
    'Survived': 'Survival (0 = No, 1 = Yes) - TARGET VARIABLE',
    'Pclass': 'Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)',
    'Name': 'Passenger name',
    'Sex': 'Gender',
    'Age': 'Age in years',
    'SibSp': 'Number of siblings/spouses aboard',
    'Parch': 'Number of parents/children aboard',
    'Ticket': 'Ticket number',
    'Fare': 'Passenger fare',
    'Cabin': 'Cabin number',
    'Embarked': 'Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)'
}

for feature, desc in feature_desc.items():
    if feature in train_df.columns:
        print(f"  {feature}: {desc}")

print("\n\nSummary Statistics:")
print(train_df.describe())

print("\n\nCategorical Features Summary:")
print(train_df.describe(include='object'))

print("\n\nMissing Values:")
missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing[missing > 0],
    'Percentage': missing_pct[missing > 0]
})
print(missing_df)

print("\n\nTarget Variable Distribution:")
survival_counts = train_df['Survived'].value_counts()
print(survival_counts)
print(f"\nSurvival Rate: {train_df['Survived'].mean()*100:.2f}%")

# ============================================================================
# VISUALIZATION 1: EDA - Distributions and Relationships
# ============================================================================
print("\n\nCreating EDA visualizations...")

fig = plt.figure(figsize=(20, 16))

# 1. Survival Distribution
ax1 = plt.subplot(4, 3, 1)
survival_counts.plot(kind='bar', color=['red', 'green'], alpha=0.7, edgecolor='black')
ax1.set_title('Survival Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Survived (0=No, 1=Yes)', fontsize=10)
ax1.set_ylabel('Count', fontsize=10)
ax1.set_xticklabels(['Died', 'Survived'], rotation=0)
for i, v in enumerate(survival_counts.values):
    ax1.text(i, v + 10, str(v), ha='center', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Gender Distribution
ax2 = plt.subplot(4, 3, 2)
train_df['Sex'].value_counts().plot(kind='bar', color=['steelblue', 'pink'], 
                                     alpha=0.7, edgecolor='black')
ax2.set_title('Gender Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Gender', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.set_xticklabels(['Male', 'Female'], rotation=0)
ax2.grid(True, alpha=0.3)

# 3. Class Distribution
ax3 = plt.subplot(4, 3, 3)
train_df['Pclass'].value_counts().sort_index().plot(kind='bar', 
                                                     color=['gold', 'silver', 'brown'],
                                                     alpha=0.7, edgecolor='black')
ax3.set_title('Passenger Class Distribution', fontsize=12, fontweight='bold')
ax3.set_xlabel('Class', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
ax3.grid(True, alpha=0.3)

# 4. Age Distribution
ax4 = plt.subplot(4, 3, 4)
ax4.hist(train_df['Age'].dropna(), bins=30, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(train_df['Age'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax4.axvline(train_df['Age'].median(), color='green', linestyle='--', linewidth=2, label='Median')
ax4.set_title('Age Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Age (years)', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Fare Distribution
ax5 = plt.subplot(4, 3, 5)
ax5.hist(train_df['Fare'], bins=50, color='orange', alpha=0.7, edgecolor='black')
ax5.set_title('Fare Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('Fare ($)', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. Embarkation Port
ax6 = plt.subplot(4, 3, 6)
train_df['Embarked'].value_counts().plot(kind='bar', color=['cyan', 'magenta', 'yellow'],
                                         alpha=0.7, edgecolor='black')
ax6.set_title('Embarkation Port Distribution', fontsize=12, fontweight='bold')
ax6.set_xlabel('Port', fontsize=10)
ax6.set_ylabel('Count', fontsize=10)
ax6.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'], rotation=0)
ax6.grid(True, alpha=0.3)

# 7. Survival by Gender
ax7 = plt.subplot(4, 3, 7)
survival_sex = pd.crosstab(train_df['Sex'], train_df['Survived'])
survival_sex.plot(kind='bar', ax=ax7, color=['red', 'green'], alpha=0.7, edgecolor='black')
ax7.set_title('Survival by Gender', fontsize=12, fontweight='bold')
ax7.set_xlabel('Gender', fontsize=10)
ax7.set_ylabel('Count', fontsize=10)
ax7.set_xticklabels(['Female', 'Male'], rotation=0)
ax7.legend(['Died', 'Survived'])
ax7.grid(True, alpha=0.3)

# 8. Survival by Class
ax8 = plt.subplot(4, 3, 8)
survival_class = pd.crosstab(train_df['Pclass'], train_df['Survived'])
survival_class.plot(kind='bar', ax=ax8, color=['red', 'green'], alpha=0.7, edgecolor='black')
ax8.set_title('Survival by Passenger Class', fontsize=12, fontweight='bold')
ax8.set_xlabel('Class', fontsize=10)
ax8.set_ylabel('Count', fontsize=10)
ax8.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
ax8.legend(['Died', 'Survived'])
ax8.grid(True, alpha=0.3)

# 9. Survival Rate by Gender (%)
ax9 = plt.subplot(4, 3, 9)
survival_rate_sex = train_df.groupby('Sex')['Survived'].mean() * 100
survival_rate_sex.plot(kind='bar', ax=ax9, color=['lightblue', 'lightpink'],
                       alpha=0.7, edgecolor='black')
ax9.set_title('Survival Rate by Gender (%)', fontsize=12, fontweight='bold')
ax9.set_xlabel('Gender', fontsize=10)
ax9.set_ylabel('Survival Rate (%)', fontsize=10)
ax9.set_xticklabels(['Female', 'Male'], rotation=0)
ax9.axhline(y=train_df['Survived'].mean()*100, color='red', linestyle='--', 
            label=f'Overall: {train_df["Survived"].mean()*100:.1f}%')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 10. Survival Rate by Class (%)
ax10 = plt.subplot(4, 3, 10)
survival_rate_class = train_df.groupby('Pclass')['Survived'].mean() * 100
survival_rate_class.plot(kind='bar', ax=ax10, color=['gold', 'silver', 'brown'],
                         alpha=0.7, edgecolor='black')
ax10.set_title('Survival Rate by Class (%)', fontsize=12, fontweight='bold')
ax10.set_xlabel('Class', fontsize=10)
ax10.set_ylabel('Survival Rate (%)', fontsize=10)
ax10.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
ax10.axhline(y=train_df['Survived'].mean()*100, color='red', linestyle='--',
             label=f'Overall: {train_df["Survived"].mean()*100:.1f}%')
ax10.legend()
ax10.grid(True, alpha=0.3)

# 11. Age vs Fare scatter
ax11 = plt.subplot(4, 3, 11)
survived = train_df[train_df['Survived'] == 1]
died = train_df[train_df['Survived'] == 0]
ax11.scatter(died['Age'], died['Fare'], c='red', alpha=0.5, s=30, label='Died', edgecolors='black')
ax11.scatter(survived['Age'], survived['Fare'], c='green', alpha=0.5, s=30, label='Survived', edgecolors='black')
ax11.set_title('Age vs Fare (by Survival)', fontsize=12, fontweight='bold')
ax11.set_xlabel('Age (years)', fontsize=10)
ax11.set_ylabel('Fare ($)', fontsize=10)
ax11.legend()
ax11.grid(True, alpha=0.3)

# 12. Correlation Heatmap
ax12 = plt.subplot(4, 3, 12)
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_matrix = train_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax12)
ax12.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

plt.suptitle('TITANIC DATASET: EXPLORATORY DATA ANALYSIS', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('titanic_eda.png', dpi=300, bbox_inches='tight')
print("✓ Saved: titanic_eda.png")
plt.close()

# Key insights
print("\n\n" + "="*60)
print("KEY INSIGHTS FROM EDA:")
print("="*60)
print(f"\n1. SURVIVAL RATE:")
print(f"   Overall: {train_df['Survived'].mean()*100:.1f}%")
print(f"   Female: {train_df[train_df['Sex']=='female']['Survived'].mean()*100:.1f}%")
print(f"   Male: {train_df[train_df['Sex']=='male']['Survived'].mean()*100:.1f}%")

print(f"\n2. CLASS IMPACT:")
for pclass in [1, 2, 3]:
    rate = train_df[train_df['Pclass']==pclass]['Survived'].mean()*100
    print(f"   Class {pclass}: {rate:.1f}% survival rate")

print(f"\n3. FAMILY SIZE:")
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
print(f"   Avg family size: {train_df['FamilySize'].mean():.2f}")
print(f"   Solo travelers: {len(train_df[train_df['FamilySize']==1])} ({len(train_df[train_df['FamilySize']==1])/len(train_df)*100:.1f}%)")

print(f"\n4. AGE:")
print(f"   Average age: {train_df['Age'].mean():.1f} years")
print(f"   Children (<18): {len(train_df[train_df['Age']<18])} passengers")
print(f"   Missing age values: {train_df['Age'].isnull().sum()} ({train_df['Age'].isnull().sum()/len(train_df)*100:.1f}%)")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA PREPROCESSING")
print("=" * 80)

# Combine for preprocessing
print("\nCombining train and test for consistent preprocessing...")
train_df['Dataset'] = 'Train'
test_df['Dataset'] = 'Test'
combined = pd.concat([train_df, test_df], axis=0, sort=False)

print(f"Combined dataset shape: {combined.shape}")

# Feature Engineering
print("\n\n--- FEATURE ENGINEERING ---")

# 1. Family Size
combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
print(f"✓ Created FamilySize: SibSp + Parch + 1")

# 2. Is Alone
combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
print(f"✓ Created IsAlone: 1 if FamilySize==1, else 0")

# 3. Title from Name
combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(f"✓ Extracted Title from Name")
print(f"  Unique titles: {combined['Title'].nunique()}")
print(f"  Top titles: {combined['Title'].value_counts().head()}")

# Group rare titles
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
    'Capt': 'Rare', 'Sir': 'Rare'
}
combined['Title'] = combined['Title'].map(title_mapping)
combined['Title'].fillna('Rare', inplace=True)
print(f"✓ Grouped rare titles")

# 4. Age Bands
combined['AgeBand'] = pd.cut(combined['Age'], bins=[0, 12, 18, 35, 60, 100],
                             labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# 5. Fare Bands
combined['FareBand'] = pd.qcut(combined['Fare'].fillna(combined['Fare'].median()), 
                               q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'],
                               duplicates='drop')

print("\n\n--- HANDLING MISSING VALUES ---")

# Check missing values
missing_before = combined.isnull().sum()
print(f"Missing values before imputation:")
print(missing_before[missing_before > 0])

# 1. Age - Fill with median by Pclass and Sex
print("\n✓ Filling Age with median by Pclass and Sex...")
combined['Age'] = combined.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)

# 2. Embarked - Fill with mode
print("✓ Filling Embarked with mode (S)...")
combined['Embarked'].fillna(combined['Embarked'].mode()[0], inplace=True)

# 3. Fare - Fill with median
print("✓ Filling Fare with median...")
combined['Fare'].fillna(combined['Fare'].median(), inplace=True)

# 4. Cabin - Create binary feature
print("✓ Creating HasCabin binary feature...")
combined['HasCabin'] = combined['Cabin'].notna().astype(int)

# Check missing values after
missing_after = combined.isnull().sum()
print(f"\nMissing values after imputation:")
print(missing_after[missing_after > 0])

print("\n\n--- ENCODING CATEGORICAL VARIABLES ---")

# Label Encoding for binary features
le = LabelEncoder()
combined['Sex'] = le.fit_transform(combined['Sex'])
print(f"✓ Encoded Sex: female=0, male=1")

# One-hot encoding for multi-category features
print("✓ One-hot encoding: Embarked, Title, AgeBand, FareBand...")
combined = pd.get_dummies(combined, columns=['Embarked', 'Title', 'AgeBand', 'FareBand'],
                         drop_first=False)

print(f"\nDataset shape after encoding: {combined.shape}")

# Select features for modeling
feature_columns = [col for col in combined.columns if col not in 
                  ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Dataset']]

print(f"\nFeatures selected for modeling ({len(feature_columns)}):")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")

# Split back to train and test
train_processed = combined[combined['Dataset'] == 'Train'].copy()
test_processed = combined[combined['Dataset'] == 'Test'].copy()

print(f"\nProcessed train shape: {train_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")

# Prepare X and y
X = train_processed[feature_columns]
y = train_processed['Survived']

print(f"\nX (features) shape: {X.shape}")
print(f"y (target) shape: {y.shape}")

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                   random_state=42, stratify=y)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# Feature Scaling
print("\n\n--- FEATURE SCALING ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("✓ Features standardized using StandardScaler")
print(f"  Mean after scaling: {X_train_scaled.mean():.10f}")
print(f"  Std after scaling: {X_train_scaled.std():.10f}")

# ============================================================================
# 3. MODEL BUILDING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: MODEL BUILDING")
print("=" * 80)

print("\nBuilding Logistic Regression model...")
print("Parameters: solver='lbfgs', max_iter=1000, random_state=42")

# Build model
logreg = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)

# Train model
print("\nTraining model...")
logreg.fit(X_train_scaled, y_train)

print("✓ Model trained successfully!")

# Model coefficients
print("\n\n--- MODEL COEFFICIENTS ---")
coef_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': logreg.coef_[0],
    'Abs_Coefficient': np.abs(logreg.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(coef_df.head(10).to_string(index=False))

print(f"\nIntercept: {logreg.intercept_[0]:.4f}")

# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: MODEL EVALUATION")
print("=" * 80)

# Predictions
y_train_pred = logreg.predict(X_train_scaled)
y_val_pred = logreg.predict(X_val_scaled)

y_train_pred_proba = logreg.predict_proba(X_train_scaled)[:, 1]
y_val_pred_proba = logreg.predict_proba(X_val_scaled)[:, 1]

# Calculate metrics
print("\n--- PERFORMANCE METRICS ---")

# Training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)

print("\nTRAINING SET:")
print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")
print(f"  ROC-AUC:   {train_roc_auc:.4f}")

# Validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)

print("\nVALIDATION SET:")
print(f"  Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall:    {val_recall:.4f}")
print(f"  F1-Score:  {val_f1:.4f}")
print(f"  ROC-AUC:   {val_roc_auc:.4f}")

# Cross-validation
print("\n\n--- CROSS-VALIDATION (5-FOLD) ---")
cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Classification Report
print("\n\n--- DETAILED CLASSIFICATION REPORT (Validation Set) ---")
print(classification_report(y_val, y_val_pred, target_names=['Died', 'Survived']))

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
print("\n--- CONFUSION MATRIX (Validation Set) ---")
print(cm)
print(f"\nTrue Negatives (TN):  {cm[0,0]} (Correctly predicted as Died)")
print(f"False Positives (FP): {cm[0,1]} (Incorrectly predicted as Survived)")
print(f"False Negatives (FN): {cm[1,0]} (Incorrectly predicted as Died)")
print(f"True Positives (TP):  {cm[1,1]} (Correctly predicted as Survived)")

# ============================================================================
# VISUALIZATION 2: Model Performance
# ============================================================================
print("\n\nCreating model performance visualizations...")

fig = plt.figure(figsize=(18, 12))

# 1. Feature Importance (Coefficients)
ax1 = plt.subplot(2, 3, 1)
top_features = coef_df.head(15)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
ax1.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['Feature'])
ax1.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
ax1.set_title('Top 15 Feature Coefficients\n(Green=Increases survival, Red=Decreases survival)', 
             fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax1.grid(True, alpha=0.3, axis='x')

# 2. Confusion Matrix Heatmap
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
           cbar_kws={"shrink": 0.8}, ax=ax2,
           xticklabels=['Died (0)', 'Survived (1)'],
           yticklabels=['Died (0)', 'Survived (1)'])
ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax2.set_title('Confusion Matrix (Validation Set)', fontsize=12, fontweight='bold')

# 3. ROC Curve
ax3 = plt.subplot(2, 3, 3)
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_proba)

ax3.plot(fpr_train, tpr_train, 'b-', linewidth=2, label=f'Training (AUC={train_roc_auc:.3f})')
ax3.plot(fpr_val, tpr_val, 'r-', linewidth=2, label=f'Validation (AUC={val_roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC=0.5)')
ax3.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax3.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax3.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# 4. Metrics Comparison
ax4 = plt.subplot(2, 3, 4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
train_metrics = [train_accuracy, train_precision, train_recall, train_f1, train_roc_auc]
val_metrics = [val_accuracy, val_precision, val_recall, val_f1, val_roc_auc]

x = np.arange(len(metrics))
width = 0.35

ax4.bar(x - width/2, train_metrics, width, label='Training', alpha=0.8, 
       color='steelblue', edgecolor='black')
ax4.bar(x + width/2, val_metrics, width, label='Validation', alpha=0.8,
       color='coral', edgecolor='black')

ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
ax4.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, rotation=15)
ax4.legend()
ax4.set_ylim(0, 1.1)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (train_val, val_val) in enumerate(zip(train_metrics, val_metrics)):
    ax4.text(i - width/2, train_val + 0.02, f'{train_val:.3f}', 
            ha='center', fontsize=9, fontweight='bold')
    ax4.text(i + width/2, val_val + 0.02, f'{val_val:.3f}',
            ha='center', fontsize=9, fontweight='bold')

# 5. Predicted Probability Distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(y_val_pred_proba[y_val==0], bins=30, alpha=0.7, color='red',
        edgecolor='black', label='Actually Died')
ax5.hist(y_val_pred_proba[y_val==1], bins=30, alpha=0.7, color='green',
        edgecolor='black', label='Actually Survived')
ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax5.set_xlabel('Predicted Survival Probability', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Predicted Probability Distribution', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Cross-Validation Scores
ax6 = plt.subplot(2, 3, 6)
ax6.bar(range(1, 6), cv_scores, color='purple', alpha=0.7, edgecolor='black')
ax6.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {cv_scores.mean():.4f}')
ax6.set_xlabel('Fold', fontsize=11, fontweight='bold')
ax6.set_ylabel('Accuracy Score', fontsize=11, fontweight='bold')
ax6.set_title('5-Fold Cross-Validation Scores', fontsize=12, fontweight='bold')
ax6.set_xticks(range(1, 6))
ax6.legend()
ax6.set_ylim(0.7, 0.9)
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, score in enumerate(cv_scores, 1):
    ax6.text(i, score + 0.005, f'{score:.4f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('LOGISTIC REGRESSION: MODEL PERFORMANCE EVALUATION', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('titanic_model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: titanic_model_performance.png")
plt.close()

# ============================================================================
# 5. INTERPRETATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: MODEL INTERPRETATION")
print("=" * 80)

print("\n--- COEFFICIENT INTERPRETATION ---")
print("\nLogistic Regression Equation:")
print("log(p/(1-p)) = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ")
print("\nWhere:")
print("  p = probability of survival")
print("  β₀ = intercept")
print("  βᵢ = coefficient for feature i")
print("  Xᵢ = value of feature i")

print(f"\n\nTop 10 Features INCREASING Survival Probability:")
positive_coef = coef_df[coef_df['Coefficient'] > 0].head(10)
for idx, row in positive_coef.iterrows():
    odds_ratio = np.exp(row['Coefficient'])
    print(f"  • {row['Feature']}: β={row['Coefficient']:.4f}")
    print(f"      → Odds ratio: {odds_ratio:.4f} (increases odds by {(odds_ratio-1)*100:.1f}%)")

print(f"\n\nTop 10 Features DECREASING Survival Probability:")
negative_coef = coef_df[coef_df['Coefficient'] < 0].head(10)
for idx, row in negative_coef.iterrows():
    odds_ratio = np.exp(row['Coefficient'])
    print(f"  • {row['Feature']}: β={row['Coefficient']:.4f}")
    print(f"      → Odds ratio: {odds_ratio:.4f} (decreases odds by {(1-odds_ratio)*100:.1f}%)")

print("\n\n--- FEATURE SIGNIFICANCE ---")
print("\nMost significant factors for survival:")
print("\n1. GENDER (Sex):")
print("   • Being female DRAMATICALLY increases survival chances")
print("   • This reflects the 'women and children first' evacuation policy")

print("\n2. PASSENGER CLASS (Pclass):")
print("   • 1st class passengers had much higher survival rates")
print("   • 3rd class passengers had significantly lower survival rates")
print("   • Reflects both proximity to lifeboats and social privilege")

print("\n3. TITLE (Social Status):")
print("   • Titles like 'Mrs' and 'Miss' (women) show positive impact")
print("   • Title 'Mr' shows negative impact")
print("   • Master (young boys) shows positive impact")

print("\n4. FAMILY RELATIONSHIPS:")
print("   • Being alone vs having family shows mixed effects")
print("   • Moderate family sizes show better survival than very large families")

print("\n5. AGE:")
print("   • Children had better survival rates")
print("   • Different age bands show varying survival probabilities")

print("\n6. FARE:")
print("   • Higher fares correlate with better survival")
print("   • Indirectly captures both wealth and cabin location")

# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODEL AND RESULTS")
print("=" * 80)

import pickle

# Save model
with open('titanic_logreg_model.pkl', 'wb') as f:
    pickle.dump(logreg, f)
print("\n✓ Saved model: titanic_logreg_model.pkl")

# Save scaler
with open('titanic_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved scaler: titanic_scaler.pkl")

# Save feature names
with open('titanic_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("✓ Saved feature names: titanic_feature_names.pkl")

# Save results
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Training': [train_accuracy, train_precision, train_recall, train_f1, train_roc_auc],
    'Validation': [val_accuracy, val_precision, val_recall, val_f1, val_roc_auc]
})
results_df.to_csv('titanic_model_results.csv', index=False)
print("✓ Saved results: titanic_model_results.csv")

# Save feature importance
coef_df.to_csv('titanic_feature_importance.csv', index=False)
print("✓ Saved feature importance: titanic_feature_importance.csv")

# ============================================================================
# INTERVIEW QUESTIONS ANSWERS
# ============================================================================
print("\n" + "=" * 80)
print("INTERVIEW QUESTIONS - ANSWERS")
print("=" * 80)

print("""
QUESTION 1: What is the difference between precision and recall?
-----------------------------------------------------------------

PRECISION: "Of all passengers we predicted would survive, how many actually did?"
  Formula: TP / (TP + FP)
  
  Example: If we predicted 100 people would survive, and 80 actually did:
    Precision = 80/100 = 0.80 (80%)
  
  HIGH PRECISION = Few false alarms
  Use when: False positives are costly
  Example: Medical diagnosis - don't want to wrongly tell someone they're sick

RECALL: "Of all passengers who actually survived, how many did we correctly identify?"
  Formula: TP / (TP + FN)
  
  Example: If 90 people actually survived, and we correctly identified 80:
    Recall = 80/90 = 0.89 (89%)
  
  HIGH RECALL = Don't miss the positive cases
  Use when: False negatives are costly
  Example: Disease screening - don't want to miss anyone who is sick

REAL-WORLD TITANIC EXAMPLE:
---------------------------
In our model:
  Validation Precision: {val_precision:.4f}
    → Of passengers we predicted would survive, {val_precision*100:.1f}% actually did
  
  Validation Recall: {val_recall:.4f}
    → Of passengers who actually survived, we correctly identified {val_recall*100:.1f}%

TRADE-OFF:
----------
• You can't always maximize both simultaneously
• If you predict more people survive (liberal threshold):
    → Higher recall (catch more survivors)
    → Lower precision (more false alarms)
  
• If you predict fewer people survive (conservative threshold):
    → Higher precision (fewer false alarms)
    → Lower recall (miss some survivors)

F1-SCORE: Harmonic mean of precision and recall
  Balances both metrics: F1 = 2 × (Precision × Recall) / (Precision + Recall)
  Our F1-Score: {val_f1:.4f}


QUESTION 2: What is cross-validation, and why is it important in binary classification?
----------------------------------------------------------------------------------------

CROSS-VALIDATION: A technique to evaluate model performance more reliably

HOW IT WORKS (K-FOLD):
1. Split data into K equal parts (e.g., 5 folds)
2. Train on K-1 folds, test on the remaining fold
3. Repeat K times, each time using a different fold for testing
4. Average the K performance scores

EXAMPLE WITH 5-FOLD CV:
  Iteration 1: Train on [1,2,3,4], Test on [5] → Score: 0.82
  Iteration 2: Train on [1,2,3,5], Test on [4] → Score: 0.79
  Iteration 3: Train on [1,2,4,5], Test on [3] → Score: 0.81
  Iteration 4: Train on [1,3,4,5], Test on [2] → Score: 0.80
  Iteration 5: Train on [2,3,4,5], Test on [1] → Score: 0.83
  
  Mean CV Score: 0.81 (±0.015)

OUR TITANIC MODEL:
  CV Scores: {cv_scores}
  Mean: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})

WHY IT'S IMPORTANT FOR BINARY CLASSIFICATION:
----------------------------------------------

1. MORE RELIABLE PERFORMANCE ESTIMATE:
   • Single train-test split can be lucky/unlucky
   • CV gives average performance across multiple splits
   • Reduces variance in performance estimation

2. DETECT OVERFITTING:
   • If training score >> CV score → overfitting
   • If both are similar → good generalization
   • Our model: Training={train_accuracy:.4f}, CV={cv_scores.mean():.4f} → Good!

3. CLASS IMBALANCE HANDLING:
   • Binary classification often has imbalanced classes
   • Stratified K-fold ensures each fold has same class ratio
   • Prevents biased evaluation from one class dominating

4. BETTER USE OF LIMITED DATA:
   • Especially important with smaller datasets (like Titanic: 891 samples)
   • Every data point gets to be in test set exactly once
   • No data is "wasted" on single test set

5. MODEL SELECTION:
   • Compare different models using same CV protocol
   • Choose model with best mean CV score
   • More robust than single-split comparison

6. HYPERPARAMETER TUNING:
   • Grid search with CV finds best parameters
   • Tests each parameter combination on multiple folds
   • Reduces risk of overfitting to validation set

BEST PRACTICES:
---------------
✓ Use stratified k-fold for classification (maintains class balance)
✓ Use 5 or 10 folds (good bias-variance trade-off)
✓ Report mean ± std deviation of CV scores
✓ Use same CV splits when comparing models
✓ Don't tune on test set - use nested CV for hyperparameters
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

print(f"""
SUMMARY:
--------
✓ Dataset: Titanic (891 training samples)
✓ Target: Survival prediction (binary classification)
✓ Algorithm: Logistic Regression
✓ Features: {len(feature_columns)} engineered features
✓ Validation Accuracy: {val_accuracy*100:.2f}%
✓ ROC-AUC Score: {val_roc_auc:.4f}
✓ Cross-Validation: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})

TOP INSIGHTS:
-------------
• Gender is the strongest predictor (female = much higher survival)
• Passenger class significantly impacts survival (1st > 2nd > 3rd)
• Children had better survival rates (women & children first policy)
• Higher fare (proxy for wealth) correlates with survival
• Family size shows non-linear relationship with survival

FILES GENERATED:
----------------
1. titanic_eda.png - Comprehensive exploratory data analysis
2. titanic_model_performance.png - Model evaluation visualizations
3. titanic_logreg_model.pkl - Trained model (for deployment)
4. titanic_scaler.pkl - Feature scaler (for deployment)
5. titanic_feature_names.pkl - Feature list (for deployment)
6. titanic_model_results.csv - Performance metrics summary
7. titanic_feature_importance.csv - Feature coefficients & importance

NEXT STEP: Deploy with Streamlit!
""")

print("\n" + "=" * 80)
