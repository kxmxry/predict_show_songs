import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

# Read the cleaned dataset
df = pd.read_csv('dataset/clean_dataset.csv')

# Select the specified features
selected_features = [
    'בוצע בהופעה "Live קיסריה 2022"',
    'בוצע בהופעה "מנורה 2023"',
    'צפיות ביוטיוב',
    'האם סינגל'
]

# Prepare features and target
X = df[selected_features]
y = df['בוצע ב"מופע העשור 2024"']  # Target variable

# Print information about missing values
print("\nMissing values in original dataset:")
print(X.isna().sum())

# Option 1: Remove rows with NaN values
X_clean = X.dropna()
y_clean = y[X_clean.index]

print("\nNumber of rows after removing NaN values:", len(X_clean))
if len(X_clean) < 57:  # 45 + 12
    print("Warning: Not enough samples after removing NaN values. Using imputation instead.")
    # Use imputation instead
    imputer = SimpleImputer(strategy='median')  # Using median as default strategy
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Split the imputed dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y,
        train_size=45,
        test_size=12,
        random_state=42,
        stratify=y
    )
    
    # Save the imputed splits
    X_train.to_csv('dataset/X_train_median.csv', index=False, encoding='utf-8-sig')
    X_test.to_csv('dataset/X_test_median.csv', index=False, encoding='utf-8-sig')
    y_train.to_csv('dataset/y_train_median.csv', index=False, encoding='utf-8-sig')
    y_test.to_csv('dataset/y_test_median.csv', index=False, encoding='utf-8-sig')
    
    print("\nDataset shapes after imputation:")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("\nTarget distribution in training set:")
    print(y_train.value_counts(normalize=True))
    print("\nTarget distribution in test set:")
    print(y_test.value_counts(normalize=True))
else:
    # Split the cleaned dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, 
        train_size=45, 
        test_size=12,
        random_state=42,
        stratify=y_clean
    )
    
    # Save the cleaned splits
    X_train.to_csv('dataset/X_train_clean.csv', index=False, encoding='utf-8-sig')
    X_test.to_csv('dataset/X_test_clean.csv', index=False, encoding='utf-8-sig')
    y_train.to_csv('dataset/y_train_clean.csv', index=False, encoding='utf-8-sig')
    y_test.to_csv('dataset/y_test_clean.csv', index=False, encoding='utf-8-sig')
    
    print("\nDataset shapes after cleaning:")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("\nTarget distribution in training set:")
    print(y_train.value_counts(normalize=True))
    print("\nTarget distribution in test set:")
    print(y_test.value_counts(normalize=True)) 