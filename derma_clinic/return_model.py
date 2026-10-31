#Data preprocessing
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

dataset = pd.read_csv('dermatology_clinic.csv')
X = dataset.drop(columns=['customer_id', 'returned_within_90_days'])
y = dataset['returned_within_90_days']


num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(include='object').columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

ct = ColumnTransformer(transformers=[('num', Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)])

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# print("Dataset shape:", dataset.shape)
# print("\nFirst 5 rows:\n", dataset.head())

# training the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# Evaluate the model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))



joblib.dump(ct, "clinic_preprocessor.joblib")
joblib.dump(classifier, "clinic_classifier.joblib")
