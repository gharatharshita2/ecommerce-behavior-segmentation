import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from sklearn.preprocessing import StandardScaler

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

loader = DataLoader('data/ecommerce.db')
df = loader.load_table('sessions')

fe = FeatureEngineer(df)
df = (fe.add_total_pages()
        .add_total_duration()
        .add_product_focus_ratio()
        .get_dataframe())

features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
            'TotalPages', 'TotalDuration', 'ProductFocusRatio']

X = df[features]
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Training set size:', X_train.shape)
print('Test set size:', X_test.shape)
print()
print('Conversion rate in training set:', round(y_train.mean() * 100, 2), '%')
print('Conversion rate in test set:', round(y_test.mean() * 100, 2), '%')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)

print()
print('Logistic Regression Results:')
print('Accuracy: ', round(accuracy_score(y_test, y_pred), 3))
print('Precision:', round(precision_score(y_test, y_pred), 3))
print('Recall:   ', round(recall_score(y_test, y_pred), 3))
print('F1 Score: ', round(f1_score(y_test, y_pred), 3))
print()
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

print()
print('Random Forest Results:')
print('Accuracy: ', round(accuracy_score(y_test, y_pred_rf), 3))
print('Precision:', round(precision_score(y_test, y_pred_rf), 3))
print('Recall:   ', round(recall_score(y_test, y_pred_rf), 3))
print('F1 Score: ', round(f1_score(y_test, y_pred_rf), 3))
print()
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_rf))

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print()
print('Best parameters:', grid_search.best_params_)
print('Best cross-validated F1 score:', round(grid_search.best_score_, 3))

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)

print()
print('Tuned Random Forest Results (on test set):')
print('Accuracy: ', round(accuracy_score(y_test, y_pred_best), 3))
print('Precision:', round(precision_score(y_test, y_pred_best), 3))
print('Recall:   ', round(recall_score(y_test, y_pred_best), 3))
print('F1 Score: ', round(f1_score(y_test, y_pred_best), 3))

import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print()
print('Saved model.pkl and scaler.pkl')