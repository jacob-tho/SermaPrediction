import numpy as np
import pandas as pd

"""
NaN-Werte Behandeln! 
Domänenwissen (kann man Dinge korrekt auffüllen?)
Sind NaN-Werte auch schon aussagekräftig?
Korellationsfilter / VarianceTreshold 
MultipleImputation??

Numerische Features: t-Test / Mann-Whitney U (Univariate Tests)
Kategorische Features: Chi^2 oder Fisher Test 
Punktbiseriale Korrelation, Cramers V

Logistic Regression (L1, ElasticNet)
Mehrfaches Bootstraping
RandomForest oder XGBoost

Cross-Validation
Regularisierung 
PCA
Permutation Importance
SHAP-Werte 

"""

df = pd.read_excel("Serma_Pilotstudie.xlsx")

pd.options.display.max_columns = None
print(df.shape)
print(list(df.columns))
print(df.isna().mean().sort_values(ascending=False).head(20))

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



y = df["Serom_postoperativ"]
#true = df[df["Serom_postoperativ"] == 1]
#false = df[df["Serom_postoperativ"] == 0]
X = df.drop(columns=["Serom_postoperativ"])

numerical_columns = X.select_dtypes(include = np.number).columns
catergorical_columns = X.select_dtypes(exclude = np.number).columns

numeric_transformer = Pipeline([
    ("imputer", IterativeImputer(random_state=42)),
    ("scaler", StandardScaler())
    ])


categorical_transformer = Pipeline([
    ("imputer", IterativeImputer(initial_strategy = "most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numerical_columns),
    ("categorical", categorical_transformer, catergorical_columns)
    ])

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

models = {
        "Random Forest": RandomForestClassifier(n_estimators = 300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42)
        }
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
        ])

    scores = cross_val_score(pipe, X, y, cv=cv, scoring = "roc_auc")
    print(f"{name}: AUC = {scores.mean():.3f} +- {scores.std():.3f}")

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

rf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=500, random_state=42))
])

rf.fit(X, y)
result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

importances = pd.Series(result.importances_mean, index=rf.named_steps["preprocess"].get_feature_names_out())
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8,5))
top_features.plot(kind="barh")
plt.title("Feature Importance (Permutation)")

'''
import shap
explainer = shap.TreeExplainer(rf.named_steps["model"])
shap_values = explainer.shap_values(rf.named_steps["preprocess"].transform(X))
shap.summary_plot(shap_values[1], features=rf.named_steps["preprocess"].transform(X))
'''

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

plt.show()


"""
df = df.loc[:, df.isna().mean() <= 0.7]
print(df.shape)
print(df["Seite"])
"""
