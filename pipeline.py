import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from Scikit-Learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import set_config
from sklearn.metrics import confusion_matrix

class FlagConstructor(BaseEstimator, TransformerMixin):
    def __init__(self, threshold, input_col, output_col):
        self.threshold = threshold
        self.input_col = input_col
        self.output_col = output_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.output_col] = (X[self.input_col] > self.threshold).astype(int)
        return X
        
class RatioFeatureConstructor(BaseEstimator, TransformerMixin):
    def __init__(self, ratios):
        # ratios: list of tuples of the form (numerator column, denominator column)
        self.ratios = ratios

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for num, denom in self.ratios:
            X[f"{num}_to_{denom}"] = X[num]/X[denom]

        return X


df = pd.read_csv("kidney_dataset.csv")

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("CKD_Status", axis=1),
    df["CKD_Status"],
    test_size=0.2,
    random_state=5
)

CATEGORICAL_VARS = ["Diabetes", "Hypertension", "Medication", "CKD_Status"]
NUMERICAL_VARS = ["Creatinine", "BUN", "GFR", "Urine_Output", "Age", "Protein_in_Urine", "Water_Intake"]
OBJECT_VARS = ["Medication"]
FLAG_VARS = "Urine_Output"
RATIO_VARS = [("Protein_in_Urine", "Creatinine"), ("BUN", "Creatinine")]

set_config(transform_output="pandas")


feature_engineering = Pipeline([
    ("high_urine_output_flag",
     FlagConstructor(
         threshold=1300,
         input_col="Urine_Output",
         output_col="High_Urine_Output_Flag"
     )),
    
    ("ratio_features",
     RatioFeatureConstructor([
         ("Protein_in_Urine", "Creatinine"),
         ("BUN", "Creatinine")
     ]))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat",
         Pipeline([
             ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
             ("encoder", OneHotEncoder(sparse_output=False))
         ]),
         OBJECT_VARS)
])

pipe = Pipeline([
    ("feature_engineering", feature_engineering),
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(random_state=5))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
