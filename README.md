# Regression with a Crab Age Dataset


## Objective
This project aims to predict the age of crabs using regression models based on the "Crab Age" dataset. Various regression models and their combinations were employed in the project.

## About the Dataset
The dataset contains 15,000 entries and 9 columns:

- `Sex`: The sex of the crab (object)
- `Length`: The length of the crab (mm, float64)
- `Diameter`: The diameter of the crab (mm, float64)
- `Height`: The height of the crab (mm, float64)
- `Weight`: The weight of the crab (g, float64)
- `Shucked Weight`: The weight of the crab meat (g, float64)
- `Viscera Weight`: The weight of the viscera (g, float64)
- `Shell Weight`: The weight of the shell (g, float64)
- `Age`: The age of the crab (years, float64)

The dataset's memory usage is 1.7 MB.

## Libraries Used
The following Python libraries were used in the project:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor, Ridge, ElasticNet
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
```

## Feature Engineering
The project involved calculating various measurements of the crabs and creating new features using the following formulas and functions:

- **Viscera Ratio** = Viscera Weight / Weight
- **Shell-to-body Ratio** = Shell Weight / Weight
- **Meat Yield** = Shucked Weight / (Weight + Shell Weight)
- **Length-to-Diameter Ratio** = Length / Diameter
- **Weight-to-VisceraWeight Ratio** = Weight / Viscera Weight
- **Weight-to-ShellWeight Ratio** = Weight / Shell Weight
- **Weight-to-ShuckedWeight Ratio** = Weight / Shucked Weight

Function:

```python
def feature_engineering(df):
    df["Viscera Ratio"] = (df["Viscera Weight"] / df["Weight"])
    df["Shell-to-body Ratio"] = (df["Shell Weight"] / df["Weight"])
    df["Meat Yield"] = df["Shucked Weight"] / (df["Weight"] + df["Shell Weight"])
    df["Length-to-Diameter Ratio"] = (df["Length"] / df["Diameter"])
    df["Weight-to-VisceraWeight Ratio"] = (df["Weight"] / df["Viscera Weight"])
    df["Weight-to-ShellWeight Ratio"] = (df["Weight"] / df["Shell Weight"])
    df["Weight-to-ShuckedWeight Ratio"] = (df["Weight"] / df["Shucked Weight"])
    
    df['Length_Bins'] = pd.qcut(df['Length'], q=4, labels=[1, 2, 3, 4])
    df['BCI'] = np.sqrt(df['Length'] * df['Weight'] * df['Shucked Weight'])
    df['Weight_wo_Viscera'] = (df['Shucked Weight'] - df['Viscera Weight'])
    df['Log Weight'] = np.log(df['Weight'] + 1)

    return df
```

### Encoding the Sex Feature:
To encode the sex feature using the "One-Hot Encoding" method, the following function is used:

```python
def encoder(df):
    enc = OneHotEncoder(sparse_output=False)
    enc_data = enc.fit_transform(df[["Sex"]])
    enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names_out(['Sex']), index=df.index)
    df = pd.concat([df, enc_df], axis=1)
    df.drop("Sex", axis=1, inplace=True)
    
    return df
```

## Models
The following regression models were tested in the project:

1. **Linear Regression**:
   - Polynomial degree: 2
   - Pipeline:
   ```python
   linear = Pipeline([
       ('poly', PolynomialFeatures(degree=2)),
       ('linear', LinearRegression()),
   ])
   ```

2. **Lasso Regression**:
   - Polynomial degree: 3
   - Alpha: 0.0008
   - Pipeline:
   ```python
   lasso = Pipeline([
       ('poly', PolynomialFeatures(degree=3)),
       ('lasso', Lasso(alpha=0.0008))
   ])
   ```

3. **Huber Regressor**:
   - Polynomial degree: 2
   - Epsilon: 1.3
   - Pipeline:
   ```python
   huber = Pipeline([
       ('poly', PolynomialFeatures(degree=2)),
       ('huber', HuberRegressor(epsilon=1.3))
   ])
   ```

4. **Ridge Regression**:
   - Polynomial degree: 2
   - Alpha: 4.2
   - Pipeline:
   ```python
   ridge = Pipeline([
       ('poly', PolynomialFeatures(degree=2)),
       ('ridge', Ridge(alpha=4.2))
   ])
   ```

5. **Elastic Net**:
   - Alpha: 0.0001
   - l1_ratio: 0.90294
   - Pipeline:
   ```python
   elastic = Pipeline([
       ('elastic_net', ElasticNet(alpha=0.0001, l1_ratio=0.90294))
   ])
   ```

6. **Stacking Regressor**:
   - Base models:
     - Ridge, Linear, Lasso, Huber, ElasticNet
   - Final model: Huber Regressor
   - Pipeline:
   ```python
   stacking = StackingRegressor(
       estimators=[
           ("ridge", ridge),
           ('linear', linear),
           ('lasso',lasso),
           ('huber', huber),
           ('elastic', elastic)
       ],
       final_estimator = HuberRegressor(epsilon=1.11, alpha=0.0001),
       cv=5
   )
   ```

