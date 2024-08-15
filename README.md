# Regression with a Crab Age Dataset

## Maqsad
Ushbu loyiha "Crab Age" (Qisqichbaqa yoshi) datasetidan foydalanib, regressiya modellari yordamida qisqichbaqalar yoshini bashorat qilishga qaratilgan. Loyihada turli regressiya modellari va ularning kombinatsiyalari qo'llanilgan.

## Dataset haqida
Ushbu datasetda 15,000 ta yozuv mavjud va 9 ta ustun bor:

- `Sex`: Qisqichbaqaning jinsi (object)
- `Length`: Qisqichbaqaning uzunligi (mm, float64)
- `Diameter`: Qisqichbaqaning diametri (mm, float64)
- `Height`: Qisqichbaqaning balandligi (mm, float64)
- `Weight`: Qisqichbaqaning og'irligi (g, float64)
- `Shucked Weight`: Yopiq qisqichbaqaning og'irligi (g, float64)
- `Viscera Weight`: Ichak og'irligi (g, float64)
- `Shell Weight`: Qobiq og'irligi (g, float64)
- `Age`: Qisqichbaqaning yoshi (yil, float64)

Datasetning xotira hajmi 1.7 MB.

## Kutubxonalar
Loyihada quyidagi Python kutubxonalari ishlatilgan:

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
Qisqichbaqalarning turli o'lchamlarini hisoblash va yangi xususiyatlarni yaratish uchun quyidagi formula va funksiyalar ishlatilgan:

- **Viscera Ratio** = Viscera Weight / Weight
- **Shell-to-body Ratio** = Shell Weight / Weight
- **Meat Yield** = Shucked Weight / (Weight + Shell Weight)
- **Length-to-Diameter Ratio** = Length / Diameter
- **Weight-to-VisceraWeight Ratio** = Weight / Viscera Weight
- **Weight-to-ShellWeight Ratio** = Weight / Shell Weight
- **Weight-to-ShuckedWeight Ratio** = Weight / Shucked Weight

Funksiya:

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

### Jinsni kodlash:
Jinsni "One-Hot Encoding" usuli bilan kodlash uchun quyidagi funksiya ishlatiladi:

```python
def encoder(df):
    enc = OneHotEncoder(sparse_output=False)
    enc_data = enc.fit_transform(df[["Sex"]])
    enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names_out(['Sex']), index=df.index)
    df = pd.concat([df, enc_df], axis=1)
    df.drop("Sex", axis=1, inplace=True)
    
    return df
```

## Modellar
Loyihada quyidagi regressiya modellari sinovdan o'tkazildi:

1. **Linear Regression**:
   - Polynomial darajasi: 2
   - Pipeline:
   ```python
   linear = Pipeline([
       ('poly', PolynomialFeatures(degree=2)),
       ('linear', LinearRegression()),
   ])
   ```

2. **Lasso Regression**:
   - Polynomial darajasi: 3
   - Alpha: 0.0008
   - Pipeline:
   ```python
   lasso = Pipeline([
       ('poly', PolynomialFeatures(degree=3)),
       ('lasso', Lasso(alpha=0.0008))
   ])
   ```

3. **Huber Regressor**:
   - Polynomial darajasi: 2
   - Epsilon: 1.3
   - Pipeline:
   ```python
   huber = Pipeline([
       ('poly', PolynomialFeatures(degree=2)),
       ('huber', HuberRegressor(epsilon=1.3))
   ])
   ```

4. **Ridge Regression**:
   - Polynomial darajasi: 2
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
   - Birlamchi modellari:
     - Ridge, Linear, Lasso, Huber, ElasticNet
   - Yakuniy model: Huber Regressor
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
