import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
df=sns.load_dataset('diamonds')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
x=df[['carat']]
y=df[['price']]
clarity_ranking=['I1','SI2','SI1','VS2','VS1','VVS2','VVS2','IF']
sns.scatterplot(x='carat',y='price',
                hue='clarity',size='depth',
                palette='ch:r=-.2,d=.3_r',
                hue_order=clarity_ranking,
                sizes=(1,8),linewidth=0,
                data=df)
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2,random_state=0)
RL=LinearRegression()
RL.fit(x_train, y_train)
print(f'intercept: {RL.intercept_}')
print(f'coeficiente {RL.coef_}')
print(f'coeficiente de determinacion R^2', RL.score(x,y))
pred=RL.predict(x_test)
print(f'primeras cinco predicciones: {pred[0:5]}')
y_pred=pred
mse=mean_squared_error(y_true=y_test, y_pred=pred)
rmse=root_mean_squared_error(y_true=y_test, y_pred=pred)
print(f'coeficiente de determinacion R^2', RL.score(x,y))
print(f'El error (MSE) de test es: {mse}')
print(f'El error (RMSE) de test es: {rmse}')
x_range=np.linspace(df['carat'].min(),df['carat'].max(),100)
y_range=np.linspace(df['price'].min(),df['price'].max(),100)
intercept = RL.intercept_
coef = RL.coef_[0]
x_range = np.linspace(df['carat'].min(), df['carat'].max(),100)
y_range = coef * x_range + intercept

plt.plot(x_range, y_range, color='red', label=f'y = {RL.coef_[0][0]:.2f} * x + {RL.intercept_[0]:.2f}')
plt.show()
res = y_test - y_pred
sns.histplot(res,kde=True,color='red', bins=30)
plt.title('distribucion residuos')
plt.xlabel('residuos')
plt.ylabel('distribucion')
plt.axvline(0, color='black',linestyle='--',label='media=0')
plt.legend()
plt.show()
numeric_vars = df.select_dtypes(include=['float64', 'int64'])

# Análisis de distribución
for col in numeric_vars.columns:
    plt.figure(figsize=(12, 6))
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.axvline(df[col].mean(), color='red', linestyle='--', label='Media')
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()
    print(f"Variable: {col}")
    print(f"Media: {df[col].mean():.2f}")
    print(f"Desviación estándar: {df[col].std():.2f}")
    print(f"Mínimo: {df[col].min():.2f}")
    print(f"Máximo: {df[col].max():.2f}")
    print("-" * 30)
