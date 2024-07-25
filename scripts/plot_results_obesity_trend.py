import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregar os datasets
df_gender = pd.read_csv('data/child_ob_gender.csv')
df_age = pd.read_csv('data/obesity_child_age.csv')

# Visualizar as primeiras linhas dos datasets
print(df_gender.head())
print(df_age.head())

# Tratar valores ausentes e inválidos
df_gender['PercentOW'] = df_gender['PercentOW'].apply(lambda x: max(x, 0))
df_gender['PercentOB'] = df_gender['PercentOB'].apply(lambda x: max(x, 0))

# Preencher valores ausentes apenas nas colunas numéricas
numeric_cols = df_gender.select_dtypes(include=[np.number]).columns
df_gender[numeric_cols] = df_gender[numeric_cols].fillna(
    df_gender[numeric_cols].mean())

# Criar a nova feature
df_gender['OW_to_OB_Ratio'] = df_gender['PercentOW'] / df_gender['PercentOB']

# Padronizar os dados
scaler = StandardScaler()
df_gender[['PercentOW', 'PercentOB', 'OW_to_OB_Ratio']] = scaler.fit_transform(
    df_gender[['PercentOW', 'PercentOB', 'OW_to_OB_Ratio']])

# Dividir os dados em treino e teste
X = df_gender[['PercentOW', 'PercentOB', 'OW_to_OB_Ratio']]
y = df_gender['index']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Avaliar o modelo
mse = np.mean((y - y_pred) ** 2)
r2 = model.score(X, y)
print(f'MSE: {mse}')
print(f'R^2: {r2}')

"""# Plotar os resíduos
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y - y_pred, c='green', marker='s', label='Teste')
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='red')
plt.xlabel('Valores Previstos')
plt.ylabel('Resíduos')
plt.legend(loc='upper left')
plt.title('Gráfico de Dispersão dos Resíduos')
plt.savefig('images/residuals.png')
plt.show()"""

# Plotar a tendência de obesidade ao longo do tempo
df_age['Year'] = df_age['Time'].apply(
    lambda x: int(x.strip('[]').split(',')[0]))
plt.figure(figsize=(10, 6))
for age_group in df_age['Age'].unique():
    df_age_group = df_age[df_age['Age'] == age_group]
    plt.plot(df_age_group['Year'],
             df_age_group['PercentObese'], label=f'Idade {age_group}')
plt.xlabel('Ano')
plt.ylabel('Percentual de Obesidade')
plt.legend()
plt.title('Evolução dos Percentuais de Obesidade ao Longo do Tempo')
plt.savefig('images/obesity_trend.png')
plt.show()
