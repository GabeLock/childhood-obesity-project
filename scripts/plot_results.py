import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler  # Adicionar esta linha
import matplotlib.pyplot as plt
import os

# Verificar se os arquivos existem
files = ['data/child_ob_gender.csv', 'data/obesity_child_age.csv']
for file in files:
    if not os.path.exists(file):
        raise FileNotFoundError(f'O arquivo {file} não foi encontrado.')

# Carregar os datasets
df_gender = pd.read_csv('data/child_ob_gender.csv')
df_age = pd.read_csv('data/obesity_child_age.csv')

# Verificar se os arquivos não estão vazios
if df_gender.empty:
    raise ValueError('O arquivo data/child_ob_gender.csv está vazio.')
if df_age.empty:
    raise ValueError('O arquivo data/obesity_child_age.csv está vazio.')

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Treinar um modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R^2: {r2}')

# Plotar os resíduos
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_pred - y_test, c='green', marker='s', label='Teste')
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='red')
plt.xlabel('Valores Previstos')
plt.ylabel('Resíduos')
plt.legend(loc='upper left')
plt.title('Gráfico de Dispersão dos Resíduos')
plt.savefig('images/residuals.png')
plt.show()

# Plotar o histograma dos valores previstos
plt.figure(figsize=(10, 6))
plt.hist(y_pred, bins=20, color='blue', edgecolor='black')
plt.xlabel('Valores Previstos')
plt.ylabel('Frequência')
plt.title('Histograma dos Valores Previstos')
plt.savefig('images/histogram.png')
plt.show()

# Plotar gráfico de linha dos percentuais ao longo do tempo
df_age['YEAR_NUM'] = df_age['YEAR_NUM'].apply(pd.to_numeric, errors='coerce')
df_age = df_age.dropna(subset=['YEAR_NUM'])

plt.figure(figsize=(10, 6))
plt.plot(df_age['YEAR_NUM'], df_age['PercentObese'],
         label='Percentual de Obesidade', color='red')
plt.xlabel('Ano')
plt.ylabel('Percentual')
plt.legend()
plt.title('Evolução dos Percentuais de Obesidade ao Longo do Tempo')
plt.savefig('images/obesity_trend.png')
plt.show()
