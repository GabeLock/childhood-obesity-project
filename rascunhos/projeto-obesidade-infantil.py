# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Função principal para executar o projeto


def main():
    # Passo 1: Carregamento dos Dados
    df_gender = pd.read_csv('C:\Users\Administrator\Documents\Programação\Projetos\childhood-obesity-project\child_ob_gender.csv')
    df_age = pd.read_csv('C:\Users\Administrator\Documents\Programação\Projetos\childhood-obesity-project\obesity_child_age.csv')

    # Passo 2: Visualização e Análise Inicial dos Dados
    print("Primeiras linhas do dataset - Gênero:")
    print(df_gender.head())
    print("\nInformações do dataset - Gênero:")
    print(df_gender.info())
    print("\nEstatísticas descritivas do dataset - Gênero:")
    print(df_gender.describe())

    print("\nPrimeiras linhas do dataset - Idade:")
    print(df_age.head())
    print("\nInformações do dataset - Idade:")
    print(df_age.info())
    print("\nEstatísticas descritivas do dataset - Idade:")
    print(df_age.describe())

    # Passo 3: Tratamento de Valores Ausentes e Outliers
    # Tratar valores negativos e preencher valores ausentes
    df_gender['PercentOW'] = df_gender['PercentOW'].apply(lambda x: max(x, 0))
    df_gender['PercentOB'] = df_gender['PercentOB'].apply(lambda x: max(x, 0))
    df_gender['OW_to_OB_Ratio'] = df_gender['PercentOW'] / \
        df_gender['PercentOB']
    df_gender.fillna(df_gender.mean(), inplace=True)

    df_age['PercentObese'] = df_age['PercentObese'].apply(lambda x: max(x, 0))
    df_age['SE'] = df_age['SE'].apply(lambda x: max(x, 0))
    df_age.fillna(df_age.mean(), inplace=True)

    print("\nVerificação de valores ausentes após o tratamento - Gênero:")
    print(df_gender.isnull().sum())
    print("\nVerificação de valores ausentes após o tratamento - Idade:")
    print(df_age.isnull().sum())

    # Passo 4: Transformação dos Dados
    scaler = StandardScaler()
    df_gender[['PercentOW', 'PercentOB', 'OW_to_OB_Ratio']] = scaler.fit_transform(
        df_gender[['PercentOW', 'PercentOB', 'OW_to_OB_Ratio']])
    df_age[['PercentObese', 'SE']] = scaler.fit_transform(
        df_age[['PercentObese', 'SE']])

    print("\nDados após a padronização - Gênero:")
    print(df_gender.head())
    print("\nDados após a padronização - Idade:")
    print(df_age.head())

    # Passo 5: Modelagem dos Dados
    X_gender = df_gender[['PercentOW', 'OW_to_OB_Ratio']]
    y_gender = df_gender['PercentOB']
    X_age = df_age[['PercentObese', 'SE']]
    y_age = df_age['PercentObese']

    X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(
        X_gender, y_gender, test_size=0.3, random_state=42)
    X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(
        X_age, y_age, test_size=0.3, random_state=42)

    # Treinamento do modelo
    model_gender = LinearRegression()
    model_gender.fit(X_train_gender, y_train_gender)

    model_age = LinearRegression()
    model_age.fit(X_train_age, y_train_age)

    # Avaliação do modelo
    y_train_pred_gender = model_gender.predict(X_train_gender)
    y_test_pred_gender = model_gender.predict(X_test_gender)
    mse_train_gender = mean_squared_error(y_train_gender, y_train_pred_gender)
    r2_train_gender = r2_score(y_train_gender, y_train_pred_gender)
    mse_test_gender = mean_squared_error(y_test_gender, y_test_pred_gender)
    r2_test_gender = r2_score(y_test_gender, y_test_pred_gender)

    y_train_pred_age = model_age.predict(X_train_age)
    y_test_pred_age = model_age.predict(X_test_age)
    mse_train_age = mean_squared_error(y_train_age, y_train_pred_age)
    r2_train_age = r2_score(y_train_age, y_train_pred_age)
    mse_test_age = mean_squared_error(y_test_age, y_test_pred_age)
    r2_test_age = r2_score(y_test_age, y_test_pred_age)

    print(
        f'\nGênero - Treino - MSE: {mse_train_gender}, R²: {r2_train_gender}')
    print(f'Gênero - Teste - MSE: {mse_test_gender}, R²: {r2_test_gender}')

    print(f'\nIdade - Treino - MSE: {mse_train_age}, R²: {r2_train_age}')
    print(f'Idade - Teste - MSE: {mse_test_age}, R²: {r2_test_age}')

    # Passo 6: Visualização dos Resultados
    plt.scatter(y_train_pred_gender, y_train_pred_gender -
                y_train_gender, c='blue', marker='o', label='Treino - Gênero')
    plt.scatter(y_test_pred_gender, y_test_pred_gender -
                y_test_gender, c='green', marker='s', label='Teste - Gênero')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Resíduos')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-2, xmax=2, color='red')
    plt.show()

    plt.scatter(y_train_pred_age, y_train_pred_age - y_train_age,
                c='blue', marker='o', label='Treino - Idade')
    plt.scatter(y_test_pred_age, y_test_pred_age - y_test_age,
                c='green', marker='s', label='Teste - Idade')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Resíduos')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-2, xmax=2, color='red')
    plt.show()

    # Exportar dados transformados para análise futura
    df_gender.to_csv('dados_transformados_gender.csv', index=False)
    df_age.to_csv('dados_transformados_age.csv', index=False)


# Execução do script principal
if __name__ == "__main__":
    main()
