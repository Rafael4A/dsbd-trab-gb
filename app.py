import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Carregar o modelo treinado
modelo = joblib.load('./models/modelo.pkl')

# Lista de gêneros disponíveis
GENRES_LIST = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']

# Função para padronizar os dados de duração
def convert_duration(duration):
    if isinstance(duration, str):
        duration = duration.strip()
        parts = duration.split(' ')
        if len(parts) == 2:
            if parts[1] == 'h':
                hours = int(parts[0])
                return hours * 60
            elif parts[1] == 'm':
                minutes = int(parts[0])
                return minutes
        elif len(parts) == 4:
            hours = int(parts[0])
            minutes = int(parts[2])
            return hours * 60 + minutes
    return np.nan

# Função para exibir dados relevantes do dataset e gerar gráficos
def exibir_dados(df):
    st.write("Amostra de Dados do Dataset:")
    st.write(df.head())
    st.write("Descrição Estatística do Dataset:")
    st.write(df.describe())

    # Aplicar a conversão para a coluna 'Duration'
    df['Duration'] = df['Duration'].apply(convert_duration)
    # Imputar valores NaN na coluna 'Duration'
    df['Duration'] = df['Duration'].fillna(df['Duration'].mean()).astype(int)

    # Gráfico de Distribuição de Avaliações
    st.subheader("Distribuição das Avaliações")
    st.bar_chart(df['Rating'].value_counts())
    
    # Gráfico de Distribuição da Duração
    st.subheader("Distribuição da Duração dos Filmes (em minutos)")
    st.bar_chart(df['Duration'].value_counts().sort_index())

# Função para transformar dados de input
def transforma_dados(input_dados):
    input_transformado = pd.DataFrame([input_dados], columns=['No of Persons Voted', 'Duration', 'Genres', 'Num_Genres'])
    return input_transformado

# Função principal do Streamlit
def main():
    st.title('Previsão do Modelo de ML de Avaliação de Filmes')

    # Carregar o dataset para exibição
    df = pd.read_csv('./data/dataset.csv', index_col=0)

    # Exibir dados relevantes do dataset
    if st.checkbox("Exibir Dados do Dataset"):
        exibir_dados(df)

    # Receber dados do usuário para inferência
    st.subheader("Insira os Dados para Inferência")
    num_voted = st.number_input('Número de Votos', min_value=0)
    duration = st.number_input('Duração (min)', min_value=0)
    selected_genres = st.multiselect('Selecione os Gêneros', GENRES_LIST, default=['Drama', 'Action'])

    # Definir automaticamente o número de gêneros
    num_genres = len(selected_genres)
    
    # Exibir número de gêneros selecionados
    st.text(f'Número de Gêneros: {num_genres}')

    # Transformar os dados do usuário
    genres = ', '.join(selected_genres)
    user_input = [num_voted, duration, genres, num_genres]
    dados_transformados = transforma_dados(user_input)

    if st.button('Realizar Previsão'):
        # Realizar a inferência
        resultado = modelo.predict(dados_transformados)
        
        # Formatação do Resultado
        st.subheader("Resultado da Inferência")
        st.metric(label="Avaliação Prevista", value=f"{resultado[0]:.2f}")
        
        # Adicionar um marcador de sucesso
        st.success("Previsão realizada com sucesso!")

if __name__ == "__main__":
    main()