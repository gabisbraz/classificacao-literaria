import streamlit as st
import pandas as pd

from app.model.modelo import (
    classificar_livro,
    get_data,
    get_model,
    get_treino,
    get_avaliacao,
)


@st.fragment
def main(aba: str, df: pd.DataFrame, vectorizer, clf):

    # =====================================
    # INTERFACE STREAMLIT
    # =====================================
    st.title("📚 Classificador Literário")

    # ===========================
    # ABA 1: Explorar por Gênero
    # ===========================
    if aba == "🎯 Explorar por Gênero":
        genero_escolhido = st.selectbox(
            "Selecione um gênero:", sorted(df["genero_principal"].unique())
        )
        livros_genero = df[df["genero_principal"] == genero_escolhido]

        st.markdown(f"### Livros do gênero **{genero_escolhido}**")

        for idx, row in livros_genero.iterrows():
            st.markdown(f"**{row['livro']}**")
            st.markdown(f"*{row['descricao'][:300]}...*")
            st.markdown("---")

    # ===================================
    # ABA 2: Prever e recomendar similares
    # ===================================
    elif aba == "🔍 Prever Gênero e Recomendar":
        livro_selecionado = st.selectbox("Selecione um livro:", df["livro"].unique())
        dados_livro = df[df["livro"] == livro_selecionado].iloc[0]

        st.markdown(f"### Livro selecionado: **{dados_livro['livro']}**")
        st.write(dados_livro["descricao"])

        genero_previsto = classificar_livro(
            descricao=dados_livro["descricao"], vectorizer=vectorizer, clf=clf
        )
        st.subheader(f"Gênero predominante previsto pelo modelo: **{genero_previsto}**")
        generos = eval(dados_livro["lista_de_generos"])  # transforma string em lista
        generos_formatados = ", ".join(generos)
        st.write(
            f"Os gêneros referentes ao livro e que estão presentes na base de dados que alimenta o modelo, são: <p style='color: red;'>{generos_formatados}</p>",
            unsafe_allow_html=True,
        )

        similares = df[
            (df["genero_principal"] == genero_previsto)
            & (df["livro"] != livro_selecionado)
        ]
        st.markdown(
            f"<h3 style='color: blue;'>Livros que você também pode gostar:</h3>",
            unsafe_allow_html=True,
        )

        for idx, row in similares.head(5).iterrows():
            st.markdown(f"**{row['livro']}**")
            st.markdown(f"*{row['descricao']}...*")
            st.markdown("---")


page_selected = st.sidebar.radio(
    "Escolha uma opção:",
    ["🎯 Explorar por Gênero", "🔍 Prever Gênero e Recomendar"],
)


@st.cache_data(show_spinner=False)
def get_data_model_app():
    # Etapa 1: carregar e preparar os dados
    df = get_data()

    # Etapa 2: vetorização
    X, y, vectorizer = get_model(df)

    # Etapa 3: treino e teste
    clf, X_train, X_test, y_train, y_test = get_treino(X, y)

    # Etapa 4: avaliação
    get_avaliacao(X_test, y_test, clf)

    return df, vectorizer, clf


df, vectorizer, clf = get_data_model_app()
main(
    aba=page_selected, df=df, vectorizer=vectorizer, clf=clf
)  # Passa o dataframe para a função main
