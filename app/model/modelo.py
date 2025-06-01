import pandas as pd
import nltk
import string
import re
import ast
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Downloads necessários do NLTK
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Inicializações
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = re.sub(r"\d+", "", str(text))
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)


def to_list(gen_str):
    try:
        return ast.literal_eval(gen_str)
    except:
        return []


def get_data():
    df = pd.read_excel("app/data/BASE_MODELO.xlsx")
    df = df.loc[df["lista_de_generos"] != "[]"]
    df = df.dropna(subset=["descricao"])
    df["genero_principal"] = df["lista_de_generos"].apply(to_list)
    df = df.explode("genero_principal")
    df["descricao_processada"] = df["descricao"].apply(preprocess_text)

    top_generos = df["genero_principal"].value_counts().head(10).index
    df = df[df["genero_principal"].isin(top_generos)]
    df = df.drop_duplicates("livro")

    return df


def get_model(df):
    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 5), min_df=3, max_df=0.9
    )
    X = vectorizer.fit_transform(df["descricao_processada"])
    y = df["genero_principal"]
    return X, y, vectorizer


def get_treino(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test


def get_avaliacao(X_test, y_test, clf):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Acurácia do modelo: {acc:.4f}")
    print("\n📊 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))


def classificar_livro(descricao, vectorizer, clf):
    desc_proc = preprocess_text(descricao)
    desc_vect = vectorizer.transform([desc_proc])
    pred = clf.predict(desc_vect)[0]
    return pred
