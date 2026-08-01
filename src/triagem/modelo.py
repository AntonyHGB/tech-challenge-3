"""Modelo baseline de triagem: TF-IDF + Regressão Logística."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_DADOS = RAIZ / "dados" / "laudos.csv"
ARQUIVO_TESTE = RAIZ / "dados" / "laudos_teste.csv"
ARQUIVO_MODELO = RAIZ / "modelos" / "modelo.joblib"


def treinar_modelo(arquivo_dados: Path = ARQUIVO_DADOS) -> Pipeline:
    """Treina o classificador de urgência a partir do CSV de laudos."""
    laudos = pd.read_csv(arquivo_dados)
    modelo = Pipeline(
        [
            (
                "vetorizador",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=50_000,
                    stop_words="english",
                ),
            ),
            (
                "classificador",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    modelo.fit(laudos["texto"], laudos["urgencia"])
    return modelo


def avaliar_modelo(
    modelo: Pipeline, arquivo_teste: Path = ARQUIVO_TESTE
) -> dict[str, float]:
    """Calcula acurácia e F1 macro do modelo no conjunto de teste."""
    laudos = pd.read_csv(arquivo_teste)
    previsoes = modelo.predict(laudos["texto"])
    return {
        "acuracia": accuracy_score(laudos["urgencia"], previsoes),
        "f1_macro": f1_score(laudos["urgencia"], previsoes, average="macro"),
    }


def salvar_modelo(modelo: Pipeline, arquivo: Path = ARQUIVO_MODELO) -> Path:
    """Serializa o modelo treinado em disco."""
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, arquivo)
    return arquivo


def carregar_modelo(arquivo: Path = ARQUIVO_MODELO) -> Pipeline:
    """Carrega o modelo salvo, treinando e salvando na primeira execução."""
    if not arquivo.exists():
        salvar_modelo(treinar_modelo(), arquivo)
    return joblib.load(arquivo)


def classificar_laudo(modelo: Pipeline, texto: str) -> tuple[str, float]:
    """Retorna a urgência prevista para o laudo e a confiança do modelo."""
    probabilidades = modelo.predict_proba([texto])[0]
    indice = probabilidades.argmax()
    return str(modelo.classes_[indice]), float(probabilidades[indice])
