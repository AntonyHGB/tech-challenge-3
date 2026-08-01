"""Modelo baseline de triagem: TF-IDF + Regressão Logística."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_DADOS = RAIZ / "dados" / "laudos.csv"
ARQUIVO_MODELO = RAIZ / "modelos" / "modelo.joblib"


def treinar_modelo(arquivo_dados: Path = ARQUIVO_DADOS) -> Pipeline:
    """Treina o classificador de urgência a partir do CSV de laudos."""
    laudos = pd.read_csv(arquivo_dados)
    modelo = Pipeline(
        [
            ("vetorizador", TfidfVectorizer(ngram_range=(1, 2))),
            ("classificador", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    modelo.fit(laudos["texto"], laudos["urgencia"])
    return modelo


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
