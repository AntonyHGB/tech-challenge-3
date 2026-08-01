"""Modelo de classificação de laudos: TF-IDF + Regressão Logística."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_TREINO = RAIZ / "dados" / "medical_tc_train.csv"
ARQUIVO_TESTE = RAIZ / "dados" / "medical_tc_test.csv"
ARQUIVO_ROTULOS = RAIZ / "dados" / "medical_tc_labels.csv"
ARQUIVO_MODELO = RAIZ / "modelos" / "modelo.joblib"


def carregar_laudos(arquivo: Path) -> pd.DataFrame:
    """Lê um arquivo do corpus e troca o código da condição pelo nome."""
    laudos = pd.read_csv(arquivo)
    rotulos = pd.read_csv(ARQUIVO_ROTULOS)
    nomes = dict(
        zip(rotulos["condition_label"], rotulos["condition_name"], strict=True)
    )
    laudos["condicao"] = laudos["condition_label"].map(nomes)
    return laudos


def treinar_modelo() -> Pipeline:
    """Treina o classificador de condição clínica com os dados de treino."""
    laudos = carregar_laudos(ARQUIVO_TREINO)
    modelo = Pipeline(
        [
            ("vetorizador", TfidfVectorizer(min_df=3, stop_words="english")),
            (
                "classificador",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    modelo.fit(laudos["medical_abstract"], laudos["condicao"])
    return modelo


def avaliar_modelo(modelo: Pipeline) -> dict[str, float]:
    """Calcula acurácia e F1 macro do modelo no conjunto de teste."""
    laudos = carregar_laudos(ARQUIVO_TESTE)
    previsoes = modelo.predict(laudos["medical_abstract"])
    return {
        "acuracia": accuracy_score(laudos["condicao"], previsoes),
        "f1_macro": f1_score(laudos["condicao"], previsoes, average="macro"),
    }


def carregar_modelo() -> Pipeline:
    """Carrega o modelo salvo, treinando e salvando na primeira execução."""
    if not ARQUIVO_MODELO.exists():
        ARQUIVO_MODELO.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(treinar_modelo(), ARQUIVO_MODELO)
    return joblib.load(ARQUIVO_MODELO)


def classificar_laudo(modelo: Pipeline, texto: str) -> tuple[str, float]:
    """Retorna a condição prevista para o laudo e a confiança do modelo."""
    probabilidades = modelo.predict_proba([texto])[0]
    indice = probabilidades.argmax()
    return str(modelo.classes_[indice]), float(probabilidades[indice])
