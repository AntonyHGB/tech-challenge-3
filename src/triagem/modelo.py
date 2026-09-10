"""Modelo de classificação de laudos: TF-IDF + Regressão Logística."""

from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_TREINO = RAIZ / "dados" / "medical_tc_train.csv"
ARQUIVO_TESTE = RAIZ / "dados" / "medical_tc_test.csv"
ARQUIVO_ROTULOS = RAIZ / "dados" / "medical_tc_labels.csv"
DIRETORIO_MODELOS = Path(os.environ.get("MODELOS_DIR", RAIZ / "modelos"))
LIMIAR_QUALIDADE = 0.40
TEXTOS_PARIDADE = [
    "naïve T cells",
    "treatment-naïve cancer patients",
    "CÂNCER β cells",
    "İNFARCTION",
    "ΣΟΣ cancer",
]


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
    caminho = resolver_artefato("modelo.joblib")
    if not caminho.exists():
        publicar_modelo(treinar_modelo())
        caminho = resolver_artefato("modelo.joblib")
    return joblib.load(caminho)


def classificar_laudo(modelo: Pipeline, texto: str) -> tuple[str, float]:
    """Retorna a condição prevista para o laudo e a confiança do modelo."""
    probabilidades = modelo.predict_proba([texto])[0]
    indice = probabilidades.argmax()
    return str(modelo.classes_[indice]), float(probabilidades[indice])


def converter_onnx(modelo: Pipeline) -> bytes:
    """Preserva a tokenização Unicode e usa lowercase Python na entrada."""
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType

    modelo = deepcopy(modelo)
    vetorizador = modelo.named_steps["vetorizador"]
    # O lowercase do StringNormalizer difere do Python em casos como İ.
    vetorizador.lowercase = False
    onnx = to_onnx(
        modelo,
        initial_types=[("texto", StringTensorType([None, 1]))],
        options={
            id(vetorizador): {
                "tokenexp": r"[\p{L}\p{N}_]{2,}",
                "locale": "C.UTF-8",
            }
        },
        target_opset=18,
    )
    metadado = onnx.metadata_props.add()
    metadado.key = "triagem.preprocessamento"
    metadado.value = "lower-python"
    return onnx.SerializeToString()


def preparar_entrada(sessao: InferenceSession, textos: list[str]) -> dict:
    """Aplica o pré-processamento declarado pelo artefato, inclusive legados."""
    if sessao.get_modelmeta().custom_metadata_map.get(
        "triagem.preprocessamento"
    ) == "lower-python":
        textos = [texto.lower() for texto in textos]
    return {sessao.get_inputs()[0].name: [[texto] for texto in textos]}


def validar_paridade(
    modelo: Pipeline, sessao: InferenceSession, textos: list[str] | None = None
) -> None:
    """Rejeita divergências de classes ou probabilidades antes da promoção."""
    if textos is None:
        textos = carregar_laudos(ARQUIVO_TESTE)["medical_abstract"].tolist()
        textos += TEXTOS_PARIDADE
    classes, probabilidades = sessao.run(None, preparar_entrada(sessao, textos))
    esperado = modelo.predict_proba(textos)
    obtido = np.array([[p[c] for c in modelo.classes_] for p in probabilidades])
    classes_esperadas = modelo.classes_[esperado.argmax(axis=1)]
    if not np.array_equal(classes, classes_esperadas) or not np.allclose(
        esperado, obtido, atol=1e-6, rtol=1e-5
    ):
        raise ValueError("O ONNX diverge das classes ou probabilidades do modelo.")


def resolver_artefato(nome: str) -> Path:
    """Resolve uma versão imutável; no Docker, usa o modelo inicial da imagem."""
    diretorio = DIRETORIO_MODELOS
    manifesto = diretorio / "atual.json"
    if not manifesto.exists() and diretorio != RAIZ / "modelos":
        diretorio = RAIZ / "modelos"
        manifesto = diretorio / "atual.json"
    if manifesto.exists():
        versao = json.loads(manifesto.read_text())["versao"]
        if not versao.startswith("versao-") or Path(versao).name != versao:
            raise ValueError("Versão inválida no manifesto do modelo.")
        return diretorio / versao / nome
    return diretorio / nome


def publicar_modelo(modelo: Pipeline) -> dict[str, float]:
    """Valida ambos os artefatos e ativa a versão por troca atômica de manifesto."""
    from onnxruntime import InferenceSession

    metricas = avaliar_modelo(modelo)
    if any(
        not np.isfinite(valor) or valor <= LIMIAR_QUALIDADE
        for valor in metricas.values()
    ):
        raise ValueError(f"Modelo reprovado no limiar {LIMIAR_QUALIDADE}: {metricas}")
    onnx = converter_onnx(modelo)
    sessao = InferenceSession(onnx, providers=["CPUExecutionProvider"])
    validar_paridade(modelo, sessao)

    DIRETORIO_MODELOS.mkdir(parents=True, exist_ok=True)
    versao = Path(mkdtemp(prefix="versao-", dir=DIRETORIO_MODELOS))
    temporario = None
    try:
        joblib.dump(modelo, versao / "modelo.joblib")
        (versao / "modelo.onnx").write_bytes(onnx)
        # Airflow cria os arquivos com UID do host; a API só precisa de leitura.
        versao.chmod(0o755)
        for arquivo in versao.iterdir():
            arquivo.chmod(0o644)
        with NamedTemporaryFile(
            mode="w", dir=DIRETORIO_MODELOS, suffix=".json", delete=False
        ) as arquivo:
            temporario = Path(arquivo.name)
            json.dump({"versao": versao.name, "metricas": metricas}, arquivo)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        temporario.chmod(0o644)
        os.replace(temporario, DIRETORIO_MODELOS / "atual.json")
    except BaseException:
        shutil.rmtree(versao)
        raise
    finally:
        if temporario is not None:
            temporario.unlink(missing_ok=True)
    return metricas


def exportar_onnx() -> Path:
    """Publica uma versão validada do modelo atual, incluindo o ONNX."""
    caminho = resolver_artefato("modelo.joblib")
    modelo = joblib.load(caminho) if caminho.exists() else treinar_modelo()
    publicar_modelo(modelo)
    return resolver_artefato("modelo.onnx")


@lru_cache(maxsize=1)
def _abrir_sessao(caminho: str) -> InferenceSession:
    from onnxruntime import InferenceSession

    return InferenceSession(caminho, providers=["CPUExecutionProvider"])


def carregar_sessao_onnx(caminho: Path | None = None) -> InferenceSession:
    """Reutiliza a sessão da versão ativa e recarrega após uma promoção."""
    if caminho is not None:
        return _abrir_sessao(str(caminho))
    caminho = resolver_artefato("modelo.onnx")
    if not caminho.exists():
        exportar_onnx()
        caminho = resolver_artefato("modelo.onnx")
    return _abrir_sessao(str(caminho))


def classificar_laudo_onnx(sessao: InferenceSession, texto: str) -> tuple[str, float]:
    """Retorna a condição prevista pelo modelo ONNX e a confiança."""
    entrada = preparar_entrada(sessao, [texto])
    condicao, probabilidades = sessao.run(None, entrada)
    return str(condicao[0]), float(max(probabilidades[0].values()))
