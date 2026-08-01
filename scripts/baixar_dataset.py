"""Baixa o Medical Abstracts TC Corpus e gera os CSVs de treino e teste."""

from pathlib import Path

import pandas as pd

URL_BASE = (
    "https://raw.githubusercontent.com/"
    "sebischair/Medical-Abstracts-TC-Corpus/main"
)
PASTA_DADOS = Path(__file__).resolve().parents[1] / "dados"

# As 5 condições do corpus agrupadas nos 3 níveis de urgência da triagem.
URGENCIA_POR_CONDICAO = {
    1: "atencao",  # neoplasms
    2: "atencao",  # digestive system diseases
    3: "urgente",  # nervous system diseases
    4: "urgente",  # cardiovascular diseases
    5: "normal",  # general pathological conditions
}


def preparar_arquivo(nome_origem: str, nome_destino: str) -> Path:
    """Baixa um arquivo do corpus e salva com as colunas texto e urgencia."""
    dados = pd.read_csv(f"{URL_BASE}/{nome_origem}")
    dados["texto"] = dados["medical_abstract"].str.strip()
    dados["urgencia"] = dados["condition_label"].map(URGENCIA_POR_CONDICAO)
    destino = PASTA_DADOS / nome_destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    dados[["texto", "urgencia"]].to_csv(destino, index=False)
    return destino


def main() -> None:
    """Gera os dois arquivos usados pelo treino e pela avaliação."""
    arquivos = [
        ("medical_tc_train.csv", "laudos.csv"),
        ("medical_tc_test.csv", "laudos_teste.csv"),
    ]
    for origem, destino in arquivos:
        arquivo = preparar_arquivo(origem, destino)
        total = len(pd.read_csv(arquivo))
        print(f"Arquivo gerado: {arquivo} ({total} amostras)")


if __name__ == "__main__":
    main()
