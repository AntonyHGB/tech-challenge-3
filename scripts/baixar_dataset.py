"""Baixa o Medical Abstracts TC Corpus para a pasta dados/."""

import urllib.request
from pathlib import Path

URL_BASE = (
    "https://raw.githubusercontent.com/"
    "sebischair/Medical-Abstracts-TC-Corpus/main"
)
PASTA_DADOS = Path(__file__).resolve().parents[1] / "dados"
ARQUIVOS = ["medical_tc_train.csv", "medical_tc_test.csv", "medical_tc_labels.csv"]


def main() -> None:
    """Baixa os arquivos do corpus, mantendo o formato original."""
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS:
        destino = PASTA_DADOS / nome
        urllib.request.urlretrieve(f"{URL_BASE}/{nome}", destino)
        print(f"Arquivo baixado: {destino}")


if __name__ == "__main__":
    main()
