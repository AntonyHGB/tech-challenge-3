"""DAG de treino do classificador de laudos médicos."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
from airflow.sdk import dag, task

from triagem.modelo import (
    ARQUIVO_TREINO,
    DIRETORIO_MODELOS,
    carregar_laudos,
    publicar_modelo,
    treinar_modelo,
)


@dag(schedule=None, catchup=False, max_active_runs=1, tags=["triagem"])
def treino_laudos():
    """Carrega os dados, treina o classificador e salva o modelo."""

    @task
    def carregar_dados() -> int:
        """Lê o CSV de treino e devolve a quantidade de laudos."""
        laudos = carregar_laudos(ARQUIVO_TREINO)
        print(f"Laudos carregados: {len(laudos)}")
        return len(laudos)

    @task
    def treinar(total: int) -> str:
        """Treina o classificador e grava o modelo em um arquivo temporário."""
        print(f"Treinando com {total} laudos")
        modelo = treinar_modelo()
        DIRETORIO_MODELOS.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=DIRETORIO_MODELOS, prefix="candidato-", suffix=".joblib", delete=False
        ) as arquivo:
            caminho = Path(arquivo.name)
        try:
            joblib.dump(modelo, caminho)
        except BaseException:
            caminho.unlink(missing_ok=True)
            raise
        return str(caminho)

    @task
    def salvar_modelo(caminho: str) -> dict[str, float]:
        """Avalia o modelo treinado e o promove para o arquivo definitivo."""
        modelo = joblib.load(caminho)
        metricas = publicar_modelo(modelo)
        print(f"Acurácia: {metricas['acuracia']:.4f}")
        print(f"F1 macro: {metricas['f1_macro']:.4f}")
        Path(caminho).unlink(missing_ok=True)
        return metricas

    salvar_modelo(treinar(carregar_dados()))


treino_laudos()
