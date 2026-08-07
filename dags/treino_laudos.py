"""DAG de treino do classificador de laudos médicos."""

import joblib
from airflow.sdk import dag, task

from triagem.modelo import (
    ARQUIVO_MODELO,
    ARQUIVO_TREINO,
    avaliar_modelo,
    carregar_laudos,
    treinar_modelo,
)

ARQUIVO_TEMPORARIO = ARQUIVO_MODELO.with_name("modelo_novo.joblib")


@dag(schedule=None, catchup=False, tags=["triagem"])
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
        ARQUIVO_TEMPORARIO.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(treinar_modelo(), ARQUIVO_TEMPORARIO)
        return str(ARQUIVO_TEMPORARIO)

    @task
    def salvar_modelo(caminho: str) -> dict[str, float]:
        """Avalia o modelo treinado e o promove para o arquivo definitivo."""
        modelo = joblib.load(caminho)
        metricas = avaliar_modelo(modelo)
        print(f"Acurácia: {metricas['acuracia']:.4f}")
        print(f"F1 macro: {metricas['f1_macro']:.4f}")
        joblib.dump(modelo, ARQUIVO_MODELO)
        ARQUIVO_TEMPORARIO.unlink()
        return metricas

    salvar_modelo(treinar(carregar_dados()))


treino_laudos()
