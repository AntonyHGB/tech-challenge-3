"""DAG de treino do classificador de laudos médicos."""

from airflow.sdk import dag, task

from triagem.modelo import (
    ARQUIVO_TREINO,
    carregar_laudos,
    salvar_modelo,
    treinar_modelo,
)


@dag(schedule=None, catchup=False, max_active_runs=1, tags=["triagem"])
def treino_laudos():
    """Lê o CSV, treina o classificador e salva o modelo."""

    @task
    def carregar_dados() -> int:
        """Lê o CSV de treino e devolve a quantidade de laudos."""
        total = len(carregar_laudos(ARQUIVO_TREINO))
        print(f"Laudos carregados: {total}")
        return total

    @task
    def treinar(total: int) -> dict[str, float]:
        """Treina o classificador e salva os artefatos .joblib e .onnx."""
        print(f"Treinando com {total} laudos")
        metricas = salvar_modelo(treinar_modelo())
        print(f"Acurácia: {metricas['acuracia']:.4f}")
        print(f"F1 macro: {metricas['f1_macro']:.4f}")
        return metricas

    treinar(carregar_dados())


treino_laudos()
