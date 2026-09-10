"""Treina o modelo, avalia no conjunto de teste e salva os artefatos."""

from triagem.modelo import ARQUIVO_MODELO, salvar_modelo, treinar_modelo


def main() -> None:
    """Treina o classificador, imprime as métricas e salva .joblib e .onnx."""
    metricas = salvar_modelo(treinar_modelo())
    print(f"Acurácia: {metricas['acuracia']:.4f}")
    print(f"F1 macro: {metricas['f1_macro']:.4f}")
    print(f"Modelo salvo em {ARQUIVO_MODELO}")


if __name__ == "__main__":
    main()
