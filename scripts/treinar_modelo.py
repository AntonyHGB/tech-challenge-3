"""Treina o modelo de triagem, avalia no conjunto de teste e salva o artefato."""

from triagem.modelo import avaliar_modelo, salvar_modelo, treinar_modelo


def main() -> None:
    """Treina o classificador, imprime as métricas e grava o arquivo .joblib."""
    modelo = treinar_modelo()
    metricas = avaliar_modelo(modelo)
    print(f"Acurácia: {metricas['acuracia']:.4f}")
    print(f"F1 macro: {metricas['f1_macro']:.4f}")
    arquivo = salvar_modelo(modelo)
    print(f"Modelo salvo em {arquivo}")


if __name__ == "__main__":
    main()
