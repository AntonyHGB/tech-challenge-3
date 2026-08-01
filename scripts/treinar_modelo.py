"""Treina o modelo, avalia no conjunto de teste e salva o artefato."""

import joblib

from triagem.modelo import ARQUIVO_MODELO, avaliar_modelo, treinar_modelo


def main() -> None:
    """Treina o classificador, imprime as métricas e grava o arquivo .joblib."""
    modelo = treinar_modelo()
    metricas = avaliar_modelo(modelo)
    print(f"Acurácia: {metricas['acuracia']:.4f}")
    print(f"F1 macro: {metricas['f1_macro']:.4f}")
    ARQUIVO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"Modelo salvo em {ARQUIVO_MODELO}")


if __name__ == "__main__":
    main()
