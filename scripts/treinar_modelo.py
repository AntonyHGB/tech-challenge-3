"""Treina o modelo, avalia no conjunto de teste e salva o artefato."""

from triagem.modelo import publicar_modelo, resolver_artefato, treinar_modelo


def main() -> None:
    """Treina, valida e publica os artefatos .joblib e .onnx da mesma versão."""
    modelo = treinar_modelo()
    metricas = publicar_modelo(modelo)
    print(f"Acurácia: {metricas['acuracia']:.4f}")
    print(f"F1 macro: {metricas['f1_macro']:.4f}")
    print(f"Modelo salvo em {resolver_artefato('modelo.joblib')}")


if __name__ == "__main__":
    main()
