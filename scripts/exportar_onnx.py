"""Exporta o modelo treinado para o formato ONNX."""

from triagem.modelo import exportar_onnx


def main() -> None:
    """Converte o modelo .joblib em .onnx para inferência mais rápida."""
    print(f"Modelo ONNX salvo em {exportar_onnx()}")


if __name__ == "__main__":
    main()
