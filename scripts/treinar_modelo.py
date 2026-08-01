"""Treina o modelo de triagem e salva o artefato em modelos/."""

from triagem.modelo import salvar_modelo, treinar_modelo


def main() -> None:
    """Treina o classificador e grava o arquivo .joblib."""
    arquivo = salvar_modelo(treinar_modelo())
    print(f"Modelo salvo em {arquivo}")


if __name__ == "__main__":
    main()
