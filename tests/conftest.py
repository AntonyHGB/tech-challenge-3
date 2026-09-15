"""Fixtures com artefatos temporários, independentes do modelo local."""

import pytest
from fastapi.testclient import TestClient

from triagem import modelo as modulo_modelo


@pytest.fixture(scope="session")
def modelo(tmp_path_factory):
    """Treina os artefatos da suíte sem alterar a pasta modelos/."""
    pasta = tmp_path_factory.mktemp("modelos")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(modulo_modelo, "ARQUIVO_MODELO", pasta / "modelo.joblib")
        patch.setattr(modulo_modelo, "ARQUIVO_ONNX", pasta / "modelo.onnx")
        yield modulo_modelo.carregar_modelo()


@pytest.fixture(scope="session")
def cliente(modelo):
    """Carrega a API depois de preparar os artefatos temporários."""
    from triagem.api import app

    with TestClient(app) as cliente:
        yield cliente
