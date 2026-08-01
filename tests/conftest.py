"""Fixtures compartilhadas pelos testes."""

import pytest

from triagem.modelo import carregar_modelo


@pytest.fixture(scope="session")
def modelo():
    """Modelo treinado, carregado uma única vez para toda a sessão de testes."""
    return carregar_modelo()
