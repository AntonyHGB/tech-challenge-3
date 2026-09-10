"""Artefatos isolados: a suíte não depende dos modelos existentes no checkout."""

import pytest

from triagem import modelo as modulo


@pytest.fixture(scope="session", autouse=True)
def modelo(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(modulo, "DIRETORIO_MODELOS", tmp_path_factory.mktemp("modelos"))
        treinado = modulo.treinar_modelo()
        modulo.publicar_modelo(treinado)
        yield treinado
        modulo._abrir_sessao.cache_clear()
