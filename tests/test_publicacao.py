"""Regressões da publicação, integridade e adoção do modelo pela API."""

import json
from copy import deepcopy

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from triagem import modelo as modulo
from triagem.api import app


def test_primeira_exportacao_publica_uma_unica_versao(modelo, tmp_path, monkeypatch):
    monkeypatch.setattr(modulo, "RAIZ", tmp_path)
    monkeypatch.setattr(modulo, "DIRETORIO_MODELOS", tmp_path / "modelos")
    monkeypatch.setattr(modulo, "treinar_modelo", lambda: modelo)

    caminho = modulo.exportar_onnx()

    assert caminho.exists()
    assert len(list((tmp_path / "modelos").glob("versao-*"))) == 1
    modulo.validar_paridade(modelo, modulo.carregar_sessao_onnx())


def test_publicacao_recarrega_api_sem_reiniciar(modelo, tmp_path, monkeypatch):
    monkeypatch.setattr(modulo, "DIRETORIO_MODELOS", tmp_path)
    modulo.publicar_modelo(modelo)
    anterior = modulo.resolver_artefato("modelo.onnx")
    sessao_anterior = modulo.carregar_sessao_onnx()
    with TestClient(app) as cliente:
        antes = cliente.post("/classificar", json={"texto": "myocardial infarction"})
        candidato = deepcopy(modelo)
        candidato[-1].intercept_[0] += 0.1
        modulo.publicar_modelo(candidato)
        depois = cliente.post("/classificar", json={"texto": "myocardial infarction"})
    assert antes.status_code == depois.status_code == 200
    assert antes.json()["confianca"] != depois.json()["confianca"]
    assert modulo.carregar_sessao_onnx() is not sessao_anterior
    assert anterior.exists()
    assert modulo.resolver_artefato("modelo.onnx").parent == (
        modulo.resolver_artefato("modelo.joblib").parent
    )
    assert modulo.classificar_laudo(
        modulo.carregar_modelo(), "myocardial infarction"
    )[1] == pytest.approx(depois.json()["confianca"], abs=1e-6)


@pytest.mark.parametrize("valor", [0.1, 0.4, float("nan")])
def test_candidato_reprovado_preserva_versao_ativa(
    modelo, tmp_path, monkeypatch, valor
):
    monkeypatch.setattr(modulo, "DIRETORIO_MODELOS", tmp_path)
    manifesto = tmp_path / "atual.json"
    manifesto.write_text('{"versao":"versao-anterior"}')
    monkeypatch.setattr(
        modulo, "avaliar_modelo", lambda _: {"acuracia": 0.6, "f1_macro": valor}
    )
    with pytest.raises(ValueError, match="reprovado"):
        modulo.publicar_modelo(modelo)
    assert json.loads(manifesto.read_text())["versao"] == "versao-anterior"
    assert list(tmp_path.iterdir()) == [manifesto]


@pytest.mark.parametrize("etapa", ["joblib", "manifesto"])
def test_falha_na_escrita_preserva_modelo_ativo(
    modelo, tmp_path, monkeypatch, etapa
):
    monkeypatch.setattr(modulo, "DIRETORIO_MODELOS", tmp_path)
    modulo.publicar_modelo(modelo)
    manifesto = (tmp_path / "atual.json").read_bytes()
    anterior = modulo.resolver_artefato("modelo.joblib")
    conteudo = anterior.read_bytes()

    def falhar(*args, **kwargs):
        raise OSError("Escrita interrompida")

    if etapa == "joblib":
        monkeypatch.setattr(joblib, "dump", falhar)
    else:
        monkeypatch.setattr(modulo.os, "replace", falhar)
    with pytest.raises(OSError, match="interrompida"):
        modulo.publicar_modelo(modelo)
    assert (tmp_path / "atual.json").read_bytes() == manifesto
    assert anterior.read_bytes() == conteudo
    assert len(list(tmp_path.glob("versao-*"))) == 1
    assert modulo.classificar_laudo(modulo.carregar_modelo(), "cancer")[0]


def test_onnx_corrompido_nao_e_publicado(modelo, tmp_path, monkeypatch):
    monkeypatch.setattr(modulo, "DIRETORIO_MODELOS", tmp_path)
    monkeypatch.setattr(modulo, "converter_onnx", lambda _: b"invalido")
    with pytest.raises(Exception, match="INVALID_PROTOBUF"):
        modulo.publicar_modelo(modelo)
    assert not (tmp_path / "atual.json").exists()


def test_paridade_rejeita_modelo_divergente(modelo):
    candidato = deepcopy(modelo)
    candidato[-1].intercept_ += np.arange(5) * 10
    with pytest.raises(ValueError, match="diverge"):
        modulo.validar_paridade(
            candidato, modulo.carregar_sessao_onnx(), ["myocardial infarction"]
        )


def test_modelo_inicial_da_imagem_e_usado_antes_da_primeira_promocao(
    modelo, tmp_path, monkeypatch
):
    base = tmp_path / "imagem"
    monkeypatch.setattr(modulo, "DIRETORIO_MODELOS", base / "modelos")
    modulo.publicar_modelo(modelo)
    esperado = modulo.resolver_artefato("modelo.onnx")
    monkeypatch.setattr(modulo, "RAIZ", base)
    monkeypatch.setattr(modulo, "DIRETORIO_MODELOS", tmp_path / "compartilhado")
    assert modulo.resolver_artefato("modelo.onnx") == esperado
    assert modulo.carregar_sessao_onnx()
