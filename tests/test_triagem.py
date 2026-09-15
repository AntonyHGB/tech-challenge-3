"""Testes do modelo e da API de classificação de laudos."""

import pytest
from prometheus_client import REGISTRY

from triagem.modelo import (
    ARQUIVO_TESTE,
    avaliar_modelo,
    carregar_laudos,
    carregar_sessao_onnx,
    classificar_laudo,
    classificar_laudo_onnx,
)

LAUDO = (
    "Acute myocardial infarction with ST segment elevation and cardiogenic shock "
    "requiring immediate coronary reperfusion therapy."
)


def test_modelo_reconhece_as_cinco_condicoes(modelo):
    assert len(modelo.classes_) == 5


def test_classificacao_retorna_condicao_conhecida(modelo):
    condicao, confianca = classificar_laudo(modelo, LAUDO)
    assert condicao in modelo.classes_
    assert 0 < confianca <= 1


def test_metricas_ficam_acima_do_acaso(modelo):
    metricas = avaliar_modelo(modelo)
    assert metricas["acuracia"] > 0.40
    assert metricas["f1_macro"] > 0.40


def test_onnx_preve_o_mesmo_que_o_modelo_original(modelo):
    sessao = carregar_sessao_onnx()
    laudos = carregar_laudos(ARQUIVO_TESTE)["medical_abstract"].head(100)
    for texto in laudos:
        condicao, confianca = classificar_laudo(modelo, texto)
        condicao_onnx, confianca_onnx = classificar_laudo_onnx(sessao, texto)
        assert condicao_onnx == condicao
        assert confianca_onnx == pytest.approx(confianca, abs=1e-5)


def test_endpoint_de_saude(cliente):
    resposta = cliente.get("/saude")
    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_endpoint_de_classificacao(cliente, modelo):
    resposta = cliente.post("/classificar", json={"texto": LAUDO})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["condicao"] in modelo.classes_
    assert 0 < corpo["confianca"] <= 1
    assert corpo["tempo_ms"] > 0


@pytest.mark.parametrize("texto", ["", "   ", "\n\t"])
def test_texto_vazio_e_rejeitado(cliente, texto):
    resposta = cliente.post("/classificar", json={"texto": texto})
    assert resposta.status_code == 422


def test_endpoint_de_metricas(cliente):
    rotulos = {"rota": "/classificar", "status": "200"}
    total = REGISTRY.get_sample_value("triagem_requisicoes_total", rotulos) or 0
    cliente.post("/classificar", json={"texto": LAUDO})
    resposta = cliente.get("/metricas")
    assert resposta.status_code == 200
    assert REGISTRY.get_sample_value("triagem_requisicoes_total", rotulos) == total + 1
    assert "triagem_requisicoes_total" in resposta.text
    assert "triagem_latencia_segundos" in resposta.text
    assert REGISTRY.get_sample_value(
        "triagem_latencia_segundos_bucket", {"rota": "/classificar", "le": "0.001"}
    ) is not None
    assert REGISTRY.get_sample_value(
        "triagem_requisicoes_total", {"rota": "/metricas", "status": "200"}
    ) is None
