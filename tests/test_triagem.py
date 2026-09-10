"""Testes do modelo e da API de classificação de laudos."""

import pytest
from fastapi.testclient import TestClient

from triagem import api
from triagem.api import app
from triagem.modelo import (
    avaliar_modelo,
    carregar_sessao_onnx,
    classificar_laudo,
    classificar_laudo_onnx,
    validar_paridade,
)

cliente = TestClient(app)

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
    condicao_onnx = classificar_laudo_onnx(sessao, LAUDO)[0]
    assert condicao_onnx == classificar_laudo(modelo, LAUDO)[0]


def test_endpoint_de_saude():
    resposta = cliente.get("/saude")
    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_endpoint_de_classificacao(modelo):
    resposta = cliente.post("/classificar", json={"texto": LAUDO})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["condicao"] in modelo.classes_
    assert corpo["tempo_ms"] > 0
    assert 0 <= corpo["confianca"] <= 1
    assert corpo["confianca"] == pytest.approx(
        classificar_laudo(modelo, LAUDO)[1], abs=1e-6
    )


def test_texto_vazio_e_rejeitado():
    resposta = cliente.post("/classificar", json={"texto": ""})
    assert resposta.status_code == 422


def test_endpoint_de_metricas():
    cliente.post("/classificar", json={"texto": LAUDO})
    resposta = cliente.get("/metricas")
    assert resposta.status_code == 200
    assert "triagem_requisicoes_total" in resposta.text
    assert "triagem_latencia_segundos" in resposta.text


def test_paridade_no_corpus_completo_e_regressoes_unicode(modelo):
    validar_paridade(modelo, carregar_sessao_onnx())


@pytest.mark.parametrize("texto", [" ", "\t\n", "\u00a0", "x" * 50_001])
def test_texto_em_branco_ou_excessivo_e_rejeitado(texto):
    resposta = cliente.post("/classificar", json={"texto": texto})
    assert resposta.status_code == 422
    assert "input" not in resposta.json()["detail"][0]


@pytest.mark.parametrize("corpo", [{}, {"texto": None}, {"texto": 42}])
def test_entrada_invalida_e_rejeitada(corpo):
    assert cliente.post("/classificar", json=corpo).status_code == 422


def test_unicode_malformado_retorna_422():
    resposta = cliente.post(
        "/classificar",
        content=r'{"texto":"\ud800"}',
        headers={"Content-Type": "application/json"},
    )
    assert resposta.status_code == 422


def test_erro_interno_e_contabilizado(monkeypatch):
    contador = api.REQUISICOES.labels(rota="/classificar", status=500)
    histograma = api.LATENCIA.labels(rota="/classificar")
    antes = contador._value.get()
    soma_antes = histograma._sum.get()

    def falhar(*args):
        raise RuntimeError("Falha de inferência simulada")

    monkeypatch.setattr(api, "classificar_laudo_onnx", falhar)
    with TestClient(app, raise_server_exceptions=False) as cliente_falha:
        resposta = cliente_falha.post("/classificar", json={"texto": LAUDO})
    assert resposta.status_code == 500
    assert contador._value.get() == antes + 1
    assert histograma._sum.get() > soma_antes


def test_rotas_desconhecidas_compartilham_label():
    contador = api.REQUISICOES.labels(rota="nao_encontrada", status=404)
    antes = contador._value.get()
    for indice in range(3):
        assert cliente.get(f"/inexistente-{indice}").status_code == 404
    assert contador._value.get() == antes + 3
    assert "/inexistente-" not in cliente.get("/metricas").text


def test_scrape_nao_e_contabilizado():
    cliente.get("/metricas")
    assert 'rota="/metricas"' not in cliente.get("/metricas").text


def test_prontidao_valida_inferencia():
    with TestClient(app) as cliente_iniciado:
        assert cliente_iniciado.get("/pronto").status_code == 200


def test_prontidao_detecta_falha_do_modelo(monkeypatch):
    def falhar(*args):
        raise RuntimeError("Modelo inválido")

    monkeypatch.setattr(api, "classificar_laudo_onnx", falhar)
    assert cliente.get("/pronto").status_code == 503
    assert cliente.get("/saude").status_code == 200


def test_prontidao_informa_a_versao_testada_durante_promocao(monkeypatch):
    caminho = api.resolver_artefato("modelo.onnx")
    inferencia = api.classificar_laudo_onnx

    def promover_durante_inferencia(sessao, texto):
        monkeypatch.setattr(
            api,
            "resolver_artefato",
            lambda _: caminho.parent.parent / "versao-nova" / "modelo.onnx",
        )
        return inferencia(sessao, texto)

    monkeypatch.setattr(api, "classificar_laudo_onnx", promover_durante_inferencia)
    resposta = cliente.get("/pronto")
    assert resposta.status_code == 200
    assert resposta.json()["versao"] == caminho.parent.name
