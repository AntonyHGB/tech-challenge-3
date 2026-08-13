"""Testes do modelo e da API de classificação de laudos."""

from fastapi.testclient import TestClient

from triagem.api import app, modelo
from triagem.modelo import avaliar_modelo, classificar_laudo

cliente = TestClient(app)

LAUDO = (
    "Acute myocardial infarction with ST segment elevation and cardiogenic shock "
    "requiring immediate coronary reperfusion therapy."
)


def test_modelo_reconhece_as_cinco_condicoes():
    assert len(modelo.classes_) == 5


def test_classificacao_retorna_condicao_conhecida():
    condicao, confianca = classificar_laudo(modelo, LAUDO)
    assert condicao in modelo.classes_
    assert 0 < confianca <= 1


def test_metricas_ficam_acima_do_acaso():
    metricas = avaliar_modelo(modelo)
    assert metricas["acuracia"] > 0.40
    assert metricas["f1_macro"] > 0.40


def test_endpoint_de_saude():
    resposta = cliente.get("/saude")
    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_endpoint_de_classificacao():
    resposta = cliente.post("/classificar", json={"texto": LAUDO})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["condicao"] in modelo.classes_
    assert corpo["tempo_ms"] > 0


def test_texto_vazio_e_rejeitado():
    resposta = cliente.post("/classificar", json={"texto": ""})
    assert resposta.status_code == 422


def test_endpoint_de_metricas():
    cliente.post("/classificar", json={"texto": LAUDO})
    resposta = cliente.get("/metricas")
    assert resposta.status_code == 200
    assert "triagem_requisicoes_total" in resposta.text
    assert "triagem_latencia_segundos" in resposta.text
