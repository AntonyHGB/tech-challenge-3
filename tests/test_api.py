"""Testes da API de triagem."""

from fastapi.testclient import TestClient

from triagem.api import app

cliente = TestClient(app)


def test_endpoint_de_saude():
    resposta = cliente.get("/saude")
    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_classificar_retorna_urgencia_e_confianca():
    laudo = {"texto": "Radiografia de tórax sem alterações"}
    resposta = cliente.post("/classificar", json=laudo)
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["urgencia"] in {"normal", "atencao", "urgente"}
    assert 0 < corpo["confianca"] <= 1
    assert corpo["tempo_ms"] > 0


def test_texto_vazio_e_rejeitado():
    resposta = cliente.post("/classificar", json={"texto": ""})
    assert resposta.status_code == 422
