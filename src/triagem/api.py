"""API REST de triagem de urgência de laudos médicos."""

from time import perf_counter

from fastapi import FastAPI

from triagem.esquemas import ClassificacaoSaida, LaudoEntrada
from triagem.modelo import carregar_modelo, classificar_laudo

app = FastAPI(
    title="Triagem de Laudos",
    description="Classifica a urgência de laudos médicos em normal, atencao ou urgente",
    version="0.1.0",
)

modelo = carregar_modelo()


@app.get("/saude")
def verificar_saude() -> dict[str, str]:
    """Confirma que o serviço está no ar."""
    return {"status": "ok"}


@app.post("/classificar", response_model=ClassificacaoSaida)
def classificar(entrada: LaudoEntrada) -> ClassificacaoSaida:
    """Classifica a urgência do laudo recebido."""
    inicio = perf_counter()
    urgencia, confianca = classificar_laudo(modelo, entrada.texto)
    tempo_ms = (perf_counter() - inicio) * 1000
    return ClassificacaoSaida(urgencia=urgencia, confianca=confianca, tempo_ms=tempo_ms)
