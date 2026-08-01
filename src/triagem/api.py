"""API REST de classificação de laudos médicos."""

from time import perf_counter

from fastapi import FastAPI
from pydantic import BaseModel, Field

from triagem.modelo import carregar_modelo, classificar_laudo

app = FastAPI(title="Triagem de Laudos", version="0.1.0")
modelo = carregar_modelo()


class LaudoEntrada(BaseModel):
    """Laudo médico enviado para classificação."""

    texto: str = Field(min_length=1, description="Texto do laudo médico.")


class ClassificacaoSaida(BaseModel):
    """Resultado da classificação."""

    condicao: str = Field(description="Condição clínica prevista.")
    confianca: float = Field(description="Probabilidade da condição prevista.")
    tempo_ms: float = Field(description="Tempo de inferência em milissegundos.")


@app.get("/saude")
def verificar_saude() -> dict[str, str]:
    """Confirma que o serviço está no ar."""
    return {"status": "ok"}


@app.post("/classificar")
def classificar(entrada: LaudoEntrada) -> ClassificacaoSaida:
    """Classifica a condição clínica descrita no laudo."""
    inicio = perf_counter()
    condicao, confianca = classificar_laudo(modelo, entrada.texto)
    tempo_ms = (perf_counter() - inicio) * 1000
    return ClassificacaoSaida(
        condicao=condicao, confianca=confianca, tempo_ms=tempo_ms
    )
