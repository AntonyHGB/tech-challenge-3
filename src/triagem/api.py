"""API REST de classificação de laudos médicos."""

from contextlib import asynccontextmanager
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from triagem.modelo import (
    carregar_sessao_onnx,
    classificar_laudo_onnx,
    resolver_artefato,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Valida o modelo na inicialização, sem efeitos colaterais ao importar."""
    await run_in_threadpool(carregar_sessao_onnx)
    yield


app = FastAPI(title="Triagem de Laudos", version="0.1.0", lifespan=lifespan)

REQUISICOES = Counter(
    "triagem_requisicoes_total",
    "Total de requisições recebidas pela API.",
    ["rota", "status"],
)
LATENCIA = Histogram(
    "triagem_latencia_segundos",
    "Tempo de resposta das requisições em segundos.",
    ["rota"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 5),
)


@app.middleware("http")
async def medir_requisicoes(request: Request, chamar_rota):
    """Registra a contagem e a latência de cada requisição."""
    inicio = perf_counter()
    status = 500
    try:
        resposta = await chamar_rota(request)
        status = resposta.status_code
        return resposta
    finally:
        rota = getattr(request.scope.get("route"), "path", "nao_encontrada")
        if rota != "/metricas":
            REQUISICOES.labels(rota=rota, status=status).inc()
            LATENCIA.labels(rota=rota).observe(perf_counter() - inicio)


@app.exception_handler(RequestValidationError)
async def entrada_invalida(request: Request, erro: RequestValidationError):
    """Não reenvia o texto inválido (inclusive Unicode malformado) na resposta."""
    detalhes = [
        {chave: item[chave] for chave in ("loc", "msg", "type")}
        for item in erro.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detalhes})


class LaudoEntrada(BaseModel):
    """Laudo médico enviado para classificação."""

    texto: str = Field(
        min_length=1, max_length=50_000, description="Texto do laudo médico em inglês."
    )

    @field_validator("texto")
    @classmethod
    def rejeitar_texto_em_branco(cls, texto: str) -> str:
        texto = texto.strip()
        if not texto:
            raise ValueError("O laudo não pode conter somente espaços.")
        return texto


class ClassificacaoSaida(BaseModel):
    """Resultado da classificação."""

    condicao: Literal[
        "neoplasms",
        "digestive system diseases",
        "nervous system diseases",
        "cardiovascular diseases",
        "general pathological conditions",
    ] = Field(description="Condição clínica prevista.")
    confianca: float = Field(
        ge=0, le=1, description="Probabilidade da condição prevista."
    )
    tempo_ms: float = Field(ge=0, description="Tempo de inferência em milissegundos.")


@app.get("/saude")
def verificar_saude() -> dict[str, str]:
    """Confirma que o serviço está no ar."""
    return {"status": "ok"}


@app.get("/pronto")
def verificar_prontidao() -> dict[str, str]:
    """Verifica uma inferência com a versão ativa para o health check."""
    try:
        caminho = resolver_artefato("modelo.onnx")
        classificar_laudo_onnx(carregar_sessao_onnx(caminho), "myocardial infarction")
    except Exception as erro:
        raise HTTPException(status_code=503, detail="Modelo indisponível.") from erro
    return {
        "status": "ok",
        "versao": caminho.parent.name,
    }


@app.get("/metricas")
def expor_metricas() -> Response:
    """Expõe as métricas no formato do Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/classificar")
def classificar(entrada: LaudoEntrada) -> ClassificacaoSaida:
    """Classifica a condição clínica descrita no laudo."""
    inicio = perf_counter()
    condicao, confianca = classificar_laudo_onnx(carregar_sessao_onnx(), entrada.texto)
    tempo_ms = (perf_counter() - inicio) * 1000
    return ClassificacaoSaida(
        condicao=condicao, confianca=confianca, tempo_ms=tempo_ms
    )
