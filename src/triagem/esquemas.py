"""Esquemas de entrada e saída da API."""

from pydantic import BaseModel, Field


class LaudoEntrada(BaseModel):
    """Laudo médico enviado para triagem."""

    texto: str = Field(min_length=1, description="Texto do laudo médico.")


class ClassificacaoSaida(BaseModel):
    """Resultado da triagem de urgência."""

    urgencia: str = Field(description="Classe prevista: normal, atencao ou urgente.")
    confianca: float = Field(description="Probabilidade da classe prevista.")
    tempo_ms: float = Field(description="Tempo de inferência em milissegundos.")
