"""Mede a latência baseline da API de triagem rodando em Docker."""

import argparse
import statistics
from time import perf_counter

import httpx

LAUDO = (
    "Acute myocardial infarction with ST segment elevation and cardiogenic shock "
    "requiring immediate coronary reperfusion therapy."
)


def medir_latencia(url: str, repeticoes: int) -> list[float]:
    """Envia requisições sequenciais e devolve os tempos em milissegundos."""
    tempos = []
    with httpx.Client(timeout=30) as cliente:
        cliente.post(url, json={"texto": LAUDO})  # aquecimento
        for _ in range(repeticoes):
            inicio = perf_counter()
            resposta = cliente.post(url, json={"texto": LAUDO})
            resposta.raise_for_status()
            tempos.append((perf_counter() - inicio) * 1000)
    return tempos


def percentil(tempos: list[float], fracao: float) -> float:
    """Retorna o percentil pedido de uma lista de tempos."""
    ordenados = sorted(tempos)
    return ordenados[min(int(fracao * len(ordenados)), len(ordenados) - 1)]


def main() -> None:
    """Executa a medição e imprime o resumo da latência."""
    parser = argparse.ArgumentParser(description="Mede a latência da API de triagem.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/classificar")
    parser.add_argument("--repeticoes", type=int, default=200)
    argumentos = parser.parse_args()

    tempos = medir_latencia(argumentos.url, argumentos.repeticoes)
    print(f"Requisições: {len(tempos)}")
    print(f"Média: {statistics.mean(tempos):.2f} ms")
    print(f"P50: {percentil(tempos, 0.50):.2f} ms")
    print(f"P95: {percentil(tempos, 0.95):.2f} ms")
    print(f"P99: {percentil(tempos, 0.99):.2f} ms")


if __name__ == "__main__":
    main()
