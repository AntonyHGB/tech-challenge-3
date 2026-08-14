"""Compara a latência de inferência do modelo original com a do ONNX."""

import statistics
from time import perf_counter

from triagem.modelo import (
    ARQUIVO_TESTE,
    carregar_laudos,
    carregar_modelo,
    carregar_sessao_onnx,
    classificar_laudo,
    classificar_laudo_onnx,
)

REPETICOES = 300


def medir(classificar, laudos: list[str]) -> list[float]:
    """Mede o tempo de cada inferência em milissegundos."""
    classificar(laudos[0])  # aquecimento
    tempos = []
    for indice in range(REPETICOES):
        texto = laudos[indice % len(laudos)]
        inicio = perf_counter()
        classificar(texto)
        tempos.append((perf_counter() - inicio) * 1000)
    return tempos


def resumir(nome: str, tempos: list[float]) -> float:
    """Imprime média e P95 dos tempos e devolve a média."""
    media = statistics.mean(tempos)
    p95 = sorted(tempos)[int(0.95 * len(tempos))]
    print(f"{nome:<20} média {media:.3f} ms | P95 {p95:.3f} ms")
    return media


def main() -> None:
    """Roda a comparação e imprime o ganho obtido com o ONNX."""
    laudos = carregar_laudos(ARQUIVO_TESTE)["medical_abstract"].head(100).tolist()
    modelo = carregar_modelo()
    sessao = carregar_sessao_onnx()

    iguais = sum(
        classificar_laudo(modelo, texto)[0] == classificar_laudo_onnx(sessao, texto)[0]
        for texto in laudos
    )
    print(f"Previsões idênticas: {iguais}/{len(laudos)}")

    tempos_original = medir(lambda t: classificar_laudo(modelo, t), laudos)
    tempos_onnx = medir(lambda t: classificar_laudo_onnx(sessao, t), laudos)
    original = resumir("scikit-learn", tempos_original)
    onnx = resumir("ONNX Runtime", tempos_onnx)
    print(f"Ganho: {original / onnx:.2f}x mais rápido")


if __name__ == "__main__":
    main()
