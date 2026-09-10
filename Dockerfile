FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instalação editável: o pacote continua em /app/src, junto de dados/ e modelos/.
COPY pyproject.toml README.md requirements-modelo.txt ./
COPY src ./src
RUN pip install --no-cache-dir -c requirements-modelo.txt -e .

COPY dados ./dados
COPY scripts ./scripts
RUN python scripts/treinar_modelo.py

# A API roda com usuário sem privilégios; o modelo já foi treinado no build.
RUN useradd --create-home triagem && chown -R triagem:triagem /app
USER triagem

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/saude', timeout=4)"

CMD ["uvicorn", "triagem.api:app", "--host", "0.0.0.0", "--port", "8000"]
