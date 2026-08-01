FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY dados ./dados
COPY scripts ./scripts

# Instalação editável: o pacote continua em /app/src, junto de dados/ e modelos/.
RUN pip install --no-cache-dir -e . \
    && python scripts/treinar_modelo.py

EXPOSE 8000

CMD ["uvicorn", "triagem.api:app", "--host", "0.0.0.0", "--port", "8000"]
