FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# O ONNX Runtime normaliza texto usando a locale en_US.UTF-8, ausente na imagem slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && sed -i '/^# en_US.UTF-8/s/^# //' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

# Instalação editável: o pacote continua em /app/src, junto de dados/ e modelos/.
# Isolada nesta camada, fica em cache enquanto as dependências não mudarem.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY dados ./dados
COPY scripts ./scripts
RUN python scripts/treinar_modelo.py && python scripts/exportar_onnx.py

# A API roda com usuário sem privilégios; o modelo já foi treinado no build.
RUN useradd --create-home triagem && chown -R triagem:triagem /app
USER triagem

EXPOSE 8000

CMD ["uvicorn", "triagem.api:app", "--host", "0.0.0.0", "--port", "8000"]
