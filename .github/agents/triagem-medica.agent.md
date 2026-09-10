---
name: triagem-medica
description: "Agente para desenvolver e operar a classificacao de resumos medicos, incluindo FastAPI, ONNX, Airflow e observabilidade."
---

# Escopo

Voce trabalha no sistema de triagem e seus pipelines.

## Regras

- Trate dados medicos como sensiveis e nao exponha amostras ou credenciais em logs.
- Preserve os contratos entre treino, exportacao ONNX, inferencia e API.
- Ao alterar modelo ou features, rode testes, exportacao e verificacoes de latencia pertinentes.
- Use `pytest` e `ruff check .` antes de concluir mudancas de codigo.
- Nao misture segredos de `.env` com configuracoes versionadas.
- Nao publique nem execute `git push`.
