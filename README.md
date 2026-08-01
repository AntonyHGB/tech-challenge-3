# ML Tech Challenge — Fase 3 (Triagem de Laudos Médicos)

Etapa 1 — Decisão Arquitetural e API Inicial.

---

## 1) Estrutura do projeto

```text
.
├── dados/
│   └── laudos.csv            ← Amostra rotulada de laudos (texto + urgência)
├── modelos/                  ← Modelo serializado (gerado pelo treino, fora do git)
├── scripts/
│   ├── treinar_modelo.py     ← Treina o classificador e salva o .joblib
│   └── medir_latencia.py     ← Mede a latência baseline da API
├── src/
│   └── triagem/
│       ├── api.py            ← API FastAPI com os endpoints /saude e /classificar
│       ├── esquemas.py       ← Contratos de entrada e saída (Pydantic)
│       └── modelo.py         ← Treino, persistência e inferência do modelo
├── tests/
│   ├── test_api.py           ← Testes dos endpoints
│   └── test_modelo.py        ← Testes do classificador
├── .dockerignore
├── .gitignore
├── Dockerfile                ← Imagem do serviço de inferência
├── pyproject.toml
└── README.md
```

---

## 2) O que cada arquivo principal faz

- `dados/laudos.csv`
  Amostra de 60 laudos rotulados em três níveis de urgência (`normal`, `atencao`, `urgente`), usada para treinar o baseline desta etapa.

- `src/triagem/modelo.py`
  Pipeline do scikit-learn com TF-IDF (unigramas e bigramas) + Regressão Logística. Expõe `treinar_modelo`, `salvar_modelo`, `carregar_modelo` e `classificar_laudo`.

- `src/triagem/esquemas.py`
  Modelos Pydantic `LaudoEntrada` (texto do laudo) e `ClassificacaoSaida` (urgência, confiança e tempo de inferência).

- `src/triagem/api.py`
  Aplicação FastAPI. Carrega o modelo uma única vez na subida do serviço e responde em `GET /saude` e `POST /classificar`.

- `scripts/treinar_modelo.py`
  Treina o classificador e grava `modelos/modelo.joblib`.

- `scripts/medir_latencia.py`
  Dispara requisições sequenciais contra a API e imprime média, P50, P95 e P99 do tempo de resposta.

- `Dockerfile`
  Empacota a API em uma imagem `python:3.12-slim`, treinando o modelo durante o build para que o container já suba pronto para servir.

- `tests/`
  6 testes automatizados cobrindo o classificador e os endpoints da API.

---

## 3) Requisitos

- Python 3.12+
- Git
- Docker (para rodar a API containerizada)

---

## 4) Passo a passo para rodar

### 4.1 Clonar o repositório
```bash
git clone https://github.com/AntonyHGB/tech-challenge-3.git
cd tech-challenge-3
```

### 4.2 Criar o ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.3 Instalar dependências
```bash
pip install -e ".[dev]"
```

### 4.4 Treinar o modelo
```bash
python scripts/treinar_modelo.py
```
*Saída esperada:* `Modelo salvo em .../modelos/modelo.joblib`

### 4.5 Subir a API localmente
```bash
uvicorn triagem.api:app --host 127.0.0.1 --port 8000
```

A API fica disponível em:
- **Documentação interativa:** `http://127.0.0.1:8000/docs`
- **Health check:** `GET http://127.0.0.1:8000/saude`
- **Classificação:** `POST http://127.0.0.1:8000/classificar`

**Exemplo de requisição:**
```json
{
  "texto": "Paciente com dor torácica intensa e sudorese fria com suspeita de infarto"
}
```

**Resposta esperada:**
```json
{
  "urgencia": "urgente",
  "confianca": 0.6349,
  "tempo_ms": 1.42
}
```

### 4.6 Subir a API via Docker
```bash
docker build -t triagem-laudos .
docker run --rm -p 8000:8000 triagem-laudos
```
A API fica exposta da mesma forma em `http://127.0.0.1:8000`.

### 4.7 Medir a latência baseline
Com a API rodando (local ou em Docker), em outro terminal:
```bash
python scripts/medir_latencia.py --repeticoes 200
```

### 4.8 Rodar os testes
```bash
pytest
```
*Saída esperada:* 6 passed.

### 4.9 Rodar o linter
```bash
ruff check .
```

---

## 5) Comandos rápidos

| Comando | Descrição |
|---|---|
| `pip install -e ".[dev]"` | Instala as dependências de execução e desenvolvimento |
| `python scripts/treinar_modelo.py` | Treina o classificador e salva o artefato |
| `uvicorn triagem.api:app --port 8000` | Sobe a API localmente |
| `docker build -t triagem-laudos .` | Constrói a imagem do serviço de inferência |
| `docker run --rm -p 8000:8000 triagem-laudos` | Sobe a API no container |
| `python scripts/medir_latencia.py` | Mede a latência baseline da API |
| `pytest` | Executa os 6 testes automatizados |
| `ruff check .` | Verifica a conformidade do código |

---

## 6) Decisão arquitetural de deploy em nuvem

### 6.1 Batch ou tempo real?

A triagem existe para reduzir o tempo entre a liberação do laudo e a leitura por um médico. Um laudo com suspeita de infarto classificado como `urgente` só tem valor clínico se a informação chegar em segundos — processar em lote de hora em hora anularia o ganho do sistema.

Por isso a escolha é **inferência em tempo real (síncrona) via API REST**, com o processamento em lote mantido apenas como caminho secundário para reprocessar históricos quando um novo modelo é promovido.

| Critério | Batch | Tempo real (escolhido) |
|---|---|---|
| Latência até o resultado | Minutos a horas | Milissegundos |
| Uso clínico | Relatórios e reprocessamento | Fila de triagem viva |
| Integração com o HIS/RIS | Arquivos agendados | Chamada HTTP no momento da liberação do laudo |
| Custo | Menor por volume | Adequado, o modelo é leve (TF-IDF + Regressão Logística) |

### 6.2 Nuvem e serviços escolhidos

A arquitetura alvo é a **AWS**, com o container publicado em **Amazon ECS com Fargate** atrás de um **Application Load Balancer**:

- **Amazon ECR** — registro da imagem construída pelo pipeline de CI/CD (Etapa 2).
- **Amazon ECS + Fargate** — execução do container sem gerenciar servidores, com escalonamento horizontal por CPU e por número de requisições.
- **Application Load Balancer** — distribuição de carga, terminação TLS e health check apontando para `GET /saude`.
- **Amazon S3** — armazenamento do modelo serializado versionado, consumido na subida do container.
- **Amazon CloudWatch** — logs e alarmes, complementando o Prometheus e o Grafana da Etapa 3.

**Por que Fargate e não Lambda ou EC2:** o Lambda sofreria com cold start no carregamento do modelo e o EC2 exigiria gerenciar as instâncias. O Fargate mantém o container quente, escala por demanda e roda exatamente a mesma imagem Docker validada localmente e no CI — o que preserva a paridade entre os ambientes e sustenta as etapas seguintes (CI/CD, Airflow e observabilidade).

### 6.3 Fluxo da requisição

```text
HIS/RIS do hospital
        │  POST /classificar { texto do laudo }
        ▼
Application Load Balancer ──► ECS Fargate (container FastAPI)
                                     │
                                     ├─► modelo TF-IDF + Regressão Logística em memória
                                     └─► resposta { urgencia, confianca, tempo_ms }
```

O modelo é carregado uma única vez na inicialização do container e reaproveitado entre as requisições, evitando o custo de desserialização a cada chamada.

---

## 7) Latência baseline (Etapa 1)

Medição feita com `scripts/medir_latencia.py`, 200 requisições sequenciais em um laudo de exemplo com 76 caracteres:

| Métrica | Resultado |
|---|---|
| Média | 2,48 ms |
| P50 | 2,46 ms |
| P95 | 3,38 ms |
| P99 | 4,15 ms |

O tempo inclui a ida e volta HTTP; a inferência pura (campo `tempo_ms` da resposta) fica abaixo de 2 ms, porque o modelo já está carregado em memória.

Para repetir a medição com a API dentro do container:
```bash
docker build -t triagem-laudos .
docker run --rm -p 8000:8000 triagem-laudos
python scripts/medir_latencia.py --repeticoes 200
```

Esse é o número de referência que será comparado na Etapa 4, depois da otimização do modelo (exportação para ONNX Runtime).

---

## 8) Checklist da Etapa 1

1. [x] **Decisão arquitetural documentada:** análise batch vs. tempo real e escolha da stack de deploy em nuvem (seção 6).
2. [x] **API FastAPI:** recebe o texto do laudo e retorna a classificação de urgência.
3. [x] **Modelo de classificação de texto:** TF-IDF + Regressão Logística com scikit-learn.
4. [x] **Container Docker funcional:** imagem que já sobe com o modelo treinado.
5. [x] **Baseline de latência medido:** script de medição e resultados registrados (seção 7).
6. [x] **Testes e lint:** 6 testes com pytest e verificação com ruff.

---

## 9) Dependências

**Execução:**
- `fastapi`, `uvicorn[standard]`, `pydantic`
- `scikit-learn`, `pandas`, `joblib`

**Desenvolvimento:**
- `pytest`
- `ruff`
- `httpx`

---

## 10) Próximas etapas

- **Etapa 2:** workflow de CI/CD no GitHub Actions (lint e testes) e DAG do Airflow para o pipeline de treino.
- **Etapa 3:** instrumentação com `prometheus_client` e stack de monitoramento (API + Prometheus + Grafana) via Docker Compose.
- **Etapa 4:** otimização de latência com ONNX Runtime, comparativo com o baseline desta etapa e vídeo STAR.
