# ML Tech Challenge — Fase 3 (Triagem de Laudos Médicos)

Etapa 1 — Decisão Arquitetural e API Inicial.

---

## 1) Estrutura do projeto

```text
.
├── dados/
│   ├── laudos.csv            ← Conjunto de treino (11.550 laudos rotulados)
│   └── laudos_teste.csv      ← Conjunto de teste (2.888 laudos rotulados)
├── modelos/                  ← Modelo serializado (gerado pelo treino, fora do git)
├── scripts/
│   ├── baixar_dataset.py     ← Baixa o corpus público e gera os CSVs
│   ├── treinar_modelo.py     ← Treina, avalia e salva o .joblib
│   └── medir_latencia.py     ← Mede a latência baseline da API
├── src/
│   └── triagem/
│       ├── api.py            ← API FastAPI com os endpoints /saude e /classificar
│       ├── esquemas.py       ← Contratos de entrada e saída (Pydantic)
│       └── modelo.py         ← Treino, avaliação, persistência e inferência
├── tests/
│   ├── conftest.py           ← Fixture do modelo compartilhada
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

- `dados/laudos.csv` e `dados/laudos_teste.csv`
  Laudos rotulados em três níveis de urgência (`normal`, `atencao`, `urgente`), gerados a partir do corpus público descrito na seção 6.

- `scripts/baixar_dataset.py`
  Baixa o Medical Abstracts TC Corpus e converte os rótulos originais nos três níveis de urgência.

- `src/triagem/modelo.py`
  Pipeline do scikit-learn com TF-IDF (unigramas e bigramas) + Regressão Logística. Expõe `treinar_modelo`, `avaliar_modelo`, `salvar_modelo`, `carregar_modelo` e `classificar_laudo`.

- `src/triagem/esquemas.py`
  Modelos Pydantic `LaudoEntrada` (texto do laudo) e `ClassificacaoSaida` (urgência, confiança e tempo de inferência).

- `src/triagem/api.py`
  Aplicação FastAPI. Carrega o modelo uma única vez na subida do serviço e responde em `GET /saude` e `POST /classificar`.

- `scripts/treinar_modelo.py`
  Treina o classificador, imprime acurácia e F1 macro no conjunto de teste e grava `modelos/modelo.joblib`.

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

### 4.4 (Opcional) Regerar os dados

Os CSVs já estão versionados no repositório. Para baixar o corpus original novamente:
```bash
python scripts/baixar_dataset.py
```

### 4.5 Treinar o modelo
```bash
python scripts/treinar_modelo.py
```
*Saída esperada:*
```text
Acurácia: 0.6139
F1 macro: 0.6104
Modelo salvo em .../modelos/modelo.joblib
```

### 4.6 Subir a API localmente
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
  "texto": "Acute myocardial infarction with ST segment elevation and cardiogenic shock requiring immediate coronary reperfusion therapy."
}
```

**Resposta esperada:**
```json
{
  "urgencia": "urgente",
  "confianca": 0.6694,
  "tempo_ms": 1.82
}
```

> O corpus usado no treino é composto por textos médicos em inglês (seção 6), então o modelo espera laudos nesse idioma.

### 4.7 Subir a API via Docker
```bash
docker build -t triagem-laudos .
docker run --rm -p 8000:8000 triagem-laudos
```
A API fica exposta da mesma forma em `http://127.0.0.1:8000`. O modelo é treinado durante o build, então o container já sobe pronto para servir.

### 4.8 Medir a latência baseline
Com a API rodando (local ou em Docker), em outro terminal:
```bash
python scripts/medir_latencia.py --repeticoes 200
```

### 4.9 Rodar os testes
```bash
pytest
```
*Saída esperada:* 6 passed.

### 4.10 Rodar o linter
```bash
ruff check .
```

---

## 5) Comandos rápidos

| Comando | Descrição |
|---|---|
| `pip install -e ".[dev]"` | Instala as dependências de execução e desenvolvimento |
| `python scripts/baixar_dataset.py` | Baixa o corpus público e regera os CSVs |
| `python scripts/treinar_modelo.py` | Treina o classificador, avalia e salva o artefato |
| `uvicorn triagem.api:app --port 8000` | Sobe a API localmente |
| `docker build -t triagem-laudos .` | Constrói a imagem do serviço de inferência |
| `docker run --rm -p 8000:8000 triagem-laudos` | Sobe a API no container |
| `python scripts/medir_latencia.py` | Mede a latência baseline da API |
| `pytest` | Executa os 6 testes automatizados |
| `ruff check .` | Verifica a conformidade do código |

---

## 6) Dataset e rótulos de urgência

### 6.1 Origem

O projeto usa o **Medical Abstracts TC Corpus**, um dos datasets sugeridos no enunciado, disponível publicamente em [sebischair/Medical-Abstracts-TC-Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus) sob licença **CC BY-SA 3.0**. São 14.438 resumos clínicos em inglês, divididos em 11.550 amostras de treino e 2.888 de teste — acima do mínimo de 2.000 pedido no desafio.

O `scripts/baixar_dataset.py` baixa o corpus original e grava os arquivos já no formato consumido pelo treino (`texto`, `urgencia`).

### 6.2 De condição clínica para nível de urgência

O corpus rotula a **condição médica** descrita no texto, em cinco categorias. Como o objetivo do sistema é triagem por urgência, as categorias foram agrupadas em três níveis segundo a janela de tempo típica para conduta clínica:

| Categoria original | Nível de urgência | Racional |
|---|---|---|
| Cardiovascular diseases | `urgente` | Eventos agudos com janela terapêutica curta (ex.: infarto) |
| Nervous system diseases | `urgente` | Quadros neurológicos agudos onde o atraso agrava sequelas (ex.: AVC) |
| Neoplasms | `atencao` | Investigação prioritária, mas sem emergência imediata |
| Digestive system diseases | `atencao` | Acompanhamento prioritário na maior parte dos casos |
| General pathological conditions | `normal` | Achados inespecíficos, encaminhados para rotina |

> **Transparência:** esse agrupamento é uma aproximação didática feita neste projeto, não um protocolo clínico validado. Ele serve para exercitar o ciclo de vida do modelo (deploy, CI/CD, monitoramento e latência), que é o foco do desafio. Em um cenário real, os rótulos de urgência viriam da classificação de risco feita pela própria instituição.

O agrupamento resultou em classes equilibradas: 3.981 `urgente`, 3.844 `normal` e 3.725 `atencao`.

### 6.3 Desempenho do baseline

Modelo TF-IDF (unigramas e bigramas, 50 mil features) + Regressão Logística com `class_weight="balanced"`, avaliado nas 2.888 amostras de teste:

| Métrica | Resultado |
|---|---|
| Acurácia | 0,6139 |
| F1 macro | 0,6104 |

Com três classes equilibradas, o acaso ficaria em torno de 0,33. O baseline não é o objetivo desta etapa — a modelagem e a otimização são tratadas na Etapa 4 —, mas dá o ponto de partida honesto para comparação.

---

## 7) Decisão arquitetural de deploy em nuvem

### 7.1 Batch ou tempo real?

A triagem existe para reduzir o tempo entre a liberação do laudo e a leitura por um médico. Um laudo com suspeita de infarto classificado como `urgente` só tem valor clínico se a informação chegar em segundos — processar em lote de hora em hora anularia o ganho do sistema.

Por isso a escolha é **inferência em tempo real (síncrona) via API REST**, com o processamento em lote mantido apenas como caminho secundário para reprocessar históricos quando um novo modelo é promovido.

| Critério | Batch | Tempo real (escolhido) |
|---|---|---|
| Latência até o resultado | Minutos a horas | Milissegundos |
| Uso clínico | Relatórios e reprocessamento | Fila de triagem viva |
| Integração com o HIS/RIS | Arquivos agendados | Chamada HTTP no momento da liberação do laudo |
| Custo | Menor por volume | Adequado, o modelo é leve (TF-IDF + Regressão Logística) |

### 7.2 Nuvem e serviços escolhidos

A arquitetura alvo é a **AWS**, com o container publicado em **Amazon ECS com Fargate** atrás de um **Application Load Balancer**:

- **Amazon ECR** — registro da imagem construída pelo pipeline de CI/CD (Etapa 2).
- **Amazon ECS + Fargate** — execução do container sem gerenciar servidores, com escalonamento horizontal por CPU e por número de requisições.
- **Application Load Balancer** — distribuição de carga, terminação TLS e health check apontando para `GET /saude`.
- **Amazon S3** — armazenamento do modelo serializado versionado, consumido na subida do container.
- **Amazon CloudWatch** — logs e alarmes, complementando o Prometheus e o Grafana da Etapa 3.

**Por que Fargate e não Lambda ou EC2:** o Lambda sofreria com cold start no carregamento do modelo e o EC2 exigiria gerenciar as instâncias. O Fargate mantém o container quente, escala por demanda e roda exatamente a mesma imagem Docker validada localmente e no CI — o que preserva a paridade entre os ambientes e sustenta as etapas seguintes (CI/CD, Airflow e observabilidade).

### 7.3 Fluxo da requisição

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

## 8) Latência baseline (Etapa 1)

Medição feita com `scripts/medir_latencia.py` contra o **container Docker**, com 200 requisições sequenciais em um laudo de exemplo de 133 caracteres:

| Métrica | Resultado |
|---|---|
| Média | 3,99 ms |
| P50 | 3,80 ms |
| P95 | 5,72 ms |
| P99 | 7,31 ms |

O tempo medido inclui a ida e volta HTTP. A inferência pura (campo `tempo_ms` da resposta) fica em torno de 1,8 ms, porque o modelo é carregado uma única vez na subida do container e reaproveitado entre as requisições.

Para reproduzir:
```bash
docker build -t triagem-laudos .
docker run --rm -p 8000:8000 triagem-laudos
python scripts/medir_latencia.py --repeticoes 200
```

Esse é o número de referência que será comparado na Etapa 4, depois da otimização do modelo (exportação para ONNX Runtime).

---

## 9) Checklist da Etapa 1

1. [x] **Decisão arquitetural documentada:** análise batch vs. tempo real e escolha da stack de deploy em nuvem (seção 7).
2. [x] **API FastAPI:** recebe o texto do laudo e retorna a classificação de urgência.
3. [x] **Dataset público:** Medical Abstracts TC Corpus com 14.438 amostras, sugerido no enunciado (seção 6).
4. [x] **Modelo de classificação de texto:** TF-IDF + Regressão Logística com scikit-learn.
5. [x] **Container Docker funcional:** imagem que já sobe com o modelo treinado.
6. [x] **Baseline de latência medido:** medição feita no container e registrada (seção 8).
7. [x] **Testes e lint:** 6 testes com pytest e verificação com ruff.

---

## 10) Dependências

**Execução:**
- `fastapi`, `uvicorn[standard]`, `pydantic`
- `scikit-learn`, `pandas`, `joblib`

**Desenvolvimento:**
- `pytest`
- `ruff`
- `httpx`

---

## 11) Próximas etapas

- **Etapa 2:** workflow de CI/CD no GitHub Actions (lint e testes) e DAG do Airflow para o pipeline de treino.
- **Etapa 3:** instrumentação com `prometheus_client` e stack de monitoramento (API + Prometheus + Grafana) via Docker Compose.
- **Etapa 4:** otimização de latência com ONNX Runtime, comparativo com o baseline desta etapa e vídeo STAR.
