# ML Tech Challenge — Fase 3 (Classificação de Laudos Médicos)

Etapas 1 e 2 — API em Docker, CI/CD no GitHub Actions e pipeline de treino no Airflow.

---

## 1) Estrutura do projeto

```text
.
├── .github/workflows/
│   └── ci.yml                ← Pipeline de CI (lint, testes e build)
├── dados/                    ← Corpus público, no formato original
├── dags/
│   └── treino_laudos.py      ← DAG do Airflow com o pipeline de treino
├── modelos/                  ← Modelo serializado (gerado pelo treino, fora do git)
├── scripts/
│   ├── baixar_dataset.py     ← Baixa o corpus
│   ├── treinar_modelo.py     ← Treina, avalia e salva o modelo
│   └── medir_latencia.py     ← Mede a latência da API
├── src/
│   └── triagem/
│       ├── api.py            ← API FastAPI (/saude e /classificar)
│       └── modelo.py         ← Treino, avaliação e inferência
├── tests/
│   └── test_triagem.py       ← 6 testes do modelo e da API
├── docker-compose.airflow.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 2) Dataset

O projeto usa o **Medical Abstracts TC Corpus**, um dos datasets sugeridos no enunciado, publicado em [sebischair/Medical-Abstracts-TC-Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus) sob licença **CC BY-SA 3.0**.

São 14.438 resumos clínicos em inglês — 11.550 de treino e 2.888 de teste — rotulados em cinco condições: *neoplasms*, *digestive system diseases*, *nervous system diseases*, *cardiovascular diseases* e *general pathological conditions*. Os arquivos são usados exatamente como distribuídos pelos autores, sem reagrupar classes nem reamostrar.

O corpus classifica **condição médica**, e não nível de urgência como no exemplo do enunciado — uso confirmado como aceito pela coordenação do curso no Discord da turma.

As classes são desbalanceadas (de 1.195 a 3.844 amostras no treino), então o classificador usa `class_weight="balanced"`, que dá peso maior às classes menos frequentes durante o treino.

---

## 3) Como rodar

### 3.1 Clonar e preparar o ambiente
```bash
git clone https://github.com/AntonyHGB/tech-challenge-3.git
cd tech-challenge-3
python -m venv .venv
```

**Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
**Linux / macOS:** `source .venv/bin/activate`

```bash
pip install -e ".[dev]"
```

### 3.2 Treinar o modelo
```bash
python scripts/treinar_modelo.py
```
*Saída esperada:*
```text
Acurácia: 0.6001
F1 macro: 0.6039
Modelo salvo em .../modelos/modelo.joblib
```

Os CSVs já estão versionados. Para baixar o corpus novamente: `python scripts/baixar_dataset.py`.

### 3.3 Subir a API

**Local:**
```bash
uvicorn triagem.api:app --port 8000
```

**Docker** (o modelo é treinado no build, então o container já sobe pronto):
```bash
docker build -t triagem-laudos .
docker run --rm -p 8000:8000 triagem-laudos
```

A API responde em `http://127.0.0.1:8000`, com documentação interativa em `/docs`.

**Exemplo de requisição** para `POST /classificar`:
```json
{
  "texto": "Acute myocardial infarction with ST segment elevation and cardiogenic shock requiring immediate coronary reperfusion therapy."
}
```

**Resposta:**
```json
{
  "condicao": "cardiovascular diseases",
  "confianca": 0.7066,
  "tempo_ms": 1.38
}
```

> O corpus é composto por textos em inglês, então o modelo espera laudos nesse idioma.

### 3.4 Medir a latência
Com a API rodando, em outro terminal:
```bash
python scripts/medir_latencia.py --repeticoes 200
```

### 3.5 Testes e lint
```bash
pytest
ruff check .
```

---

## 4) CI/CD e pipeline de treino (Etapa 2)

### 4.1 GitHub Actions

O workflow [.github/workflows/ci.yml](.github/workflows/ci.yml) roda a cada push e pull request na `main`, com três automações:

| Job | Comando | Papel |
|---|---|---|
| `lint` | `ruff check .` | Verificação de código |
| `testes` | `pytest` | Os 6 testes do modelo e da API |
| `build` | `docker build` | Garante que a imagem continua construindo |

O `build` só roda se lint e testes passarem. Como os dados estão versionados no repositório, o pipeline executa sem rede externa e sem segredos configurados — a imagem é construída para validação, não publicada.

### 4.2 DAG do Airflow

A DAG [dags/treino_laudos.py](dags/treino_laudos.py) reproduz o ciclo de treino em três tasks, reaproveitando as mesmas funções que a API usa:

```text
carregar_dados  →  treinar  →  salvar_modelo
 (lê o CSV e       (TF-IDF +   (avalia e promove
  conta laudos)     LogReg)     para modelo.joblib)
```

A task `treinar` grava em um arquivo temporário e só a `salvar_modelo` promove o resultado para `modelos/modelo.joblib`, depois de calcular as métricas. Assim uma execução que falhe no meio não corrompe o modelo que a API está servindo.

**Subir o Airflow:**
```bash
docker compose -f docker-compose.airflow.yml up -d
```

A interface fica em `http://localhost:8080`. O usuário é `admin` e a senha é gerada na primeira subida — procure pela linha `Password for user 'admin'` na saída de:
```bash
docker compose -f docker-compose.airflow.yml logs airflow
```

**Executar a DAG pela linha de comando:**
```bash
docker compose -f docker-compose.airflow.yml exec airflow airflow dags test treino_laudos
```

O Airflow roda em um único container (`apache/airflow:3.3.0` em modo standalone, com SQLite), suficiente para esta DAG e bem mais simples de subir que o compose oficial de seis serviços. As pastas `src/`, `dados/` e `modelos/` são montadas como volumes, então a DAG usa o mesmo código da API, sem duplicar a lógica de treino.

> O `apache-airflow` não faz parte das dependências do projeto: ele não roda nativamente no Windows e pesaria o CI sem necessidade. A DAG é executada no container.

---

## 5) Decisão arquitetural de deploy em nuvem

### 5.1 Batch ou tempo real?

A classificação existe para reduzir o tempo entre a liberação do laudo e a leitura por um médico. Um laudo que sinaliza um quadro cardiovascular agudo só tem valor clínico se a informação chegar em segundos — processar em lote de hora em hora anularia o ganho do sistema.

Por isso a escolha é **inferência em tempo real (síncrona) via API REST**, com o processamento em lote mantido apenas como caminho secundário para reprocessar históricos quando um novo modelo é promovido.

| Critério | Batch | Tempo real (escolhido) |
|---|---|---|
| Latência até o resultado | Minutos a horas | Milissegundos |
| Uso clínico | Relatórios e reprocessamento | Fila de triagem viva |
| Integração com o HIS/RIS | Arquivos agendados | Chamada HTTP na liberação do laudo |
| Custo | Menor por volume | Adequado, o modelo é leve |

### 5.2 Nuvem e serviços escolhidos

A arquitetura alvo é a **AWS**, com o container publicado em **Amazon ECS com Fargate** atrás de um **Application Load Balancer**:

- **Amazon ECR** — registro da imagem construída pelo pipeline de CI/CD (Etapa 2).
- **Amazon ECS + Fargate** — execução do container sem gerenciar servidores, com escalonamento horizontal por CPU e por número de requisições.
- **Application Load Balancer** — distribuição de carga, terminação TLS e health check apontando para `GET /saude`.
- **Amazon S3** — armazenamento do modelo serializado versionado, consumido na subida do container.
- **Amazon CloudWatch** — logs e alarmes, complementando o Prometheus e o Grafana da Etapa 3.

**Por que Fargate e não Lambda ou EC2:** o Lambda sofreria com cold start no carregamento do modelo e o EC2 exigiria gerenciar as instâncias. O Fargate mantém o container quente, escala por demanda e roda exatamente a mesma imagem Docker validada localmente e no CI — o que preserva a paridade entre os ambientes e sustenta as etapas seguintes.

### 5.3 Fluxo da requisição

```text
HIS/RIS do hospital
        │  POST /classificar { texto do laudo }
        ▼
Application Load Balancer ──► ECS Fargate (container FastAPI)
                                     │
                                     ├─► modelo TF-IDF + Regressão Logística em memória
                                     └─► resposta { condicao, confianca, tempo_ms }
```

O modelo é carregado uma única vez na inicialização do container e reaproveitado entre as requisições, evitando o custo de desserialização a cada chamada.

---

## 6) Resultados

**Modelo** — TF-IDF + Regressão Logística, avaliado nas 2.888 amostras de teste:

| Métrica | Resultado |
|---|---|
| Acurácia | 0,6001 |
| F1 macro | 0,6039 |

Com cinco classes, o acaso ficaria em torno de 0,20. A modelagem e a otimização são o foco da Etapa 4; aqui o objetivo é ter um baseline funcional servido em produção.

**Latência** — 200 requisições sequenciais contra o container Docker:

| Métrica | Resultado |
|---|---|
| Média | 3,75 ms |
| P50 | 3,44 ms |
| P95 | 5,98 ms |
| P99 | 10,28 ms |

O tempo inclui a ida e volta HTTP; a inferência pura (campo `tempo_ms` da resposta) fica em torno de 1,4 ms. Esse é o número de referência para a comparação da Etapa 4, depois da conversão para ONNX Runtime.

---

## 7) Checklist

**Etapa 1**

1. [x] **Decisão arquitetural documentada:** batch vs. tempo real e stack de deploy em nuvem (seção 5).
2. [x] **API FastAPI:** recebe o texto do laudo e retorna a classificação.
3. [x] **Dataset público:** Medical Abstracts TC Corpus, no formato original (seção 2).
4. [x] **Modelo de classificação de texto:** TF-IDF + Regressão Logística com scikit-learn.
5. [x] **Container Docker funcional:** imagem que já sobe com o modelo treinado.
6. [x] **Baseline de latência medido:** medição feita no container (seção 6).

**Etapa 2**

7. [x] **CI/CD com pelo menos 2 automações:** lint, testes e build da imagem no GitHub Actions.
8. [x] **DAG do Airflow funcional:** carregamento de dados → treino → salvamento do modelo.
9. [x] **Testes e lint:** 6 testes com pytest e verificação com ruff, executados a cada push.

---

## 8) Próximas etapas

- **Etapa 3:** instrumentação com `prometheus_client` e stack de monitoramento via Docker Compose.
- **Etapa 4:** otimização de latência com ONNX Runtime, comparativo com este baseline e vídeo STAR.
