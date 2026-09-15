# ML Tech Challenge — Fase 3 (Classificação de Laudos Médicos)

Projeto completo — API em Docker, CI/CD no GitHub Actions, pipeline de treino no Airflow, monitoramento com Prometheus e Grafana e inferência otimizada com ONNX Runtime.

---

## 1) Estrutura do projeto

```text
.
├── .github/workflows/
│   └── ci.yml                ← Pipeline de CI (lint, testes e build)
├── dados/                    ← Corpus público, no formato original
├── dags/
│   └── treino_laudos.py      ← DAG do Airflow com o pipeline de treino
├── modelos/                  ← .joblib e .onnx gerados pelo treino (fora do git)
├── monitoramento/
│   ├── prometheus.yml        ← Configuração de scrape do Prometheus
│   └── grafana/              ← Datasource e dashboard provisionados
├── scripts/
│   ├── baixar_dataset.py     ← Baixa o corpus
│   ├── treinar_modelo.py     ← Treina, avalia e salva o modelo
│   ├── exportar_onnx.py      ← Converte o modelo para ONNX
│   ├── comparar_latencia.py  ← Compara a inferência original com a ONNX
│   └── medir_latencia.py     ← Mede a latência da API
├── src/
│   └── triagem/
│       ├── api.py            ← API FastAPI (/saude, /classificar e /metricas)
│       └── modelo.py         ← Treino, avaliação e inferência
├── tests/
│   └── test_triagem.py       ← Testes do modelo, da API e das métricas
├── .env.example              ← Modelo das variáveis de ambiente (copiar para .env)
├── docker-compose.yml        ← Stack de monitoramento (API + Prometheus + Grafana)
├── docker-compose.airflow.yml
├── Dockerfile
├── Dockerfile.airflow
├── requirements-modelo.txt   ← Versões da stack de ML compartilhada
├── pyproject.toml
└── README.md
```

---

## 2) Dataset

O projeto usa o **Medical Abstracts TC Corpus**, um dos datasets sugeridos no enunciado, publicado em [sebischair/Medical-Abstracts-TC-Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus) sob licença **CC BY-SA 3.0**.

São 14.438 resumos clínicos em inglês — 11.550 de treino e 2.888 de teste — rotulados em cinco condições: *neoplasms*, *digestive system diseases*, *nervous system diseases*, *cardiovascular diseases* e *general pathological conditions*. Os arquivos são usados exatamente como distribuídos pelos autores, sem reagrupar classes nem reamostrar.

O corpus classifica **condição médica**, e não nível de urgência como no exemplo do enunciado — uso confirmado como aceito pela coordenação do curso no Discord da turma.

As classes são desbalanceadas (de 1.195 a 3.844 amostras no treino), então o classificador usa `class_weight="balanced"`, que dá peso maior às classes menos frequentes durante o treino.

**Limitação do benchmark:** 1.010 das 2.888 linhas de teste (34,97%) contêm textos
também presentes no treino, com rótulos diferentes. Há textos associados a mais de
uma condição no corpus, enquanto este projeto produz uma única classe. As métricas
abaixo usam o split original e não medem exclusivamente generalização para textos
inéditos; a auditoria dessa sobreposição não demonstrou inflação das métricas.

---

## 3) Como rodar

### Pré-requisitos

| Ferramenta | Windows | Linux |
|---|---|---|
| Python 3.12+ | [python.org](https://www.python.org/downloads/) ou `winget install Python.Python.3.12` | `sudo apt install python3.12 python3.12-venv python3-pip` (Debian/Ubuntu) |
| Git | [git-scm.com](https://git-scm.com/) ou `winget install Git.Git` | `sudo apt install git` |
| Docker | [Docker Desktop](https://www.docker.com/products/docker-desktop/) com backend WSL2 | Docker Engine + plugin Compose |
| curl | já incluído no Windows 10/11 | `sudo apt install curl` |

Confira a instalação (exige Python ≥ 3.12):

```bash
python --version
git --version
docker --version
docker compose version
```

> No Windows, `python` pode se chamar `py` — use `py --version` e troque `python` por `py` nos comandos abaixo.

### 3.0 Caminho rápido (Docker)

O modelo é treinado e exportado durante o build — três comandos e o projeto inteiro sobe:

**Linux:**
```bash
cp .env.example .env
docker compose up -d --build
docker compose -f docker-compose.airflow.yml up -d --build
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
docker compose up -d --build
docker compose -f docker-compose.airflow.yml up -d --build
```

- No Linux/WSL, ajuste `AIRFLOW_UID` no `.env` para o resultado de `id -u`; isso permite que a DAG grave o modelo no bind mount `modelos/`.
- No Windows com Docker Desktop, o valor padrão já funciona — nenhum ajuste necessário.

| O que ver | Onde |
|---|---|
| API e documentação interativa | `http://localhost:8000/docs` |
| Dashboard de monitoramento | `http://localhost:3001` (Grafana) |
| DAG de treino | `http://localhost:8080` (Airflow) |

O restante desta seção cobre a execução local, para desenvolvimento.

### 3.1 Clonar e preparar o ambiente

**Linux:**
```bash
git clone https://github.com/AntonyHGB/tech-challenge-3.git
cd tech-challenge-3
python3 -m venv .venv
source .venv/bin/activate
pip install -c requirements-modelo.txt -e ".[dev]"
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/AntonyHGB/tech-challenge-3.git
cd tech-challenge-3
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -c requirements-modelo.txt -e ".[dev]"
```

> Se o Windows bloquear a ativação do ambiente, libere só para o seu usuário e tente de novo:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> .venv\Scripts\Activate.ps1
> ```
>
> O Airflow não roda nativamente no Windows — use-o sempre via Docker (seção 3.0 ou 4.2).

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
  "tempo_ms": 1.13
}
```

> O corpus é composto por textos em inglês, então o modelo espera laudos nesse idioma.

Texto vazio ou composto apenas por espaços retorna **422**. `GET /saude` verifica se o serviço está no ar.

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
| `testes` | `pytest` | Modelo, API, métricas e equivalência ONNX em 100 textos |
| `build` | `docker build` + smoke test HTTP | Valida construção, saúde, classificação e métricas |

O `build` só roda se lint e testes passarem. Os dados estão versionados, dispensando
seu download; imagens e dependências ainda exigem rede em um ambiente limpo. A
imagem é construída e iniciada para validação, sem publicação. A suíte gera seus
artefatos em diretórios temporários, sem depender de `modelos/` do checkout.

### 4.2 DAG do Airflow

A DAG [dags/treino_laudos.py](dags/treino_laudos.py) reproduz o ciclo de treino em duas tasks, reaproveitando as mesmas funções que a API usa:

```text
carregar_dados  →  treinar
 (lê o CSV e       (TF-IDF + LogReg, avalia
  conta laudos)     e salva .joblib + .onnx)
```

A task `treinar` avalia no conjunto de teste e grava os dois artefatos em
`modelos/`. A API executada localmente usa essa pasta: reinicie o Uvicorn após
o retreino para carregar o novo modelo.

A API no Docker usa o modelo treinado dentro da imagem durante o build; não
compartilha a pasta do Airflow. Para refazer esse treino e recriar a API:

```bash
docker compose build --no-cache api
docker compose up -d api
```

**Subir o Airflow:**
```bash
docker compose -f docker-compose.airflow.yml up -d --build
```

A interface fica em `http://localhost:8080`. O usuário é `admin` e a senha é gerada na primeira subida — procure pela linha `Password for user 'admin'` na saída de:
```bash
docker compose -f docker-compose.airflow.yml logs airflow
```

**Executar a DAG pela linha de comando:**
```bash
docker compose -f docker-compose.airflow.yml exec airflow airflow dags test treino_laudos
```

O Airflow roda em um único container (imagem derivada de `apache/airflow:3.3.0`,
em modo standalone com SQLite). As dependências de ML são instaladas no build pelo
`Dockerfile.airflow`, com as mesmas versões usadas na API em
`requirements-modelo.txt`; não há instalação pip a cada inicialização. As pastas
`src/`, `dados/` e `modelos/` são montadas como volumes.

> O `apache-airflow` não faz parte das dependências do projeto: ele não roda nativamente no Windows e pesaria o CI sem necessidade. A DAG é executada no container.

---

## 5) Monitoramento e observabilidade (Etapa 3)

### 5.1 Métricas da API

A API é instrumentada com `prometheus_client` por um middleware que registra duas métricas para cada requisição (exceto as do próprio `/metricas`):

| Métrica | Tipo | O que mede |
|---|---|---|
| `triagem_requisicoes_total` | Counter | Total de requisições, por rota e status HTTP |
| `triagem_latencia_segundos` | Histogram | Tempo de resposta, por rota |

As métricas ficam expostas em `GET /metricas`, no formato do Prometheus.
O histograma inclui intervalos de 0,5 ms, 1 ms e 2,5 ms para distinguir as
respostas rápidas da API. O P95 do Grafana é uma estimativa por intervalos;
o script de latência calcula os percentis das medições individuais.

### 5.2 Subir a stack

Antes da primeira subida, crie o `.env` com as credenciais do Grafana (o arquivo fica fora do git):

```bash
cp .env.example .env
```

```bash
docker compose up -d --build
```

| Serviço | Endereço | Credenciais |
|---|---|---|
| API | `http://localhost:8000` | — |
| Prometheus | `http://localhost:9090` | — |
| Grafana | `http://localhost:3001` | usuário `admin`, senha do `.env` |

Sem o `.env`, o compose se recusa a subir — não existe senha padrão embutida no repositório.

> O Grafana usa a porta 3001 no host para não conflitar com outros serviços comuns na 3000. O Prometheus raspa a API a cada 5 segundos pelo endereço interno `api:8000`.

### 5.3 Dashboard

O Grafana já sobe com o datasource e o dashboard **Triagem de Laudos** provisionados — nenhuma configuração manual é necessária. São três painéis:

1. **Total de requisições** — `sum(triagem_requisicoes_total)`
2. **Latência (P95 e média)** — quantil sobre os buckets do histograma
3. **Taxa de erro (%)** — proporção de respostas 4xx/5xx nos últimos 5 minutos,
   mostrando zero quando não há erros ou tráfego

O JSON do dashboard está versionado em [monitoramento/grafana/dashboards/triagem.json](monitoramento/grafana/dashboards/triagem.json).

Para alimentar os gráficos, gere tráfego com:
```bash
python scripts/medir_latencia.py --repeticoes 200
```

---

## 6) Otimização de latência (Etapa 4)

### 6.1 Técnica aplicada

O pipeline treinado (TF-IDF + Regressão Logística) é exportado para **ONNX** com o
`skl2onnx` e servido pelo **ONNX Runtime**: a vetorização e a classificação passam
a rodar em um grafo compilado.

```bash
python scripts/exportar_onnx.py
```

O treino já salva os dois artefatos; `exportar_onnx.py` regenera o `.onnx` a
partir do `.joblib`. No Docker, o modelo é gerado durante o build e carregado
uma única vez na subida da API.

### 6.2 Comparação

```bash
python scripts/comparar_latencia.py
```

Sem HTTP, com 300 chamadas:

| Modelo | Média | P95 |
|---|---|---|
| scikit-learn (`.joblib`) | 0,521 ms | 0,599 ms |
| ONNX Runtime (`.onnx`) | 0,098 ms | 0,136 ms |

**Ganho de ~5,3x** na média. Os valores variam com a carga da máquina, não com o modelo.

O arquivo também encolheu: **1,04 MB → 791 KB**.

### 6.3 As previsões continuam as mesmas

Antes de medir, o comparador verifica se as duas versões preveem a mesma classe
nos mesmos 100 laudos:

```text
Previsões idênticas: 100/100
```

Há ainda um teste automatizado (`test_onnx_preve_o_mesmo_que_o_modelo_original`)
que trava essa garantia no CI.

---

## 7) Decisão arquitetural

### 7.1 Batch ou tempo real?

A escolha é **inferência em tempo real (síncrona) via API REST**: cada texto
recebe uma classificação imediatamente, sem esperar um lote agendado. Batch
seria adequado para reprocessar históricos. Este protótipo classifica condições
médicas em textos em inglês; não determina urgência nem foi validado para uso clínico.

| Critério | Batch | Tempo real (escolhido) |
|---|---|---|
| Latência até o resultado | Minutos a horas | Milissegundos |
| Uso clínico | Relatórios e reprocessamento | Fila de triagem viva |
| Integração com o HIS/RIS | Arquivos agendados | Chamada HTTP na liberação do laudo |
| Custo | Menor por volume | Adequado, o modelo é leve |

### 7.2 Deploy em nuvem (referência teórica)

O deploy executado neste projeto é local, via Docker. Como referência teórica,
a mesma imagem subiria em um serviço de containers gerenciado (ex.: AWS ECS
com Fargate), atrás de um balanceador com health check em `GET /saude`, com o
modelo versionado em object storage e logs centralizados. O fluxo é
`POST /classificar` → container FastAPI → resposta
`{ condicao, confianca, tempo_ms }`, reaproveitando a sessão ONNX entre
requisições.

---

## 8) Resultados

**Modelo** — TF-IDF + Regressão Logística, avaliado nas 2.888 amostras de teste:

| Métrica | Resultado |
|---|---|
| Acurácia | 0,6001 |
| F1 macro | 0,6039 |

Com cinco classes, o acaso ficaria em torno de 0,20. A conversão para ONNX preserva essas métricas — é uma otimização de execução, não de modelagem.

**Latência ponta a ponta** — 200 requisições sequenciais contra a API:

| Métrica | Resultado |
|---|---|
| Média | 1,26 ms |
| P50 | 1,17 ms |
| P95 | 1,91 ms |
| P99 | 2,56 ms |

O tempo inclui a ida e volta HTTP, que domina o total — por isso o ganho de ~5,3x da inferência (seção 6) aparece diluído aqui.
