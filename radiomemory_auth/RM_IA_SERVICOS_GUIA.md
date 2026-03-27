# Guia de Uso dos Servicos de IA da Radio Memory (RM)

Ultima validacao deste guia: **2026-03-22**.

Este guia cobre os endpoints ativos encontrados nos repositorios `ai-api`, `ai-api-lambda`, `ia-demo`, `ia-docs`, `ia-lambda` e exemplos do `AIHub.py`.

## 1) Base ativa hoje

Base principal ativa:
- `https://api.radiomemory.com.br/ia-idoc`

Checagens:
- `GET /health` -> `200`
- `GET /v1/docs` -> `200`
- `GET /v1/redoc` -> `200`

Observacao:
- `https://api.radiomemory.com.br/ia/*` e `.../ia-dev/*` retornaram `403` nos testes desta data.

## 2) Autenticacao

Use o script local `login.py` para obter token:

```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/login.py
```

Ele usa por padrao:
- base: `https://api.radiomemory.com.br/ia-idoc`
- endpoint de token: `/v1/auth/token`

Variaveis uteis:
- `RM_BASE_URL`
- `RM_USERNAME`
- `RM_PASSWORD`
- `RM_TIMEOUT`

## 3) Entrypoints ativos - Panoramica e Periapical (`/v1`)

Todos abaixo responderam `200` (com body vazio retornam erro de validacao de `base64_image`, o que confirma endpoint vivo):

Panoramics:
- `POST /v1/panoramics/dentition`
- `POST /v1/panoramics/longaxis`  <- longoeixo panoramica
- `POST /v1/panoramics/metals`
- `POST /v1/panoramics/panorogram` <- panorograma
- `POST /v1/panoramics/teeth_segmentation`
- `POST /v1/panoramics/procedures`
- `POST /v1/panoramics/anatomic_points`
- `POST /v1/panoramics/teeth_anomalies_heatmap`
- `POST /v1/panoramics/resto_radicular`

Periapicals:
- `POST /v1/periapicals/classification`
- `POST /v1/periapicals/longaxis`
- `POST /v1/periapicals/teeth_segmentation`
- `POST /v1/periapicals/anomalies_all`

Observacao:
- `POST /v1/panoramics/describe` e `POST /v1/periapicals/describe` retornaram `500` com payload vazio nesta validacao.

## 4) Entrypoints ativos - Tomografia (`/internal/tomos`)

Todos abaixo responderam `200` com payload vazio (validacao de `base64_image`):

- `POST /internal/tomos/SagitalClass`
- `POST /internal/tomos/SmallFOVSeg`
- `POST /internal/tomos/BigFOVSeg`
- `POST /internal/tomos/PontosCephMax`
- `POST /internal/tomos/PontosCephMan`
- `POST /internal/tomos/PontosCephBoca`
- `POST /internal/tomos/Mip8mmMax`
- `POST /internal/tomos/Mip8mmMan`
- `POST /internal/tomos/Mip8mmBoca`
- `POST /internal/tomos/PlanoOclusal`
- `POST /internal/tomos/LinhaOclusalAxial`
- `POST /internal/tomos/SagitalClassLegacy`
- `POST /internal/tomos/Teles`
- `POST /internal/tomos/CycleMax`
- `POST /internal/tomos/KeypointsAxial`

Legacy (payload `points`):
- `POST /internal/tomos/bbox-autoencoder-maxila`
- `POST /internal/tomos/bbox-autoencoder-mandibula`
- `POST /internal/tomos/bbox-autoencoder-boca`

Observacao:
- `bbox-autoencoder-maxila` retornou `500` no teste rapido com points sinteticos. Validar payload real antes de uso em producao.

## 5) Payload padrao (seguindo o padrao usado em `AIHub.py` + `ai-api-lambda`)

### 5.1 Panoramica / Periapical / Tomos (imagem)

```json
{
  "base64_image": "<imagem_em_base64>",
  "output_width": 0,
  "output_height": 0,
  "threshold": 0.0,
  "resource": "describe",
  "lang": "pt-br",
  "use_cache": false
}
```

### 5.2 Tomos legacy (autoencoder)

```json
{
  "points": [[x1, y1], [x2, y2]]
}
```

## 6) Scripts utilitarios adicionados

Arquivos novos ao lado do `login.py`:
- `rm_ia_client.py`: cliente CLI para chamar endpoints RM com token automatico.
- `rm_ia_smoke_test.py`: smoke test dos endpoints ativos.
- `rm_ia_batch_runner.py`: processamento em lote com limite configuravel.

### 6.1 Exemplos com `rm_ia_client.py`

Longoeixo panoramica:

```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_client.py \
  v1 panoramics/longaxis /caminho/imagem.jpg
```

Panorograma:

```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_client.py \
  v1 panoramics/panorogram /caminho/imagem.jpg
```

Tomografia (SagitalClass):

```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_client.py \
  tomo SagitalClass /caminho/slice.jpg
```

Tomografia (BigFOVSeg):

```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_client.py \
  tomo BigFOVSeg /caminho/slice.jpg
```

Tomografia legacy (points):

```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_client.py \
  tomo-legacy bbox-autoencoder-maxila '[[120.5,80.0],[340.0,260.0]]'
```

### 6.2 Smoke test rapido

```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_smoke_test.py
```

### 6.3 Processamento em lote (numero configuravel de imagens)

Longoeixo panoramica, processando no maximo 30 imagens:

```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_batch_runner.py \
  v1 panoramics/longaxis \
  --input-dir /caminho/imagens \
  --output-dir /caminho/saida_json \
  --limit 30
```

Tomografia (BigFOVSeg), varrendo subpastas e processando todas:

```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_batch_runner.py \
  tomo BigFOVSeg \
  --input-dir /caminho/slices \
  --output-dir /caminho/saida_tomos \
  --recursive \
  --limit 0
```

Parametros principais do batch:
- `--input-dir`: pasta de entrada (opcional, padrao `.`)
- `--output-dir`: pasta de saida (opcional, padrao `./rm_ia_batch_out`)
- `--limit`: numero maximo de imagens (`0` processa todas)
- `--recursive`: inclui subpastas

## 7) Relacao com AIHub.py (~/cuter)

Do `AIHub.py`, os nomes mais usados de tomografia e mapeamento para endpoint:
- `tomos-sagital-classification` -> `/internal/tomos/SagitalClass`
- `big-seg-axial` -> `/internal/tomos/BigFOVSeg`
- `small-seg-axial` -> `/internal/tomos/SmallFOVSeg`
- `mip8mm-max` -> `/internal/tomos/Mip8mmMax`
- `mip8mm-man` -> `/internal/tomos/Mip8mmMan`
- `mip8mm-ambos` -> `/internal/tomos/Mip8mmBoca`
- `pontos-ceph-max` -> `/internal/tomos/PontosCephMax`
- `pontos-ceph-man` -> `/internal/tomos/PontosCephMan`
- `pontos-ceph-boca` -> `/internal/tomos/PontosCephBoca`
- `plano-oclusal` -> `/internal/tomos/PlanoOclusal`
- `pontos-axial` -> `/internal/tomos/LinhaOclusalAxial`
- `cyclemax` -> `/internal/tomos/CycleMax`
- `lcefdata/lrcefdata/llcefdata` -> `/internal/tomos/Teles`

## 8) Troubleshooting rapido

- `403`: ambiente/bucket/rota nao liberada para sua origem/credencial.
- `500`: rota ativa, mas erro interno ou payload incompleto para aquele modelo.
- Erro de validacao de `base64_image`: endpoint vivo; envie imagem real em base64.
- Se a base mudar, exporte `RM_BASE_URL` antes de rodar scripts.
