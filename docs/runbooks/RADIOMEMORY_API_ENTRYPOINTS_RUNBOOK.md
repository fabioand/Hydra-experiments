# Radiomemory API Entrypoints Runbook

Status: ativo
Ultima validacao: 2026-03-21
Responsavel pela validacao: Codex (via testes HTTP reais)
Fonte de mapeamento inicial: `/Users/fabioandrade/cuter/login.py` e `/Users/fabioandrade/cuter/AIHub.py`

## Objetivo
Este documento padroniza quais entrypoints da API Radiomemory estao disponiveis para uso no projeto, como validar rapidamente e como citar esse padrao em novas implementacoes e testes.

## Como citar este documento
Use a frase abaixo em PRs, issues, tarefas e revisoes:

`Segui o runbook RADIOMEMORY_API_ENTRYPOINTS_RUNBOOK.md (validacao de 2026-03-21) para base URL, autenticacao e smoke tests dos endpoints.`

Frase pronta para pedir o uso no projeto:
`Use o runbook RADIOMEMORY_API_ENTRYPOINTS_RUNBOOK.md e o manifesto docs/runbooks/radiomemory_endpoints_manifest.json como referencia oficial da API Radiomemory nesta entrega.`

## Bases de API identificadas nos scripts
- Producao (TIPO=prod): `https://api.radiomemory.com.br/ia-idoc`
- Dev alternativo (funcionando no teste): `https://iaapi2.radiomemory.com.br/dev`
- Dev legado (DNS nao resolveu no momento da validacao): `https://iaapi.radiomemory.com.br/dev`
- Endpoint isolado citado no codigo: `https://api.radiomemory.com.br/ia-dev/api/v1/panoramics/longaxis`

## Criterio de "endpoint valido"
Para este runbook, endpoint valido significa "rota existe e responde no gateway", mesmo com payload invalido.

Interpretacao usada:
- `200`, `401`, `403`, `422`, `500` = rota existente (valida para descoberta de entrypoint)
- `404` = rota nao encontrada (nao valida)
- erro DNS/conexao = indisponibilidade de host/infra, nao invalida o endpoint por si so

## Autenticacao padrao
### Token endpoint
- Metodo: `POST`
- Rota: `/v1/auth/token`
- Content-Type: `application/x-www-form-urlencoded`
- Corpo esperado (conforme script legado):
  - `grant_type=&username=<user>&password=<pass>&scope=&client_id=&client_secret=`

### Exemplo (sem expor segredo em codigo)
```bash
export RM_BASE="https://api.radiomemory.com.br/ia-idoc"
export RM_USER="test"
export RM_PASS='***'

curl -sS -X POST "$RM_BASE/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -H "accept: application/json" \
  --data "grant_type=&username=$RM_USER&password=$RM_PASS&scope=&client_id=&client_secret="
```

## Matriz de entrypoints (extraidos de AIHub.py e validados)
### Validos (rota existente)
| Endpoint | Status observado (prod) | Observacao de payload |
|---|---:|---|
| `/v1/auth/token` | `200` | autenticacao funcionando |
| `/internal/tomos/Teles` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/SagitalClass` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/Mip8mmBoca` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/Mip8mmMax` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/Mip8mmMan` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/PontosCephMax` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/PontosCephMan` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/PontosCephBoca` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/BigFOVSeg` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/SmallFOVSeg` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/PlanoOclusal` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/LinhaOclusalAxial` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/CycleMax` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/SagitalClassLegacy` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/KeypointsAxial` | `500` | rota existe; requer imagem/entrada valida |
| `/internal/tomos/bbox-autoencoder-maxila` | `422` | confirma contrato com campo `points` |
| `/internal/tomos/bbox-autoencoder-mandibula` | `422` | confirma contrato com campo `points` |
| `/internal/tomos/bbox-autoencoder-boca` | `422` | confirma contrato com campo `points` |
| `/v1/periapicals/longaxis` | `500` | rota existe; requer imagem/entrada valida |
| `https://api.radiomemory.com.br/ia-dev/api/v1/panoramics/longaxis` | `500` | rota externa citada no script |

### Nao validos na data da verificacao (404)
| Endpoint | Status observado (prod) | Acao recomendada |
|---|---:|---|
| `/v1/periapicals/classify` | `404` | tratar como deprecated ou migrado |
| `/v1/periapicals/annomalies_all` | `404` | tratar como deprecated ou migrado |

## Payloads esperados no cliente legado (AIHub.py)
- `AIAPI3`: `{ "base64_image": "<jpeg_base64>" }`
- `GetCefPtsAE`: `{ "points": [...] }`
- `AIAPI`: `{ "file": "<jpeg_base64>" }` (usado em endpoints nao Radiomemory no script)
- `AIAPI2`: `{ "image": "<jpeg_base64>" }` (usado em endpoint nao Radiomemory no script)

## Checklist para novas implementacoes
1. Definir base URL por ambiente (`ia-idoc` prod, `iaapi2/dev` dev).
2. Autenticar em `/v1/auth/token` e guardar `token_type` + `access_token`.
3. Enviar `Authorization: <token_type> <access_token>` em todas as rotas protegidas.
4. Para endpoints de imagem, enviar JPEG em base64 no campo correto (`base64_image`).
5. Implementar timeout e retry com backoff para respostas `5xx`.
6. Logar status HTTP e trecho curto do erro para diagnostico rapido.
7. Cobrir testes de contrato para `422` em rotas `bbox-autoencoder-*` com payload sem `points`.

## Smoke tests minimos recomendados (CI/local)
1. `AuthSmoke`: `/v1/auth/token` retorna `200` e campos `access_token`, `token_type`.
2. `RouteExistenceSmoke`: endpoints criticos retornam diferente de `404`.
3. `ContractSmokeAE`: `bbox-autoencoder-*` sem `points` retorna `422`.
4. `PeriapicalDeprecationGuard`: `/v1/periapicals/classify` e `/v1/periapicals/annomalies_all` continuam `404` (ate confirmacao de nova rota).

## Script de validacao rapida (copiar e executar)
```bash
python3 - <<'PY'
import requests

base = "https://api.radiomemory.com.br/ia-idoc"
user = "${RM_USER}"
password = "${RM_PASS}"

login = requests.post(
    f"{base}/v1/auth/token",
    headers={"Content-type": "application/x-www-form-urlencoded", "accept": "application/json"},
    data=f"grant_type=&username={user}&password={password}&scope=&client_id=&client_secret=",
    timeout=30,
)
print("login", login.status_code)
if login.status_code != 200:
    raise SystemExit(1)

j = login.json()
auth = {
    "Authorization": f"{j.get('token_type', 'Bearer')} {j['access_token']}",
    "Content-type": "application/json",
    "Accept": "application/json",
}

for ep in [
    "/internal/tomos/Teles",
    "/internal/tomos/SagitalClass",
    "/internal/tomos/bbox-autoencoder-maxila",
    "/v1/periapicals/longaxis",
]:
    r = requests.post(f"{base}{ep}", headers=auth, json={"base64_image": ""}, timeout=30)
    if r.status_code == 404:
        r = requests.post(f"{base}{ep}", headers=auth, json={"points": []}, timeout=30)
    print(ep, r.status_code)
PY
```

## Observacoes importantes
- Este runbook documenta descoberta de rotas e disponibilidade de gateway, nao a corretude clinica dos modelos.
- Variacoes de `500` podem ocorrer por payload artificial de smoke test; isso nao invalida a existencia da rota.
- Como os endpoints de documentacao (`/openapi.json`, `/docs`) nao estavam publicos durante a validacao, a referencia oficial aqui e o comportamento observado em runtime.
