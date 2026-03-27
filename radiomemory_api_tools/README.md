# Radiomemory API Tools

Script para testar a rede de long axis panorâmico da API RM usando imagem real do dataset `longoeixo`.

## Script
- `probe_panoramic_longaxis.py`

## Como usar
```bash
python3 /Users/fabioandrade/hydra/radiomemory_api_tools/probe_panoramic_longaxis.py
```

Esse comando:
1. escolhe automaticamente a primeira imagem em `longoeixo/imgs`
2. autentica usando `radiomemory_auth/login.py`
3. reconverte a imagem para JPEG em memória e envia `base64_image` (compatível com `AIHub.py`)
4. chama `https://api.radiomemory.com.br/ia-dev/api/v1/panoramics/longaxis` com `Accept: text/plain`
5. imprime um resumo com status, tipo de resposta e preview do body

## Opções úteis
```bash
python3 /Users/fabioandrade/hydra/radiomemory_api_tools/probe_panoramic_longaxis.py \
  --image /Users/fabioandrade/hydra/longoeixo/imgs/arquivo.jpg \
  --save-json /Users/fabioandrade/hydra/analysis_inputs/rm_longaxis_probe.json
```

## Variáveis de ambiente (autenticação)
- `RM_TIPO`
- `RM_BASE_URL`
- `RM_USERNAME`
- `RM_PASSWORD`
- `RM_TIMEOUT`

As variáveis são lidas por `radiomemory_auth/login.py`.
