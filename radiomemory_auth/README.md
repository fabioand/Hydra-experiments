# Radiomemory Auth Tool

Utilitario de autenticacao e acesso rapido aos servicos de IA da RM.

## Arquivos
- `login.py`: autenticacao (token)
- `rm_ia_client.py`: cliente CLI para chamadas de API
- `rm_ia_smoke_test.py`: validacao rapida de disponibilidade de endpoints
- `rm_ia_batch_runner.py`: processamento em lote de imagens para um endpoint
- `RM_IA_SERVICOS_GUIA.md`: guia de uso para desenvolvedores

## Uso rapido

Autenticacao:
```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/login.py
```

Smoke test:
```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_smoke_test.py
```

Exemplo de chamada (longoeixo panoramica):
```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_client.py \
  v1 panoramics/longaxis /caminho/imagem.jpg
```

Exemplo de lote (20 imagens da pasta de entrada):
```bash
python3 /Users/fabioandrade/hydra/radiomemory_auth/rm_ia_batch_runner.py \
  v1 panoramics/longaxis \
  --input-dir /caminho/imagens \
  --output-dir /caminho/saidas \
  --limit 20
```

## Variaveis de ambiente
- `RM_TIPO`: `prod` (padrao) ou `dev`
- `RM_BASE_URL`: sobrescreve base URL principal
- `RM_DEV2_BASE_URL`: sobrescreve base URL do LoginAPIDEV
- `RM_USERNAME`: usuario (padrao: `test`)
- `RM_PASSWORD`: senha
- `RM_TIMEOUT`: timeout em segundos (padrao: `30`)
