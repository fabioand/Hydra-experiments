# Hydra Experiments

Monorepo de pesquisa e engenharia para modelos de visão odontológica com foco em:

- localização de long-eixo dentário (Hydra multitarefa)
- inferência MultiROI (center + lateral)
- autoencoders auxiliares para reconstrução/filtro
- utilitários de integração e auditoria com APIs Radiomemory

## Estrutura do repositório

- `train.py`, `eval.py`, `hydra_data.py`, `hydra_multitask_model.py`
  - pipeline principal Hydra U-Net multitarefa
- `longoeixo/`
  - scripts de treino/avaliação, presets e utilitários de dataset
- `dae_longoeixo/`
  - pipeline DAE para imputação/ajuste geométrico
- `panorama_foundation/`
  - experimentos de autoencoder para fundação visual
- `ae_recon_webapp/`
  - webapp de reconstrução (backend FastAPI + frontend)
- `ae_radiograph_filter/`
  - scripts de filtro/reconstrução local
- `radiomemory_api_tools/`, `radiomemory_auth/`, `radiomemory_api_discovery/`
  - ferramentas de API, autenticação e exploração de endpoints
- `docs/`
  - arquitetura, runbooks, planos e notas de experimento

## Requisitos

- Python 3.11 (pipeline principal)
- Ambiente virtual local (`.venv`)
- Dependências fixadas para reprodução em `requirements-lock.txt`

Instalação rápida:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

## Primeiros comandos

Treino Hydra (config principal):

```bash
source .venv/bin/activate
python train.py --config hydra_train_config.json
```

Avaliação:

```bash
source .venv/bin/activate
python eval.py --config hydra_train_config.json --split test
```

## Checkpoints oficiais locais (MultiROI)

A biblioteca MultiROI usa caminhos locais em `hydra-checkpoints/multiROI/`.

- Os binários `.ckpt` **não são versionados** neste repo.
- Apenas estrutura e documentação ficam no Git:
  - `hydra-checkpoints/multiROI/README.md`
  - `hydra-checkpoints/multiROI/center/.gitkeep`
  - `hydra-checkpoints/multiROI/lateral/.gitkeep`

Consulte também:
- `longoeixo/scripts/multiroi_composed_inference.py`

## Convenções importantes

- Artefatos pesados de treino/experimento são ignorados por `.gitignore`
- Scripts e configs legados são mantidos em:
  - `scripts antigos (NAO É PRA USAR!!!)/`

## Documentação recomendada

- `docs/arquitetura/HYDRA_UNET_MULTITASK_SPEC.md`
- `docs/runbooks/HYDRA_TRAINING_RUNBOOK.md`
- `docs/avaliacao/HYDRA_EVALUATION_PLAN.md`

## Status

Repositório em evolução contínua de pesquisa aplicada (baseline estável + ciclos rápidos de experimentação).
