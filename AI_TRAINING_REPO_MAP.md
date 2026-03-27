# AI-Training: Mapa Completo do Repositório

Documento de referência consolidado do repositório:
`/Users/fabioandrade/hydra/workspace_repos/ai-training`

Objetivo: evitar nova varredura de arquivos para entender estrutura, projetos, tecnologias, maturidade e pontos de uso.

## 1. Visão Geral

`ai-training` é um monorepo de treinamento de modelos de IA (foco odontológico/radiografias), com organização por:

- `libs/`: bibliotecas compartilhadas (redes, inferência, datasets, utilitários, visualização).
- `train/`: projetos de treinamento por tarefa.
- `present-rm/` e `present-rmia/`: apps Streamlit de demonstração/análise.
- `docker/`: empacotamento e execução (CPU/GPU).

O padrão dominante nos projetos de treino é:

- Configuração com Hydra/OmegaConf
- Treino com PyTorch Lightning
- Tracking remoto com ClearML
- Export para ONNX ao final

## 2. Stack Tecnológica

- Linguagem: Python
- Deep Learning: PyTorch, PyTorch Lightning, TorchMetrics
- Configuração: Hydra, OmegaConf
- CV/augmentação: OpenCV, Pillow, Albumentations, scikit-image
- Métricas/análise: numpy, pandas, scikit-learn
- MLOps/experimentos: ClearML
- Inferência: ONNX, ONNXRuntime, clientes HTTP
- Frontend de análise: Streamlit
- Empacotamento: Docker (CPU/GPU), setup.py por lib

## 3. Estrutura Funcional

## 3.1 Bibliotecas compartilhadas (`libs/`)

- `libs/rmnetworks`
  - Implementa arquiteturas base: UNet/AttUNet/AEUNet, classificadores, Siamese, CMT, componentes de detecção.
- `libs/rmdatasets`
  - Utilidades de dataset e classes base.
- `libs/rmutils`
  - Utilitários gerais (sem dependência de framework DL por design do README), incluindo transforms, heatmaps, métricas, curvas/bezier, ruído.
- `libs/rmvis`
  - Visualização de outputs (detecção, keypoints, segmentação, heatmaps).
- `libs/rmtorch`
  - Utilitários Torch/ONNX (export de modelos).
- `libs/inference`
  - Camada de inferência e integração API:
    - cliente para endpoints (`ApiConnect`)
    - loaders de modelos
    - classes de output para longaxis/detection
    - configs de modelos em YAML (`models_configs/*`)
- `libs/rmpresent`
  - Métricas para páginas de apresentação.

## 3.2 Projetos de treino (`train/`)

- `train/semantic_segmentation`
  - Segmentação semântica em diversos cenários (teeth, panorogram, anomalies, axial, periapical etc).
- `train/semantic_segmentation_ae`
  - Variante de segmentação com arquitetura AE/Att.
- `train/semantic_keypoints`
  - Regressão de pontos anatômicos/longaxis por heatmaps.
- `train/semantic_keypoints_classification`
  - Modelo conjunto: keypoints + classificação.
- `train/classification`
  - Classificação geral para múltiplas tarefas/datasets (inclusive panorâmica).
- `train/cmt_classification`
  - Classificação com arquitetura CMT.
- `train/object_detection`
  - Detecção de objetos (ex.: implantes) com Faster R-CNN.
- `train/regression`
  - Regressão em panorâmicas (ex.: idade).
- `train/regression_cephalometric`
  - Regressão com dados cefalométricos.
- `train/similarity_classification`
  - Classificação de similaridade (Siamese), ex.: periapicais de docs diferentes.
- `train/autoencoder`
  - Autoencoder para coordenadas/denoising.
- `train/name_email`
  - Tarefa tabular/textual específica (similaridade nome/email).
- `train/position_anomallies`
  - Scripts auxiliares de preparação/upload de dataset.

## 3.3 Apresentação/inspeção (`present-rm`, `present-rmia`)

Apps Streamlit com páginas para:

- comparação de modelos por task_id (ClearML artifacts),
- inspeção de divergências,
- análise visual de predição vs anotação,
- métricas agregadas e por faixas (incluindo páginas de gênero, idade, anomalias, similaridade, detecção etc).

## 3.4 Docker/execução

- `docker/Dockerfile.cpu` e `docker/Dockerfile.gpu`
- artefatos e compose para fluxos específicos (ex.: semantic segmentation).

Há traços de fluxo legado com Pants/PEX (comentado no README e arquivos `BUILD`), além do fluxo Python direto.

## 4. Projeto de Sexo por Radiografia Panorâmica (`panoramic_gender`)

Projeto identificado em:

- Config dataset: `train/classification/config/dataset/panoramic_gender.yaml`
- Dataset class: `train/classification/classification/datasets/panoramic_gender.py`
- Treino base: `train/classification/train.py`
- Modelo base: `train/classification/classification/models.py`
- Página de análise: `present-rmia/ia_demo/pages/10_panoramic_gender.py`

## 4.1 O que faz

Classificação binária de sexo (man/woman) em radiografia panorâmica.

- Campo de origem no metadata: `sexo`
  - `"0"` -> `man`
  - `"1"` -> `woman`
- Target configurado como multilabel com uma classe: `["woman"]` (threshold 0.5).

## 4.2 Como funciona

- Dataset:
  - baixa dados via ClearML Dataset ID,
  - divide em `train/val/test` usando `val_test_set.json`,
  - faz balanceamento no treino com oversampling,
  - carrega imagem em grayscale (`input_channels: 1`).
- Treino:
  - backbone ResNet (via `get_torchvision_resnet`),
  - loss BCEWithLogits,
  - métricas com F1 e classification report no teste,
  - export ONNX.
- Avaliação visual:
  - Streamlit com:
    - métricas globais,
    - análise por faixa etária,
    - listagem de casos divergentes com imagem.

## 4.3 Maturidade estimada

Status: protótipo avançado/experimental maduro.

Sinais positivos:

- Pipeline completo de treino + teste + artifact + demo visual.
- Estrutura operacional integrada ao padrão do repo.

Limites atuais:

- Sem documentação dedicada de critérios de aceitação clínica.
- Sem suíte de testes específica para essa task.
- Config default do módulo `classification` aponta para outra task (`name_email`), então gênero não é o foco principal atual.

## 4.4 Onde fica o dataset

Não está versionado no Git.

- Dataset remoto via ClearML:
  - `dataset_id: 335817d6a84d44fdacfa5d7e073a58fc`
  - definido em `train/classification/config/dataset/panoramics.yaml`

Em runtime:

- é baixado por `clearml.Dataset.get(...).get_local_copy()`
- estrutura esperada local:
  - `<dataset_root>/imgs/*.jpg`
  - `<dataset_root>/metadata/*.json`
  - `<dataset_root>/metadata/val_test_set.json` (quando `split_instance: local`)

## 5. ClearML vs TensorBoard (contexto do repo)

- TensorBoard: foco em visualização de curvas/métricas.
- ClearML: cobre experiment tracking completo (runs, configs, artefatos, datasets versionados, execução remota por filas/workers).

No `ai-training`, ClearML é usado como espinha dorsal operacional, e TensorBoard aparece como logger local por run.

## 6. Rodar treino com dataset remoto + GPU remota: viabilidade

Sim, o repo está preparado para isso.

Condições necessárias:

1. `clearml.conf` válido no ambiente.
2. Worker remoto com GPU rodando `clearml-agent` na fila esperada (`default` nos scripts atuais).
3. Dependências compatíveis no worker (Torch/CUDA conforme projeto).
4. Permissão para acessar os datasets remotos (IDs do ClearML).

Observação:

- Nem todos os subprojetos têm o mesmo grau de manutenção; alguns podem exigir ajuste pontual.
- O caminho mais seguro para primeiro teste é iniciar por `train/classification` com dataset conhecido.

## 7. Qualidade geral e pontos de atenção

Força do repo:

- amplitude de tarefas cobertas,
- padrão relativamente consistente de treino/avaliação/export,
- boa base de reaproveitamento em `libs`.

Atenções:

- existem sinais de legado/inconsistência em alguns arquivos (typos, imports antigos e variações históricas),
- documentação é menor que a implementação (muito conhecimento está no código/config).

## 8. Resumo Executivo

`ai-training` é uma base robusta de P&D/treino para IA odontológica, com:

- bibliotecas reutilizáveis,
- múltiplos pipelines de treino,
- integração prática com ClearML,
- demos Streamlit para análise qualitativa.

Para uso em novos projetos, a estratégia recomendada é:

1. reaproveitar `libs` e padrões de treino,
2. selecionar uma task estável como baseline,
3. endurecer com testes e documentação de critérios de aceitação por caso de uso.
