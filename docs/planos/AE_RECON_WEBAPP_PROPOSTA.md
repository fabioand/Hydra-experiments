# AE Reconstruction Web App - Proposta de Projeto

Data: 2026-03-23  
Workspace: `/Users/fabioandrade/hydra`

## 1) Contexto e objetivo

Queremos transformar o utilitário local `cv2` em um web app com UX moderna, mantendo o mesmo comportamento matemático.

Script de referência (fonte de verdade):
- `ae_radiograph_filter/scripts/show_single_reconstruction_cv2.py`

Comportamento atual do script:
1. Carrega imagem em grayscale.
2. Redimensiona para `input_size` (default 256).
3. Roda autoencoder `PanoramicResNetAutoencoder` (`backbone=resnet34`).
4. Faz upscale da reconstrução para o tamanho original (`cv2.INTER_LANCZOS4`).
5. Aplica operação com 2 sliders (`fs`, `fa`) usando a fórmula:

```text
enhanced = clip(original - fs*recon + fa*(original - recon), 0, 1)
```

6. Exibe: original, reconstrução e imagem realçada.

## 2) Escopo funcional do web app

Fluxo do usuário:
1. Usuário abre a página.
2. Faz upload por botão (`input file`) OU drag&drop.
3. Front envia a imagem para backend.
4. Backend executa inferência do AE e devolve reconstrução (já no tamanho original) + metadados.
5. Front mostra original e reconstrução.
6. Front aplica sliders em tempo real para gerar `enhanced` no cliente.

Requisitos funcionais:
- Upload de arquivo local (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`).
- Drag&drop na página.
- Dois sliders com faixa `0..1` (resolução 0.001):
  - `fs x recon`
  - `fa x (orig-recon)`
- Preview em tempo real sem nova chamada ao backend quando sliders mudarem.
- Botão de reset dos sliders para defaults do script:
  - `fs=0.300`
  - `fa=1.000`

## 3) Arquitetura recomendada

## 3.1 Backend

Recomendação: **FastAPI + Uvicorn + PyTorch + OpenCV**

Motivos:
- Python já é stack natural do workspace (PyTorch/OpenCV já usados).
- FastAPI facilita API tipada, validação e docs OpenAPI.
- Inicialização única do modelo em memória (baixa latência por requisição).

Responsabilidades do backend:
- Validar arquivo recebido.
- Carregar em grayscale.
- Preprocessar para tamanho do modelo (256x256).
- Rodar inferência do AE.
- Redimensionar reconstrução para tamanho original (LANCZOS4).
- Retornar original e reconstrução em formato eficiente para o front.

## 3.2 Frontend

Recomendação principal: **React + TypeScript + Vite + Canvas 2D**

Motivos:
- Melhor controle de UX interativa e drag&drop.
- Sliders em tempo real com operação pixel a pixel no browser (sem roundtrip).
- Fácil evolução para novas ferramentas visuais depois.

Alternativa de MVP rápido:
- **Streamlit** (muito alinhado ao histórico do pool `ai-training`), porém menos flexível para UI fina e menos fluido para renderizações contínuas por slider.

Decisão sugerida:
- MVP inicial pode ser Streamlit se o objetivo for validar em 1-2 dias.
- Para produto interno estável e evolutivo, seguir com React+TS.

## 4) Contrato de API (proposta)

Endpoint principal:
- `POST /v1/reconstruct`

Request:
- `multipart/form-data`
- campo `file`: imagem
- campo opcional `input_size` (default 256)

Response (opção recomendada):
- `application/json` com:
  - `width`, `height`
  - `original_png_base64`
  - `reconstruction_png_base64`
  - `defaults`: `{ "fs": 0.3, "fa": 1.0 }`
  - `meta`: `{ "model": "resnet34", "ckpt": "..." }`

Observação:
- Para imagens maiores ou alto volume, migrar para resposta binária com compactação/stream ou URLs temporárias.

Endpoint de saúde:
- `GET /healthz` retorna status e device (`cuda/mps/cpu`).

## 5) Distribuição de responsabilidades backend vs frontend

Backend:
- Inferência e geração da reconstrução.
- (Preferencial) já devolver reconstrução no tamanho original.

Frontend:
- Exibir original e reconstrução.
- Calcular `enhanced` dinamicamente com sliders (`Float32Array`/Canvas).
- Renderizar sem novas chamadas ao backend durante ajuste dos sliders.

## 6) Performance esperada

- Inferência: custo principal no backend (única vez por imagem).
- Sliders: custo principal no frontend (aritmética local por frame).
- Estratégia para fluidez:
  - armazenar buffers normalizados `[0,1]` de `original` e `reconstruction`.
  - recalcular `enhanced` com loop único vetorizado por JS/TypedArray.
  - desenhar em `canvas` (evitar re-encode PNG a cada interação).

## 7) Segurança e hardening mínimo

- Limite de tamanho de upload (ex.: 20 MB).
- Validação de mime/type e extensão.
- Timeout por requisição.
- Não persistir arquivos no disco por padrão (processamento in-memory).
- Log sem dados sensíveis da imagem.

## 8) Integração com o pool de projetos do workspace

Projetos relacionados já existentes:
- `ae_radiograph_filter` (filtro por erro de reconstrução e scripts de inspeção).
- `panorama_foundation` (modelo AE e base de treino).
- `workspace_repos/ai-training/present-rm` (histórico de demos Streamlit).

Conclusão de encaixe:
- Back em Python reutiliza diretamente modelo/checkpoint já existentes.
- Front React entrega melhor UX para o caso de sliders contínuos.
- Se for necessário “time-to-demo” extremo, usar Streamlit como etapa intermediária.

## 9) Plano de implementação (fases)

Fase 1 (MVP técnico):
1. Criar serviço FastAPI com carregamento único do checkpoint.
2. Implementar `POST /v1/reconstruct` reproduzindo 1:1 o script `cv2`.
3. Teste manual com 3 imagens reais do workspace.

Fase 2 (UI web):
1. Página React com upload + drag&drop.
2. Painel com original/recon/enhanced.
3. Dois sliders com defaults (`0.3`, `1.0`) e reset.
4. Operação matemática no cliente igual ao script.

Fase 3 (qualidade):
1. Testes unitários backend (pré-processamento e pós-processamento).
2. Teste de regressão visual simples (diferença máxima tolerada).
3. Dockerfile e script de execução local.

Fase 4 (opcional):
1. Export de imagem final realçada.
2. Presets de sliders.
3. Lote de imagens.

## 10) Riscos e decisões em aberto

Riscos:
- Diferenças numéricas pequenas entre OpenCV/Python e Canvas/JS.
- Imagens muito grandes podem reduzir FPS de slider.

Decisões:
1. Front final: React (recomendado) ou Streamlit (MVP rápido).
2. Formato de resposta: base64 JSON (simples) ou binário/URL temporária (mais eficiente).
3. Deploy alvo: local interno, VM dedicada ou container em serviço web.

## 11) Resumo executivo

O projeto é totalmente viável e bem alinhado ao workspace atual.

Recomendação objetiva:
- **Backend**: FastAPI em Python, reaproveitando `PanoramicResNetAutoencoder` e a lógica do script.
- **Frontend**: React + TypeScript + Canvas, para sliders suaves e UX superior.
- **MVP**: entregar em fases curtas, mantendo paridade matemática com `show_single_reconstruction_cv2.py` desde o primeiro incremento.
