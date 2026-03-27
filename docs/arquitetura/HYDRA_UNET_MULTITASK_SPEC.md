# HYDRA U-Net MultiTask - Especificação Técnica

Este documento define a arquitetura e as decisões técnicas para o modelo multitarefa de localização de long-eixo dentário.

## 1) Objetivo do modelo

Receber uma radiografia panorâmica e produzir:

1. **Mapa de pontos**: `64` canais de heatmaps gaussianos (2 pontos por dente, 32 dentes).
2. **Presença dentária**: `32` saídas binárias (presença/ausência por dente).

Assim, o modelo aprende simultaneamente localização geométrica e existência do dente.

## 2) Nome do modelo

- Nome interno: **Hydra U-Net MultiTask**
- Classe sugerida em código: `HydraUNetMultiTask`

## 3) Entradas e saídas

### Entrada

- Imagem grayscale `1 x 256 x 256`, `float32`, normalizada em `[0,1]`.

### Saídas

1. **Head de segmentação (decoder U-Net)**
   - shape: `64 x 256 x 256`
   - cada canal representa um ponto anatômico (heatmap gaussiano).

2. **Head de classificação de presença (encoder shared)**
   - shape: `32`
   - `logits` (usar `sigmoid` apenas para métrica/inferência).

## 4) Ground truth composto

### 4.1 Heatmaps (64 canais)

- `Y_heatmap`: shape final de treino `(64,256,256)`.
- Ordem canônica fixa:
  - dentes: `11..18, 21..28, 31..38, 41..48`
  - para cada dente: `p1`, `p2`
- Referência de ordem: `channel_order_64.txt` (quando gerado para auditoria).

### 4.1.1 Decisão oficial de geração (atualizada)

Para escala (incluindo 16k casos), o target **não deve ser pré-armazenado** em `stack64` no tamanho original.

Pipeline oficial:
1. Ler pontos do JSON no sistema original.
2. Projetar pontos para o grid final de treino (`256x256`).
3. Gerar heatmap gaussiano diretamente nesse grid (float32, em memória).
4. Entregar ao treino apenas `Y_heatmap (64,256,256)`.

Racional:
- evita crescimento de storage para múltiplos TB.
- mantém compatibilidade local/EC2 com fonte canônica (`imgs + data_longoeixo`).
- mantém largura gaussiana consistente entre exames no espaço de aprendizado do modelo.
- preserva nuance geométrica sem materializar targets gigantes em disco.

### 4.1.2 Escala da gaussiana (regra oficial)

- `sigma` deve ser definido em pixels do **grid de treino final** (`sigma_px_target`).
- Não usar `sigma` absoluto no tamanho original quando o target final é `256x256`, pois isso gera variação de largura entre resoluções de origem.
- A visualização de inspeção deve refletir o target real (sem suavização “cosmética”).

### 4.2 Presença dentária (32 canais)

- `Y_presence`: shape `(32,)`, binário (`0/1`).
- `Y_presence[i] = 1` se o dente canônico `i` existe no JSON.
- `Y_presence[i] = 0` se não existe.

Mapeamento entre presença e heatmap:
- dente `i` <-> canais `2*i` e `2*i+1`.
- presença pode ser derivada do JSON ou do heatmap da mesma amostra.

## 5) Arquitetura recomendada

## 5.1 Backbone (encoder)

### Recomendação principal
- **ResNet34** (melhor equilíbrio para 256x256 e escala para 16k casos).

### Estratégia por fase
- Fase protótipo (100 casos): ResNet18 opcional para iteração rápida.
- Fase produção (16k casos): ResNet34 como baseline final.

## 5.2 Decoder

- Decoder U-Net padrão com skip-connections.
- Saída de segmentação com 64 canais.

## 5.3 Head de presença

- A partir do bottleneck do encoder:
  - Global Average Pooling
  - MLP/Linear final para `32 logits`

## 6) Função de perda multitarefa

Perda total:

`L = λ_heatmap * L_heatmap + λ_presence * L_presence`

Com inicialização recomendada:
- `λ_heatmap = 1.0`
- `λ_presence = 0.3`

### 6.1 Heatmap loss

- `L_heatmap = 0.8 * MSE + 0.2 * SoftDice` (por canal).
- Opção de treino: `absent_heatmap_weight` (default `1.0`).
  - `1.0`: canais de dentes ausentes participam normalmente da loss de heatmap (comportamento legado).
  - `0.0`: canais de dentes ausentes não contribuem para loss de heatmap (a cabeça de presença segue supervisionada por BCE).
  - valores intermediários (`0 < w < 1`): penalização fraca para ausentes.

### 6.2 Presença loss

- `L_presence = BCEWithLogitsLoss` sobre 32 logits.

## 7) Inferência

1. Heatmaps: extrair ponto por canal via `argmax`.
2. Presença: `sigmoid(logit) > threshold` (ex.: 0.5).
3. Uso combinado recomendado:
   - se presença prevista de um dente for baixa, ignorar/rebaixar confiança dos 2 pontos correspondentes.

## 8) Racional das escolhas

1. MultiTask melhora robustez semântica (existência + localização).
2. Encoder compartilhado reduz custo e força features consistentes.
3. ResNet34 dá capacidade suficiente para anatomia dental sem custo extremo.
4. Heatmap gaussiano facilita aprendizagem de ponto com gradientes suaves.

## 9) Artefatos de dados relacionados

- Gerador de heatmaps: `longoeixo/scripts/generate_gaussian_point_maps.py`
- Preprocess 256: `longoeixo/scripts/unet256_preprocess.py`
- Preset treino: `longoeixo/presets/unet256_stack64_preset.json`

Nota:
- `stack64` em disco permanece suportado para debug/smoke/inspeção, mas não é a estratégia recomendada para escala.
