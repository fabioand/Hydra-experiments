# Preset de Treino U-Net 256x256 (Stack64)

Este preset define um pipeline padrão para treinar U-Net com entrada radiográfica e saída de 64 heatmaps gaussianos de pontos de long-eixo dentário.

## Arquivos do preset

- Config: `longoeixo/presets/unet256_stack64_preset.json`
- Utilitário de preprocess: `longoeixo/scripts/unet256_preprocess.py`
- Geração dos heatmaps base: `longoeixo/scripts/generate_gaussian_point_maps.py`

## Decisão de pipeline

- Gerar gaussianas no tamanho original (ground truth mais fiel).
- Reduzir **imagem + máscara** para 256x256 no carregamento para treino.

## Convenções

- Entrada da U-Net: `x` com shape `(1, 256, 256)`, `float32`, faixa `[0,1]`.
- Saída/target: `y` com shape `(64, 256, 256)`, `float32`, faixa `[0,1]`.
- Ordem dos 64 canais: arquivo `channel_order_64.txt` da geração `stack64`.

## Interpolação recomendada

- Imagem: `cv2.INTER_AREA` para downscale.
- Máscara gaussiana: `cv2.INTER_LINEAR` + `clip(0,1)`.

## Comando para gerar targets stack64 (original)

```bash
.venv/bin/python longoeixo/scripts/generate_gaussian_point_maps.py \
  --output-mode stack64 \
  --imgs-dir longoeixo/imgs \
  --json-dir longoeixo/data_longoeixo \
  --out-dir longoeixo/gaussian_maps_stack64 \
  --sigma 7.0
```

## Exemplo de preprocess de 1 amostra para 256x256

```bash
.venv/bin/python longoeixo/scripts/unet256_preprocess.py \
  --image longoeixo/imgs/ararasradiodonto.radiomemory.com.br_373829_11994.jpg \
  --mask longoeixo/gaussian_maps_stack64/ararasradiodonto.radiomemory.com.br_373829_11994.npy \
  --out-x /tmp/x_256.npy \
  --out-y /tmp/y_256.npy \
  --out-overlay /tmp/overlay_256.png
```

## Augmentacoes recomendadas

- Nao usar flip horizontal nem vertical (preserva lateralidade odontologica por canal).
- Manter transformacoes geometricas leves: rotacao pequena, scale e translacao pequenas.
- Aplicar ruidos leves apenas na imagem de entrada (nao na mascara):
  - ruído gaussiano aditivo fraco
  - ruído poisson leve
  - ruído speckle leve

Referencia: `longoeixo/presets/unet256_stack64_preset.json`

## Recomendação inicial de treino

- Modelo: U-Net encoder-decoder 2D, saída 64 canais.
- Loss inicial: `0.8*MSE + 0.2*SoftDice` por canal.
- Otimizador: AdamW (`lr=3e-4`, `weight_decay=1e-4`).
- Batch size inicial: 8 (ajustar por VRAM).
- AMP: habilitado.

## Sanidade antes de treinar

1. Validar shapes (`x: 1x256x256`, `y: 64x256x256`).
2. Validar ranges (`x` e `y` em `[0,1]`).
3. Confirmar canais de dentes ausentes com mapa zerado.
4. Conferir overlays de inspeção visual.
5. Garantir mesma ordem canônica de canais no treino e na inferência.


## Execucao Portavel (Local/EC2)

Com a mesma estrutura de pastas (`longoeixo/imgs` e `longoeixo/data_longoeixo`), os mesmos comandos funcionam no dev e na EC2:

```bash
# na raiz do repo
bash longoeixo/setup_env.sh
bash longoeixo/run_generate_stack64.sh longoeixo/gaussian_maps_stack64
```
