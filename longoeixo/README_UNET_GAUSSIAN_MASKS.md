# Geração de Máscaras Gaussianas para Treino de U-Net (Long-Eixo)

Este documento descreve como gerar e usar máscaras gaussianas de 64 canais para treinar uma U-Net que recebe a radiografia e prevê os pontos de long-eixo dos dentes.

## Objetivo

- **Entrada da rede**: imagem radiográfica (JPG).
- **Saída da rede**: tensor com **64 canais**.
- Cada canal representa **1 ponto anatômico** (ponta de um segmento de long-eixo).
- Como cada dente tem 2 pontos (`p1`, `p2`) e usamos 32 dentes, temos `32 x 2 = 64` canais.

## Script oficial

- Script: `longoeixo/scripts/generate_gaussian_point_maps.py`

Esse script lê:
- imagens: `longoeixo/imgs/*.jpg`
- anotações: `longoeixo/data_longoeixo/*.json`

E salva:
- máscaras `.npy`
- previews opcionais
- overlays opcionais

## Formato da anotação de origem (JSON)

Cada item da anotação tem, entre outros campos:
- `label`: dente (ex.: `11`, `24`, `47`)
- `pts`: lista com 2 pontos (`x,y`)

Exemplo simplificado:

```json
{
  "label": "11",
  "pts": [
    {"x": 1399.8, "y": 909.2},
    {"x": 1395.9, "y": 577.0}
  ]
}
```

## Como a nuvem gaussiana é gerada

Para cada ponto anotado `(x, y)`:

1. Converte para pixel com arredondamento:
   - `px = round(x)`
   - `py = round(y)`
2. Aplica um kernel gaussiano 2D local:

\[
G(dx,dy) = \exp\left(-\frac{dx^2 + dy^2}{2\sigma^2}\right)
\]

3. O patch é inserido no mapa no raio definido (`radius`, padrão `ceil(3*sigma)`).
4. O valor no pixel central é forçado para **1.0**.

Resultado:
- fundo preto (`0.0`)
- pico da anotação em `1.0`
- decaimento suave radial gaussiano

## Modo recomendado para U-Net: `stack64`

Use `--output-mode stack64`.

Nesse modo:
- saída por amostra: `np.ndarray float32` com shape **`(64, H, W)`**
- valores em `[0,1]`
- cada canal tem no máximo 1 ponto gaussiano
- se o dente não existir na imagem, os canais correspondentes ficam **zerados**

## Ordem canônica fixa dos 64 canais

A ordem é fixa e determinística:

- dentes: `11..18, 21..28, 31..38, 41..48`
- para cada dente: primeiro `p1`, depois `p2`

Exemplo inicial:
- canal `00`: `11_p1`
- canal `01`: `11_p2`
- canal `02`: `12_p1`
- canal `03`: `12_p2`
- ...

O script salva essa ordem em:
- `channel_order_64.txt` dentro do `--out-dir`

## Comando padrão (treino)

Rodar na raiz do repositorio:

```bash
.venv/bin/python longoeixo/scripts/generate_gaussian_point_maps.py \
  --imgs-dir longoeixo/imgs \
  --json-dir longoeixo/data_longoeixo \
  --out-dir longoeixo/gaussian_maps_stack64 \
  --output-mode stack64 \
  --sigma 7.0 \
  --num-samples 100
```

## Comandos úteis

Gerar overlays para inspeção visual:

```bash
.venv/bin/python longoeixo/scripts/generate_gaussian_point_maps.py \
  --out-dir longoeixo/gaussian_maps_stack64_overlay \
  --output-mode stack64 \
  --num-samples 10 \
  --save-overlay-png \
  --save-preview-png
```

Visualização interativa (janela OpenCV):

```bash
.venv/bin/python longoeixo/scripts/generate_gaussian_point_maps.py \
  --out-dir longoeixo/gaussian_maps_stack64_view \
  --output-mode stack64 \
  --num-samples 10 \
  --show-window
```

## Arquivos de saída

Para cada imagem `NOME.jpg`, o script gera:

- `NOME.npy`
  - `single`: shape `(H, W)`
  - `stack64`: shape `(64, H, W)`
- opcional: `preview_png/NOME.png`
- opcional: `overlay_png/NOME.png`

## Recomendações para treinamento

- Use `stack64` como target da U-Net.
- Normalize a imagem de entrada de forma consistente (ex.: `[0,1]` ou z-score).
- Garanta que a rede produza 64 canais na mesma ordem canônica.
- Use loss por canal (ex.: MSE/BCE focalizada em heatmap) e monitore métricas de localização por pico.
- Para inferência de ponto, extraia `argmax` por canal no mapa predito.

## Checagens de sanidade antes do treino

1. Verificar shape de targets: `(64, H, W)`.
2. Verificar faixa de valores: `min >= 0`, `max <= 1`.
3. Confirmar canais vazios para dentes ausentes.
4. Confirmar correspondência visual com overlays (`overlay_png`).
5. Confirmar leitura do `channel_order_64.txt` pelo pipeline de treino.

## Observações

- O modo `single` permanece disponível para debug/visualização geral.
- Para treinamento multi-canal de pontos, prefira sempre `stack64`.

## Preset 256x256 para treino

- Guia dedicado: `longoeixo/README_UNET256_PRESET.md`
- Config JSON: `longoeixo/presets/unet256_stack64_preset.json`
- Utilitario de preprocess: `longoeixo/scripts/unet256_preprocess.py`



## EC2 Quickstart

Como a arvore na EC2 e a mesma (`longoeixo/imgs` e `longoeixo/data_longoeixo`), use:

```bash
# na raiz do repo
bash longoeixo/setup_env.sh
bash longoeixo/run_generate_stack64.sh longoeixo/gaussian_maps_stack64
```

Para testar rapido com poucas amostras:

```bash
bash longoeixo/run_generate_stack64.sh longoeixo/gaussian_maps_stack64_smoke 10
```
