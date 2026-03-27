# Hydra Model API (PyTorch)

Arquivo principal:
- `hydra_multitask_model.py`

## Componentes

- `HydraUNetMultiTask`
  - entrada: `x` em `(B,1,H,W)`
  - saída:
    - `heatmap_logits`: `(B,64,H,W)`
    - `presence_logits`: `(B,32)`

- `derive_presence_from_stack64(stack64)`
  - converte target `(B,64,H,W)` em presença `(B,32)`

- `HydraMultiTaskLoss`
  - combina:
    - heatmap: `0.8*MSE + 0.2*SoftDice`
    - presença: `BCEWithLogits`
  - opcional: `absent_heatmap_weight` para reduzir/zerar contribuição de canais de dentes ausentes

## Exemplo de uso

```python
import torch
from hydra_multitask_model import (
    HydraUNetMultiTask,
    HydraMultiTaskLoss,
    derive_presence_from_stack64,
    build_attention_maps,
)

model = HydraUNetMultiTask(
    in_channels=1,
    heatmap_out_channels=64,
    presence_out_channels=32,
    backbone="resnet34",
)
criterion = HydraMultiTaskLoss(
    w_heatmap=1.0,
    w_presence=0.3,
    w_mse=0.8,
    w_dice=0.2,
    absent_heatmap_weight=1.0,  # default: comportamento legado (todos os canais contam)
)

x = torch.randn(2, 1, 256, 256)
y_heatmap = torch.rand(2, 64, 256, 256)
y_presence = derive_presence_from_stack64(y_heatmap)

pred = model(x, return_intermediates=True)
loss_out = criterion(pred, y_heatmap, y_presence)
loss = loss_out.total
loss.backward()

# mapas de atencao agregados (mean) ja no tamanho da entrada
att_mean = build_attention_maps(
    pred["intermediates"],
    mode="mean",
    normalize=True,
    out_size=(256, 256),
)
# idem com max
att_max = build_attention_maps(
    pred["intermediates"],
    mode="max",
    normalize=True,
    out_size=(256, 256),
)
```

## Observações

- Aplicar `sigmoid` em `presence_logits` apenas para métrica/inferência.
- Para extração de pontos dos heatmaps: `argmax` por canal.
- A ordem canônica dos 64 canais deve ser a mesma usada no gerador `stack64`.
- Camadas disponíveis para atenção: `enc_x1`, `enc_x2`, `enc_x3`, `bottleneck_x4`, `decoder_final`.
