# Hydra Longo-Eixo: Especificacao Oficial (Janelas Fixas + Lateral Compartilhada)

Data: 2026-03-23  
Status: **especificacao ativa para implementacao**

Este documento consolida a versao final discutida:
- 3 janelas fixas (sem dependencia de pontos anatomicos);
- rede lateral unica (lado direito canonico) com treino por espelhamento do lado esquerdo;
- rede centro com opcao separada simples ou compartilhada com espelhamento;
- particao de dentes sem redundancia.

Fonte de verdade de configuracao (codigo):
- `longoeixo/scripts/roi_lateral_shared_config.py`
- `longoeixo/scripts/multiroi_composed_inference.py`
- O modulo ja encapsula:
  - janelas fixas (`fixed_three_windows`)
  - conversoes global/local (`global_to_local`, `local_to_global`)
  - treino/restore centro com flip seguro (`center_prepare_train`, `center_restore_inference`)
  - treino lateral direita/esquerda (`lateral_prepare_right_train`, `lateral_prepare_left_train`)
  - restore de inferencia direita/esquerda (`lateral_restore_right_inference`, `lateral_restore_left_inference`)

Fonte de verdade de inferencia composta (producao):
- `longoeixo/scripts/multiroi_composed_inference.py`
- esta biblioteca passa a ser o contrato unico para executar:
  - split em 3 ROIs;
  - center + lateral (LEFT direto + RIGHT flipado);
  - unflip/remap/offset;
  - merge global de heatmaps;
  - extracao de pontos.

## 1) Motivacao

Objetivo principal:
- melhorar deteccao/localizacao de longoeixo preservando detalhe dentario;
- evitar fragilidade causada por exames com condilo/mentoniano ausentes;
- reduzir complexidade de saida por rede (64 -> 24/20).

Razoes para abandonar janelas dependentes de pontos:
- em exames recortados, `Condilo`/`Mentoniano` podem nao existir;
- erro desses pontos desloca toda ROI, propagando falha de pre-processamento.

## 2) Geometria das 3 Janelas Fixas (Sem Pontos)

Para imagem de largura `W` e altura `H`:

- `HALF = floor(W/2)`

1. `LEFT`
- `[x1, y1, x2, y2] = [0, 0, HALF, H]`

2. `RIGHT`
- `[x1, y1, x2, y2] = [W - HALF, 0, W, H]`

3. `CENTER`
- `x1 = floor((W - HALF)/2)`
- `x2 = x1 + HALF`
- `[x1, y1, x2, y2] = [x1, 0, x2, H]`

Consequencias:
- largura de cada janela = meia imagem;
- `CENTER` tem overlap de ~50% com `LEFT` e `RIGHT`;
- cobertura vertical total (`0..H`) em todas as janelas.

Nota de orientacao (dataset atual `longoeixo`):
- em quase todas as amostras, o **lado anatomico direito** aparece no lado **esquerdo da imagem**.
- portanto, para a rede lateral compartilhada, o ramo canonico deve usar a janela `LEFT`.

Exemplo para `W=2776, H=1480`:
- `LEFT   = [0, 0, 1388, 1480]`
- `CENTER = [694, 0, 2082, 1480]`
- `RIGHT  = [1388, 0, 2776, 1480]`

## 3) Definicao dos Dentes por Janela (Sem Redundancia)

Particao final:

```python
TOOTH_SETS_BY_RECT = {
    "LEFT":   ["24","25","26","27","28","34","35","36","37","38"],   # pre+molares esquerdo
    "RIGHT":  ["14","15","16","17","18","44","45","46","47","48"],   # pre+molares direito
    "CENTER": ["11","12","13","21","22","23","31","32","33","41","42","43"],  # incisivos+caninos
}
```

Saida esperada por rede:
- Rede `CENTER`: `12 dentes x 2 pontos = 24 canais`
- Rede `LATERAL_SHARED`: `10 dentes x 2 pontos = 20 canais`

Perfis de treino suportados:
- `center24`: center sem espelhamento (`N` amostras por epoca)
- `center24_shared_flip`: center com espelhamento+remapeamento (`2N` amostras por epoca)
- `lateral_shared20`: lateral compartilhada (`2N` amostras por epoca)

## 3.1) Mapeamento simetrico do CENTER (obrigatorio no flip)

No flip horizontal da janela CENTER, os labels devem ser remapeados:

```python
CENTER_FLIP_MAP = {
    "11":"21", "12":"22", "13":"23",
    "21":"11", "22":"12", "23":"13",
    "41":"31", "42":"32", "43":"33",
    "31":"41", "32":"42", "33":"43",
}
```

Sem esse remapeamento, a semantica do dente fica invertida no espelho e o treino degrada.

## 4) Rede Lateral Compartilhada (Ideia Central)

### 4.1 Convencao canonica da rede lateral

A rede lateral usa **apenas labels do lado direito** como espaco canonico:

```python
LATERAL_RIGHT_TEETH = ["14","15","16","17","18","44","45","46","47","48"]
```

### 4.2 Treino da lateral

Entrada A (canonica, direito anatomico):
- crop `LEFT` da imagem original;
- labels direitos diretos (`14..18`, `44..48`).

Entrada B (espelhada, esquerdo anatomico):
- crop `RIGHT` **flipado horizontalmente**;
- labels esquerdos remapeados para direito:

```python
LEFT_TO_RIGHT = {
    "24":"14","25":"15","26":"16","27":"17","28":"18",
    "34":"44","35":"45","36":"46","37":"47","38":"48",
}
```

Com isso, a lateral aprende um unico problema (lado direito canonico), com dados dos dois lados.

### 4.3 Inferencia da lateral

1. Direito anatomico:
- usa janela `LEFT` direto -> predicoes em labels direitos.

2. Esquerdo anatomico:
- usa janela `RIGHT` flipada -> predicoes em labels direitos canonicos;
- desfaz flip das coordenadas no crop;
- remapeia labels para esquerdo:

```python
RIGHT_TO_LEFT = {
    "14":"24","15":"25","16":"26","17":"27","18":"28",
    "44":"34","45":"35","46":"36","47":"37","48":"38",
}
```

3. Merge global:
- combina lateral-direita + lateral-esquerda-desflipada + centro;
- sem competicao entre redes (particao deterministica).

## 5) Regras Sem Ambiguidade (Implementacao)

1. Ordem de canais (obrigatoria):
- centro: `11_p1, 11_p2, 12_p1, 12_p2, ...`
- lateral: `14_p1, 14_p2, 15_p1, 15_p2, ...`

2. Convencao de flip horizontal:
- no eixo x absoluto: `x' = (W - 1) - x`
- aplicar/desfazer no sistema de coordenadas do crop correspondente.

3. Reprojecao de coordenadas:
- predicoes no crop devem voltar para coordenadas da imagem completa;
- manter mesma convencao para treino/inferencia/avaliacao.

4. Dente fora da particao:
- deve ser ignorado pela rede que nao o modela;
- nao pode entrar em loss de cabeca errada.

5. Sem redundancia de labels:
- cada dente pertence a uma unica rede no merge final.

6. CENTER flipado:
- aplicar flip de `X` no sistema local do ROI CENTER;
- aplicar remapeamento `CENTER_FLIP_MAP`;
- no restore, desfazer flip local + offset global.

### 5.1 Contrato recomendado para nao errar merge

Use sempre as funcoes encapsuladas do modulo oficial:

0. Centro (prepare/restore com ou sem flip):
- `center_prepare_train(tooth_center, pt_global, W, H, flip_horizontal)`
- `center_restore_inference(tooth_center_pred, pred_local, W, H, came_from_flipped_input)`

1. Treino lateral direita:
- `lateral_prepare_right_train(tooth_right, pt_global, W, H)`

2. Treino lateral esquerda (com flip+relabel):
- `lateral_prepare_left_train(tooth_left, pt_global, W, H)`

3. Inferencia lateral direita (local->global):
- `lateral_restore_right_inference(tooth_right, pred_local, W, H)`

4. Inferencia lateral esquerda (unflip+relabel+local->global):
- `lateral_restore_left_inference(tooth_right, pred_local_flipped, W, H)`

Regra operacional:
- para o lado esquerdo, **nunca** fazer merge manual de offset/flip/label.
- sempre passar pelo `lateral_restore_left_inference`.

## 6) Alertas e Cuidados Criticos

1. **Erro mais perigoso**: flipar coordenada e esquecer remapeamento de label.
- Isso gera predicao "espelhada semanticamente errada".

2. **Erro comum**: remapear label e esquecer desfazer flip da coordenada.
- Resultado: label correta, posicao fisica errada.

3. **Off-by-one** em `x' = W-1-x`:
- usar exatamente essa formula para pixel absoluto.

4. **Mistura de sistemas de coordenadas**:
- nunca misturar coordenada relativa do crop com coordenada global sem conversao explicita.

5. **Consistencia de order de canais**:
- qualquer desvio na ordem quebra treino e avaliacao silenciosamente.

6. **CENTER flip sem remap**:
- erro grave que cria supervisao semanticamente incorreta.

7. **CENTER flip com remap mas sem unflip no restore**:
- label pode parecer correta, mas a coordenada final fica no lado errado.

## 7) Checklist de Validacao Antes de Treino Longo

1. Self-check do modulo de configuracao:
- `python3 longoeixo/scripts/roi_lateral_shared_config.py`

2. Auditoria rapida de labels/canais:
- confirmar 24 canais no centro e 20 na lateral.

3. Teste de round-trip no lado esquerdo:
- flip input -> inferencia mock -> unflip + remap -> conferir se dentes voltam ao lado correto.

4. Auditoria visual em lote:
- inspecionar overlays com as 3 janelas fixas em amostras diversas.

## 8) Compatibilidade com o Plano Anterior

O plano antigo baseado em pontos anatomicos permanece como referencia historica, mas a implementacao ativa deve seguir esta especificacao de janelas fixas.

## 9) Comandos Oficiais de Treino (V1)

Rede CENTER (24 canais):

```bash
/Users/fabioandrade/hydra/.venv/bin/python /Users/fabioandrade/hydra/train.py \
  --config /Users/fabioandrade/hydra/hydra_train_config_roi_center24_v1.json \
  --run-name center24_v1_<tag>
```

Rede CENTER_SHARED_FLIP (24 canais, 2N por epoca):

```bash
/Users/fabioandrade/hydra/.venv/bin/python /Users/fabioandrade/hydra/train.py \
  --config /Users/fabioandrade/hydra/hydra_train_config_roi_center24_sharedflip_v1.json \
  --run-name center24_sharedflip_v1_<tag>
```

Rede LATERAL_SHARED (20 canais):

```bash
/Users/fabioandrade/hydra/.venv/bin/python /Users/fabioandrade/hydra/train.py \
  --config /Users/fabioandrade/hydra/hydra_train_config_roi_lateral_shared20_v1.json \
  --run-name lateral_shared20_v1_<tag>
```

Smoke tests usados na validacao desta implementacao:
- `smoke_center24_v1`
- `smoke_center24_sharedflip_v1`
- `smoke_lateral_shared20_v1`

## 10) Alerta Pratico de Loss (Lateral)

Se `absent_heatmap_weight=0.0`, batches com muitos dentes ausentes podem reduzir
fortemente (ou zerar) contribuicao de `heatmap_loss` naquele batch.

Recomendacao inicial:
- comecar com `absent_heatmap_weight` entre `0.1` e `0.3` na lateral;
- monitorar `val_heatmap_loss` para evitar regime dominado apenas por `presence_bce`.

## 11) Decisoes Finais Consolidadas (Implementacao Atual)

1. Geometria de ROI:
- usar somente `fixed_three_windows(W,H)` (independente de pontos anatomicos).

2. Lateral compartilhada:
- orientacao canônica da rede lateral = direito anatomico;
- no dataset atual, isso corresponde ao ROI `LEFT` da imagem;
- ramo complementar = ROI `RIGHT` flipado + remapeamento `LEFT_TO_RIGHT`.

3. Centro:
- perfil base `center24` mantido;
- perfil novo `center24_shared_flip` habilitado para dobrar amostras (`2N`);
- espelhamento central exige remapeamento simetrico de labels (`CENTER_FLIP_MAP`).

4. Contrato de coordenadas (critico):
- flips sempre no sistema local do ROI (`x' = w_roi - 1 - x`);
- offsets para global sempre via `local_to_global`;
- proibido merge manual de flip+offset sem funcoes encapsuladas.

5. Estado de validacao:
- self-check do modulo de ROI cobre round-trip lateral e center (com e sem flip);
- smoke de treino executado com sucesso para:
  - `center24_shared_flip`
  - `center24_shared_flip_nopres_absenthm1`
  - (lateral compartilhada ja validada anteriormente).

## 12) Contrato Oficial de Inferencia Composta (Obrigatorio)

### 12.1 Regra de governanca (sem ambiguidade)

A partir desta revisao, a forma **oficial e unica** de rodar as redes Multi-ROI
para inferencia combinada e:

- usar `longoeixo/scripts/multiroi_composed_inference.py` como biblioteca central.

Nao e permitido em novos scripts de producao:
- refazer manualmente flip/unflip/offset/remapeamento de labels;
- duplicar logica de split+merge fora da biblioteca;
- extrair pontos por caminho alternativo que nao passe pelo contrato da lib.

Objetivo:
- eliminar drift de implementacao;
- evitar regressao geometrica silenciosa;
- garantir paridade entre ferramentas de debug, scripts de lote e servico de API.

### 12.2 API oficial minima da biblioteca

Funcoes de entrada:
- `load_multiroi_models(center_ckpt, lateral_ckpt, device=None)`
- `infer_multiroi_from_image(image_gray, models, threshold=0.1)`
- `infer_multiroi_from_path(image_path, models, threshold=0.1)`

Saida padrao:
- `MultiROIInferenceResult` com:
  - `predictions` (lista de `ToothPrediction`);
  - `heatmaps.global_max` (heatmap composto na panoramica inteira);
  - mapas por ramo e metadados de ROI/shape.

### 12.3 Paridade operacional obrigatoria

A biblioteca oficial implementa (e deve preservar):
- `argmax` em **logits crus** para coordenadas dos pontos;
- score por dente = `min(peak_p1, peak_p2)` no mesmo espaco de logits;
- lateral direita anatomica via ROI `LEFT` (direto);
- lateral esquerda anatomica via ROI `RIGHT` flipado + unflip + remapeamento;
- merge de heatmaps globais por `max(center, left, right_unflip)`;
- mapeamento 256->ROI com convencao half-pixel consistente com `cv2.resize`.

### 12.4 Script consumidor de referencia (visualizacao)

O script de mosaico que consome apenas a biblioteca oficial e:
- `longoeixo/scripts/infer_multiroi_overlay_mosaic_lib.py`

Uso:
- renderiza paineis e HTML;
- nao pode reimplementar a geometria da inferencia.

### 12.5 Evolucao futura

Qualquer melhoria de inferencia (ex.: centroid fallback, regras de confianca,
novos checkpoints/modelos) deve ser introduzida primeiro em
`multiroi_composed_inference.py`, mantendo compatibilidade de contrato.
