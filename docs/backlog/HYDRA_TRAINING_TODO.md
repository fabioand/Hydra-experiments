# HYDRA Training TODO (Teste Local End-to-End)

Checklist objetivo do que falta para iniciar e validar o treinamento local completo.

## 1) Dataset + DataLoader

Implementar pipeline de leitura:
- entrada: `jpg` grayscale
- target heatmap: gerado on-the-fly a partir do `json` diretamente no grid de treino
- target presença: derivado do `json` (ou do `stack64` quando em modo precomputed/debug)

Saída do dataset por batch:
- `X`: `(B,1,256,256)`
- `Y_heatmap`: `(B,64,256,256)`
- `Y_presence`: `(B,32)`

Regras:
- aplicar augmentações geométricas iguais em imagem e máscara
- aplicar intensidade/ruído apenas na imagem
- sem flip horizontal/vertical

## 2) Split reprodutível

Criar split fixo `train/val` com seed:
- sugestão inicial: 80/20
- salvar em arquivo (`splits.json`) para reuso

## 3) Script de treino mínimo (`train.py`)

Implementar:
- instanciamento do `HydraUNetMultiTask`
- `HydraMultiTaskLoss`
- optimizer/scheduler
- loops de treino e validação
- checkpoint:
  - `best.ckpt`
  - `last.ckpt`

## 4) Logging de métricas

Registrar por época em:
- TensorBoard
- CSV

Campos mínimos:
- `train_total_loss`
- `train_heatmap_loss`
- `train_presence_loss`
- `val_total_loss`
- `val_heatmap_loss`
- `val_presence_loss`
- `lr`

## 5) Capturas visuais no treino

Integrar callback:
- `capture_epoch_visuals(...)`

Salvar por época:
- painéis before/after augmentação
- painéis de atenção por camada
- `manifest.jsonl` + `index.html` viewer

## 6) Script de avaliação (`eval.py`)

Comparar predição vs ground truth e gerar:
- métricas globais
- métricas por dente
- métricas de presença
- painéis visuais de comparação

## 7) Run smoke test local

Rodar treino curto para validar integração:
- 2-3 épocas
- poucos batches
- confirmar:
  - sem erro de pipeline
  - checkpoints gerados
  - logs gravados
  - viewer visual funcionando

## 8) Preparação para EC2

Garantir portabilidade:
- comandos relativos à raiz do repo
- sem paths locais hardcoded
- documentação com passo-a-passo EC2

## 9) Pronto para escala

Após smoke local aprovado:
- aumentar epochs
- ajustar batch/LR
- iniciar treino com dataset completo na EC2
- acompanhar via HTML viewer + TensorBoard/W&B/MLflow
