# AE Reconstruction Web App (Vue + FastAPI)

Projeto web para reproduzir o comportamento do script:
- `ae_radiograph_filter/scripts/show_single_reconstruction_cv2.py`

Paridade implementada:
1. Backend lê imagem em grayscale.
2. Faz downsample para `input_size` (default 256).
3. Roda `PanoramicResNetAutoencoder(resnet34)`.
4. Faz upsample da reconstrução para tamanho original com `cv2.INTER_LANCZOS4`.
5. Frontend aplica os sliders no cliente com a mesma fórmula do script:

```text
enhanced = clip(original - fs*recon + fa*(original - recon), 0, 1)
```

## Estrutura

- `backend/app/main.py`: API FastAPI.
- `backend/app/model_service.py`: inferência e utilitários de imagem.
- `frontend/`: app Vue 3 + Vite.
- `sample_images/`: pasta de exemplos no servidor.
- `scripts/run_backend.sh`: sobe backend local.
- `scripts/run_frontend.sh`: sobe frontend local.
- `Dockerfile.backend`: imagem backend para Linux/EC2.
- `Dockerfile.frontend`: imagem frontend estática (Nginx).
- `docker-compose.yml`: stack local containerizada.
- `deploy/ae-recon-backend.service`: template de serviço systemd.
- `deploy/nginx-ae-recon.conf`: exemplo de configuração Nginx.

## Endpoints

- `GET /healthz`
  - status da API, device e caminhos de configuração.
- `GET /v1/config`
  - defaults (`input_size`, limites de upload, sliders).
- `GET /v1/samples`
  - lista arquivos da pasta de samples do servidor.
- `POST /v1/reconstruct`
  - upload `multipart/form-data` com `file` e `input_size` opcional.
- `POST /v1/reconstruct/sample`
  - reconstrói um arquivo da pasta de samples (JSON com `sample_name`).

## Configuração via ambiente

Copie:

```bash
cp ae_recon_webapp/.env.example ae_recon_webapp/.env
```

Principais variáveis:
- `AE_RECON_CKPT`: checkpoint do autoencoder.
- `AE_RECON_SAMPLE_IMAGES_DIR`: pasta de imagens de exemplo no servidor.
- `AE_RECON_ALLOWED_ORIGINS`: CORS (CSV).
- `VITE_API_BASE`: URL da API usada no frontend.

## Rodar local (sem Docker)

Pré-requisitos:
- Python 3.9+
- Node 20+
- ambiente virtual do projeto em `/Users/fabioandrade/hydra/ae_recon_webapp/.venv`

Terminal 1 (backend):

```bash
cd /Users/fabioandrade/hydra
python3 -m venv ae_recon_webapp/.venv
./ae_recon_webapp/.venv/bin/python -m pip install -r ae_recon_webapp/backend/requirements.txt
bash ae_recon_webapp/scripts/run_backend.sh
```

Terminal 2 (frontend):

```bash
cd /Users/fabioandrade/hydra
bash ae_recon_webapp/scripts/run_frontend.sh
```

Acesse:
- UI: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`

## Rodar com Docker

Na raiz do workspace:

```bash
cd /Users/fabioandrade/hydra/ae_recon_webapp
docker compose up --build
```

Acesse:
- frontend: `http://localhost:8080`
- backend: `http://localhost:8000`

## Suporte a imagens de exemplo no servidor

Coloque arquivos em:
- `ae_recon_webapp/sample_images/`

A UI consulta `GET /v1/samples` e permite rodar sem upload local.

## Deploy na EC2 Linux

Fluxo recomendado:
1. Copiar o repositório para EC2 (ou fazer pull).
2. Garantir Python/Node (ou usar Docker).
3. Ajustar `ae_recon_webapp/.env` com caminhos Linux.
4. Subir backend e frontend.

Exemplo rápido (sem Docker):

```bash
cd /opt/hydra
cp ae_recon_webapp/.env.example ae_recon_webapp/.env
# editar .env para caminhos Linux
python3 -m pip install -r ae_recon_webapp/backend/requirements.txt
bash ae_recon_webapp/scripts/run_backend.sh
# em outro terminal
bash ae_recon_webapp/scripts/run_frontend.sh
```

Para produção:
- backend: `uvicorn` com `--workers` e processo supervisionado (systemd/supervisor).
- frontend: `npm run build` + Nginx estático.
- reverse proxy único (Nginx) para `/` (frontend) e `/api` (backend), se desejado.
- templates de deploy em `ae_recon_webapp/deploy/`.
- se usar proxy `/api`, buildar frontend com `VITE_API_BASE=/api`.

## Observações de paridade

- O downsample e o upsample de inferência foram mantidos no backend para preservar equivalência com o script `cv2`.
- O cliente só realiza as operações dinâmicas dos sliders para resposta em tempo real.
