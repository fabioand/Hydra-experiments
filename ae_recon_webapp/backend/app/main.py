from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_service import ReconstructionService, discover_sample_images
from .settings import load_settings


class ReconstructResponse(BaseModel):
    width: int
    height: int
    original_png_base64: str
    reconstruction_png_base64: str
    defaults: dict[str, float]
    meta: dict[str, str]


class SampleItem(BaseModel):
    name: str
    path: str


class SampleReconstructRequest(BaseModel):
    sample_name: str
    input_size: Optional[int] = None


settings = load_settings()
service = ReconstructionService(ckpt_path=settings.ckpt_path)

app = FastAPI(title="AE Reconstruction Web API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_response(orig_u8, recon_u8, input_size: int, source: str) -> ReconstructResponse:
    return ReconstructResponse(
        width=int(orig_u8.shape[1]),
        height=int(orig_u8.shape[0]),
        original_png_base64=service.png_base64_from_u8(orig_u8),
        reconstruction_png_base64=service.png_base64_from_u8(recon_u8),
        defaults={"fs": 0.0, "fa": 0.0},
        meta={
            "model": "PanoramicResNetAutoencoder(resnet34)",
            "ckpt": str(settings.ckpt_path),
            "input_size": str(input_size),
            "device": str(service.device),
            "source": source,
        },
    )


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "device": str(service.device),
        "ckpt": str(settings.ckpt_path),
        "sample_images_dir": str(settings.sample_images_dir),
    }


@app.get("/v1/config")
def config() -> dict:
    return {
        "default_input_size": settings.default_input_size,
        "max_upload_bytes": settings.max_upload_bytes,
        "defaults": {"fs": 0.0, "fa": 0.0},
        "allowed_origins": settings.allowed_origins,
    }


@app.get("/v1/samples", response_model=list[SampleItem])
def list_samples() -> list[SampleItem]:
    files = discover_sample_images(settings.sample_images_dir)
    items = [SampleItem(name=path.name, path=str(path)) for path in files]
    return items


@app.post("/v1/reconstruct", response_model=ReconstructResponse)
async def reconstruct_upload(
    file: UploadFile = File(...),
    input_size: int = Form(default=settings.default_input_size),
) -> ReconstructResponse:
    if input_size <= 0 or input_size > 2048:
        raise HTTPException(status_code=422, detail="input_size deve estar entre 1 e 2048")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"Arquivo excede limite de {settings.max_upload_bytes} bytes")

    try:
        orig_u8 = service.decode_gray(raw)
        orig_u8, recon_u8 = service.reconstruct(orig_u8=orig_u8, input_size=input_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Falha na reconstrucao: {exc}") from exc

    return _build_response(orig_u8=orig_u8, recon_u8=recon_u8, input_size=input_size, source=file.filename or "upload")


@app.post("/v1/reconstruct/sample", response_model=ReconstructResponse)
def reconstruct_sample(payload: SampleReconstructRequest) -> ReconstructResponse:
    input_size = payload.input_size or settings.default_input_size
    if input_size <= 0 or input_size > 2048:
        raise HTTPException(status_code=422, detail="input_size deve estar entre 1 e 2048")

    sample_path = (settings.sample_images_dir / payload.sample_name).resolve()
    sample_root = settings.sample_images_dir.resolve()

    if sample_root not in sample_path.parents:
        raise HTTPException(status_code=400, detail="sample_name invalido")
    if not sample_path.exists() or not sample_path.is_file():
        raise HTTPException(status_code=404, detail=f"Sample nao encontrado: {payload.sample_name}")

    try:
        raw = sample_path.read_bytes()
        orig_u8 = service.decode_gray(raw)
        orig_u8, recon_u8 = service.reconstruct(orig_u8=orig_u8, input_size=input_size)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Falha na reconstrucao do sample: {exc}") from exc

    return _build_response(orig_u8=orig_u8, recon_u8=recon_u8, input_size=input_size, source=sample_path.name)
