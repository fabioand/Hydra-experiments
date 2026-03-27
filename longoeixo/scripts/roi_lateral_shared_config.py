#!/usr/bin/env python3
"""Config oficial para estrategia Multi-ROI fixa com rede lateral compartilhada.

Resumo:
- 3 janelas fixas por geometria da imagem (sem dependencia de pontos anatomicos):
  LEFT, CENTER, RIGHT
- Rede CENTER (24 canais): incisivos+caninos (12 dentes x 2 pontos)
- Rede LATERAL (20 canais): pre-molares+molares do lado DIREITO (10 dentes x 2 pontos)
- No treino da rede lateral:
  - crop direito entra direto;
  - crop esquerdo entra flipado horizontalmente e com remapeamento de labels
    esquerda->direita.
- Na inferencia:
  - direito: direto;
  - esquerdo: flip input, inferir, desfazer flip das coordenadas e remapear labels
    direita->esquerda.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# Dentes canonicamente modelados na rede center (24 canais = 12 dentes x 2 pontos)
CENTER_TEETH: List[str] = [
    "11",
    "12",
    "13",
    "21",
    "22",
    "23",
    "31",
    "32",
    "33",
    "41",
    "42",
    "43",
]

# Mapeamento de simetria horizontal para dentes centrais (flip do ROI CENTER).
# Direita -> Esquerda no arco superior, e Direita -> Esquerda no arco inferior.
CENTER_RIGHT_TO_LEFT: Dict[str, str] = {
    "11": "21",
    "12": "22",
    "13": "23",
    "21": "11",
    "22": "12",
    "23": "13",
    "41": "31",
    "42": "32",
    "43": "33",
    "31": "41",
    "32": "42",
    "33": "43",
}

CENTER_LEFT_TO_RIGHT: Dict[str, str] = {v: k for k, v in CENTER_RIGHT_TO_LEFT.items()}


# Dentes canonicamente modelados na rede lateral (lado direito anatomico)
LATERAL_RIGHT_TEETH: List[str] = [
    "14",
    "15",
    "16",
    "17",
    "18",
    "44",
    "45",
    "46",
    "47",
    "48",
]


# Dentes equivalentes do lado esquerdo anatomico
LATERAL_LEFT_TEETH: List[str] = [
    "24",
    "25",
    "26",
    "27",
    "28",
    "34",
    "35",
    "36",
    "37",
    "38",
]


# Mapeamento no flip horizontal para treinar lateral unica no espaco "direito".
# Esquerda -> Direita
LEFT_TO_RIGHT: Dict[str, str] = {
    "24": "14",
    "25": "15",
    "26": "16",
    "27": "17",
    "28": "18",
    "34": "44",
    "35": "45",
    "36": "46",
    "37": "47",
    "38": "48",
}

# Direita -> Esquerda (para desfazer na inferencia do lado esquerdo)
RIGHT_TO_LEFT: Dict[str, str] = {v: k for k, v in LEFT_TO_RIGHT.items()}


def center_channels_24() -> List[Tuple[str, int, str]]:
    """Retorna ordem canonica dos 24 canais: p1,p2 por dente center."""
    ch = []
    for tooth in CENTER_TEETH:
        ch.append((tooth, 0, f"{tooth}_p1"))
        ch.append((tooth, 1, f"{tooth}_p2"))
    return ch


def lateral_channels_20_right_space() -> List[Tuple[str, int, str]]:
    """Retorna ordem canonica dos 20 canais da rede lateral no espaco 'direito'."""
    ch = []
    for tooth in LATERAL_RIGHT_TEETH:
        ch.append((tooth, 0, f"{tooth}_p1"))
        ch.append((tooth, 1, f"{tooth}_p2"))
    return ch


def fixed_three_windows(width: int, height: int) -> Dict[str, List[int]]:
    """3 janelas fixas independentes de pontos anatomicos.

    Formato de retorno: {name: [x1, y1, x2, y2]} com x2/y2 exclusivos.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be > 0")

    half = width // 2
    left = [0, 0, half, height]
    right = [width - half, 0, width, height]
    center_x1 = (width - half) // 2
    center = [center_x1, 0, center_x1 + half, height]
    return {"LEFT": left, "CENTER": center, "RIGHT": right}


def rect_width(rect_xyxy: List[int]) -> int:
    x1, _, x2, _ = rect_xyxy
    w = x2 - x1
    if w <= 0:
        raise ValueError(f"invalid rect width: {rect_xyxy}")
    return w


def rect_height(rect_xyxy: List[int]) -> int:
    _, y1, _, y2 = rect_xyxy
    h = y2 - y1
    if h <= 0:
        raise ValueError(f"invalid rect height: {rect_xyxy}")
    return h


def global_to_local(pt_global_xy: Tuple[float, float], rect_xyxy: List[int]) -> Tuple[float, float]:
    """Converte ponto global -> local do crop."""
    x, y = pt_global_xy
    x1, y1, _, _ = rect_xyxy
    return x - x1, y - y1


def local_to_global(pt_local_xy: Tuple[float, float], rect_xyxy: List[int]) -> Tuple[float, float]:
    """Converte ponto local do crop -> global."""
    x, y = pt_local_xy
    x1, y1, _, _ = rect_xyxy
    return x + x1, y + y1


def flip_x(x: float, width: int) -> float:
    """Flip horizontal no espaco de pixel absoluto."""
    if width <= 0:
        raise ValueError("width must be > 0")
    return (width - 1) - x


def remap_left_to_right(tooth: str) -> str:
    if tooth not in LEFT_TO_RIGHT:
        raise KeyError(f"Tooth {tooth} not in LEFT_TO_RIGHT mapping")
    return LEFT_TO_RIGHT[tooth]


def remap_right_to_left(tooth: str) -> str:
    if tooth not in RIGHT_TO_LEFT:
        raise KeyError(f"Tooth {tooth} not in RIGHT_TO_LEFT mapping")
    return RIGHT_TO_LEFT[tooth]


def remap_center_flip(tooth: str) -> str:
    if tooth not in CENTER_RIGHT_TO_LEFT:
        raise KeyError(f"Tooth {tooth} not in CENTER_RIGHT_TO_LEFT mapping")
    return CENTER_RIGHT_TO_LEFT[tooth]


def center_prepare_train(
    tooth_center: str,
    pt_global_xy: Tuple[float, float],
    image_width: int,
    image_height: int,
    flip_horizontal: bool = False,
) -> Tuple[str, Tuple[float, float]]:
    """Prepara label+coordenada no ROI CENTER para treino.

    Pipeline (sem flip):
      global -> local(CENTER)
    Pipeline (com flip):
      global -> local(CENTER) -> flip_x(local) -> remap label simetrico
    """
    if tooth_center not in CENTER_TEETH:
        raise KeyError(f"tooth {tooth_center} not in CENTER_TEETH")
    rect_center = fixed_three_windows(image_width, image_height)["CENTER"]
    x_local, y_local = global_to_local(pt_global_xy, rect_center)
    if not flip_horizontal:
        return tooth_center, (x_local, y_local)
    x_local_flip = flip_x(x_local, rect_width(rect_center))
    return remap_center_flip(tooth_center), (x_local_flip, y_local)


def center_restore_inference(
    tooth_center_pred: str,
    pred_local_xy: Tuple[float, float],
    image_width: int,
    image_height: int,
    came_from_flipped_input: bool = False,
) -> Tuple[str, Tuple[float, float]]:
    """Restaura predição do ROI CENTER para coordenada global.

    Entrada esperada:
    - pred_local_xy no espaço local do ROI CENTER.
    - se came_from_flipped_input=True, pred_local_xy está no espaço FLIPADO.
      Então desfaz flip em X e remapeia label simétrica.
    """
    if tooth_center_pred not in CENTER_TEETH:
        raise KeyError(f"tooth {tooth_center_pred} not in CENTER_TEETH")

    rect_center = fixed_three_windows(image_width, image_height)["CENTER"]
    x, y = pred_local_xy
    tooth_out = tooth_center_pred
    if came_from_flipped_input:
        x = flip_x(x, rect_width(rect_center))
        tooth_out = remap_center_flip(tooth_center_pred)
    return tooth_out, local_to_global((x, y), rect_center)


def lateral_prepare_right_train(
    tooth_right: str,
    pt_global_xy: Tuple[float, float],
    image_width: int,
    image_height: int,
) -> Tuple[str, Tuple[float, float]]:
    """Treino lateral canônico (direito anatômico) com crop LEFT sem flip.

    Retorna:
    - label canônica direita
    - coordenada local no crop LEFT (sem flip)
    """
    if tooth_right not in LATERAL_RIGHT_TEETH:
        raise KeyError(f"tooth {tooth_right} not in LATERAL_RIGHT_TEETH")
    # No dataset atual, direito anatômico aparece majoritariamente no lado LEFT.
    rect = fixed_three_windows(image_width, image_height)["LEFT"]
    return tooth_right, global_to_local(pt_global_xy, rect)


def lateral_prepare_left_train(
    tooth_left: str,
    pt_global_xy: Tuple[float, float],
    image_width: int,
    image_height: int,
) -> Tuple[str, Tuple[float, float]]:
    """Treino lateral espelhado: crop RIGHT flipado para espaço canônico direito.

    Pipeline aplicado:
    1) recorta RIGHT
    2) converte global->local(RIGHT)
    3) flipa X no espaço local do crop
    4) remapeia label esquerda->direita
    """
    tooth_right = remap_left_to_right(tooth_left)
    rect_right = fixed_three_windows(image_width, image_height)["RIGHT"]
    x_local, y_local = global_to_local(pt_global_xy, rect_right)
    x_local_flipped = flip_x(x_local, rect_width(rect_right))
    return tooth_right, (x_local_flipped, y_local)


def lateral_restore_right_inference(
    tooth_right: str,
    pred_local_xy: Tuple[float, float],
    image_width: int,
    image_height: int,
) -> Tuple[str, Tuple[float, float]]:
    """Inferência lateral do ramo canônico: local LEFT -> global."""
    if tooth_right not in LATERAL_RIGHT_TEETH:
        raise KeyError(f"tooth {tooth_right} not in LATERAL_RIGHT_TEETH")
    rect_left = fixed_three_windows(image_width, image_height)["LEFT"]
    return tooth_right, local_to_global(pred_local_xy, rect_left)


def lateral_restore_left_inference(
    tooth_right: str,
    pred_local_xy_flipped: Tuple[float, float],
    image_width: int,
    image_height: int,
) -> Tuple[str, Tuple[float, float]]:
    """Inferência lateral do lado esquerdo anatômico via input do RIGHT flipado.

    Entrada esperada:
    - predição local no crop RIGHT já flipado (espaço canônico direito)
    Saída:
    - label esquerda
    - coordenada global na imagem original
    """
    tooth_left = remap_right_to_left(tooth_right)
    rect_right = fixed_three_windows(image_width, image_height)["RIGHT"]
    w_right = rect_width(rect_right)
    x_flip, y = pred_local_xy_flipped
    x_unflip = flip_x(x_flip, w_right)
    pt_global = local_to_global((x_unflip, y), rect_right)
    return tooth_left, pt_global


def _self_check() -> None:
    # 1) cardinalidades esperadas
    assert len(CENTER_TEETH) == 12
    assert len(LATERAL_RIGHT_TEETH) == 10
    assert len(LATERAL_LEFT_TEETH) == 10
    assert len(center_channels_24()) == 24
    assert len(lateral_channels_20_right_space()) == 20

    # 2) disjuncao center vs lateral
    assert set(CENTER_TEETH).isdisjoint(set(LATERAL_RIGHT_TEETH))
    assert set(CENTER_TEETH).isdisjoint(set(LATERAL_LEFT_TEETH))

    # 3) mapeamento bijetivo esquerda<->direita
    assert set(LEFT_TO_RIGHT.keys()) == set(LATERAL_LEFT_TEETH)
    assert set(LEFT_TO_RIGHT.values()) == set(LATERAL_RIGHT_TEETH)
    for left_tooth, right_tooth in LEFT_TO_RIGHT.items():
        assert RIGHT_TO_LEFT[right_tooth] == left_tooth

    # 3b) mapeamento simetrico central bijetivo e involutivo
    assert set(CENTER_RIGHT_TO_LEFT.keys()) == set(CENTER_TEETH)
    assert set(CENTER_RIGHT_TO_LEFT.values()) == set(CENTER_TEETH)
    assert set(CENTER_LEFT_TO_RIGHT.keys()) == set(CENTER_TEETH)
    assert set(CENTER_LEFT_TO_RIGHT.values()) == set(CENTER_TEETH)
    for t in CENTER_TEETH:
        t2 = CENTER_RIGHT_TO_LEFT[t]
        assert CENTER_RIGHT_TO_LEFT[t2] == t
        assert CENTER_LEFT_TO_RIGHT[t2] == t

    # 4) janelas fixas cobrindo toda altura e largura de meia imagem
    rects = fixed_three_windows(2776, 1480)
    assert rects["LEFT"] == [0, 0, 1388, 1480]
    assert rects["RIGHT"] == [1388, 0, 2776, 1480]
    assert rects["CENTER"] == [694, 0, 2082, 1480]

    # 5) flip involutivo
    w = 2776
    for x in [0, 1, 100.5, 1388, 2775]:
        x2 = flip_x(flip_x(x, w), w)
        assert abs(x2 - x) < 1e-6

    # 6) round-trip lado esquerdo (prepare treino -> restore inferencia)
    W, H = 2776, 1480
    pt_global_left = (2000.25, 700.5)  # deve cair no crop RIGHT
    tooth_left = "24"
    tooth_right, pt_local_flip = lateral_prepare_left_train(tooth_left, pt_global_left, W, H)
    restored_left_tooth, restored_global = lateral_restore_left_inference(
        tooth_right, pt_local_flip, W, H
    )
    assert restored_left_tooth == tooth_left
    assert abs(restored_global[0] - pt_global_left[0]) < 1e-6
    assert abs(restored_global[1] - pt_global_left[1]) < 1e-6

    # 7) round-trip lado direito canônico (prepare treino -> restore inferencia)
    pt_global_right = (1000.75, 650.25)  # deve cair no crop LEFT
    tooth_right = "14"
    t_right, pt_local = lateral_prepare_right_train(tooth_right, pt_global_right, W, H)
    restored_t_right, restored_global_right = lateral_restore_right_inference(t_right, pt_local, W, H)
    assert restored_t_right == tooth_right
    assert abs(restored_global_right[0] - pt_global_right[0]) < 1e-6
    assert abs(restored_global_right[1] - pt_global_right[1]) < 1e-6

    # 8) round-trip center sem flip
    pt_center = (1500.5, 700.25)
    t_center = "12"
    t_prep, p_local = center_prepare_train(t_center, pt_center, W, H, flip_horizontal=False)
    assert t_prep == t_center
    t_rest, p_global = center_restore_inference(t_prep, p_local, W, H, came_from_flipped_input=False)
    assert t_rest == t_center
    assert abs(p_global[0] - pt_center[0]) < 1e-6
    assert abs(p_global[1] - pt_center[1]) < 1e-6

    # 9) round-trip center com flip + remap
    t_center2 = "11"
    pt_center2 = (1200.75, 555.5)
    t_flip, p_local_flip = center_prepare_train(t_center2, pt_center2, W, H, flip_horizontal=True)
    assert t_flip == "21"
    t_back, p_global_back = center_restore_inference(
        t_flip,
        p_local_flip,
        W,
        H,
        came_from_flipped_input=True,
    )
    assert t_back == t_center2
    assert abs(p_global_back[0] - pt_center2[0]) < 1e-6
    assert abs(p_global_back[1] - pt_center2[1]) < 1e-6


if __name__ == "__main__":
    _self_check()
    print("OK: roi_lateral_shared_config self-check passed")
    print("CENTER channels:", len(center_channels_24()))
    print("LATERAL channels:", len(lateral_channels_20_right_space()))
    print("LEFT_TO_RIGHT:", LEFT_TO_RIGHT)
