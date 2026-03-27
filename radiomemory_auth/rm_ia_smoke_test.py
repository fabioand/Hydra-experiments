from typing import Dict, Tuple

import requests

from login import LINKAPI, LoginAPI


BASE_URL = LINKAPI.rstrip("/")

V1_ENDPOINTS = [
    "/v1/panoramics/dentition",
    "/v1/panoramics/longaxis",
    "/v1/panoramics/metals",
    "/v1/panoramics/panorogram",
    "/v1/panoramics/teeth_segmentation",
    "/v1/panoramics/procedures",
    "/v1/panoramics/anatomic_points",
    "/v1/panoramics/teeth_anomalies_heatmap",
    "/v1/panoramics/resto_radicular",
    "/v1/periapicals/classification",
    "/v1/periapicals/longaxis",
    "/v1/periapicals/teeth_segmentation",
    "/v1/periapicals/anomalies_all",
]

TOMOS_ENDPOINTS = [
    "/internal/tomos/SagitalClass",
    "/internal/tomos/SmallFOVSeg",
    "/internal/tomos/BigFOVSeg",
    "/internal/tomos/PontosCephMax",
    "/internal/tomos/PontosCephMan",
    "/internal/tomos/PontosCephBoca",
    "/internal/tomos/Mip8mmMax",
    "/internal/tomos/Mip8mmMan",
    "/internal/tomos/Mip8mmBoca",
    "/internal/tomos/PlanoOclusal",
    "/internal/tomos/LinhaOclusalAxial",
    "/internal/tomos/SagitalClassLegacy",
    "/internal/tomos/Teles",
    "/internal/tomos/CycleMax",
    "/internal/tomos/KeypointsAxial",
]


def auth_headers() -> Dict[str, str]:
    auth = LoginAPI()
    token_type = auth.get("token_type")
    access_token = auth.get("access_token")

    if not token_type or not access_token:
        raise RuntimeError(f"Authentication failed: {auth}")

    return {
        "Authorization": f"{token_type} {access_token}",
        "Content-type": "application/json",
        "Accept": "application/json",
    }


def check_get(path: str, headers: Dict[str, str] = None) -> Tuple[int, str]:
    url = f"{BASE_URL}{path}"
    response = requests.get(url, headers=headers, timeout=30)
    return response.status_code, url


def check_post(path: str, body: Dict, headers: Dict[str, str] = None) -> Tuple[int, str, str]:
    url = f"{BASE_URL}{path}"
    response = requests.post(url, headers=headers, json=body, timeout=45)
    preview = response.text[:200].replace("\n", " ")
    return response.status_code, url, preview


def main() -> None:
    print(f"BASE\t{BASE_URL}")

    auth = auth_headers()

    status, url = check_get("/health")
    print(f"{status}\tGET\t{url}")

    status, url = check_get("/v1/docs", headers=auth)
    print(f"{status}\tGET\t{url}")

    print("\n# V1 (expects auth + base64_image)")
    for path in V1_ENDPOINTS:
        status, url, preview = check_post(path, body={}, headers=auth)
        print(f"{status}\tPOST\t{url}\t{preview}")

    print("\n# TOMOS (expects base64_image)")
    for path in TOMOS_ENDPOINTS:
        status, url, preview = check_post(path, body={})
        print(f"{status}\tPOST\t{url}\t{preview}")

    print("\n# TOMOS legacy points endpoint")
    status, url, preview = check_post(
        "/internal/tomos/bbox-autoencoder-maxila",
        body={"points": [[1.0, 2.0], [3.0, 4.0]]},
    )
    print(f"{status}\tPOST\t{url}\t{preview}")


if __name__ == "__main__":
    main()
