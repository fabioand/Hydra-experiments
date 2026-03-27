import os

import requests

# LINKAPI='https://f3thzslz6i.execute-api.us-east-1.amazonaws.com'

TIPO = os.getenv("RM_TIPO", "prod")
# TIPO='dev'
if TIPO == 'dev':
    LINKAPI = os.getenv("RM_BASE_URL", 'https://iaapi.radiomemory.com.br/dev')
else:
    LINKAPI = os.getenv("RM_BASE_URL", 'https://api.radiomemory.com.br/ia-idoc')
    # LINKAPI='https://iaapi-dev.radiomemory.com.br'

USERNAME = os.getenv("RM_USERNAME", "test")
PASSWORD = os.getenv("RM_PASSWORD", "A)mks8aNKjanm9")
TIMEOUT = int(os.getenv("RM_TIMEOUT", "30"))


def LoginAPI():

    addr = LINKAPI + '/v1/auth/token'
    headers = {"Content-type": 'application/x-www-form-urlencoded', 'accept': 'application/json'}
    body = (
        f'grant_type=&username={USERNAME}&password={PASSWORD}'
        '&scope=&client_id=&client_secret='
    )

    r = requests.post(addr, headers=headers, data=body, timeout=TIMEOUT)
    return r.json()


def LoginAPIDEV():

    addr = os.getenv("RM_DEV2_BASE_URL", 'https://iaapi2.radiomemory.com.br/dev') + '/v1/auth/token'
    headers = {"Content-type": 'application/x-www-form-urlencoded', 'accept': 'application/json'}
    body = (
        f'grant_type=&username={USERNAME}&password={PASSWORD}'
        '&scope=&client_id=&client_secret='
    )

    r = requests.post(addr, headers=headers, data=body, timeout=TIMEOUT)
    return r.json()


if __name__ == "__main__":
    try:
        auth = LoginAPI()
        print(
            {
                "base": LINKAPI,
                "token_type": auth.get("token_type"),
                "has_access_token": bool(auth.get("access_token")),
            }
        )
    except requests.RequestException as exc:
        print({"base": LINKAPI, "error": str(exc)})
