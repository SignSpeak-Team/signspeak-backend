from asyncio import timeout
from wsgiref import headers

import httpx

async def forward_request(
        method: str,
        url: str,
        headers: dict | None = None,
        json: dict | None = None,
):
    timeout = httpx.Timeout(10.0)


    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method=method,
            url=url,
            headers=headers,
            json=json,
        )
    return response