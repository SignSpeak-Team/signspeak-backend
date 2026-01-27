from typing import Any

import httpx


async def forward_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
):
    timeout = httpx.Timeout(10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(method, url, json=json, headers=headers)
        return response
