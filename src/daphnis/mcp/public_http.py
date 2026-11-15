"""HTTPS transport for Registry selections, pinned to a vetted public address."""

import asyncio
import socket
from ipaddress import IPv6Address, ip_address, ip_network

import httpx2

_TRANSLATION = (ip_network("64:ff9b::/96"), ip_network("64:ff9b:1::/48"))


def _public_ip(value: str) -> bool:
    address = ip_address(value)
    if isinstance(address, IPv6Address) and (
        address.ipv4_mapped or address.sixtofour or address.teredo
        or any(address in network for network in _TRANSLATION)
    ):
        return False
    return address.is_global and not (address.is_multicast or address.is_reserved) and "%" not in value


def public_url(url: httpx2.URL) -> None:
    host = url.host.rstrip(".").lower()
    if (
        url.scheme != "https" or url.port not in (None, 443) or url.userinfo
        or "#" in str(url) or "?" in str(url) or not host
    ):
        raise ValueError("在线发现只允许无凭据、查询参数或片段的 HTTPS 默认端口地址。")
    try:
        ip_address(host)
    except ValueError:
        if "." not in host or host.endswith((".localhost", ".local", ".internal", ".home.arpa")):
            raise ValueError("在线发现不允许本机或内网地址。") from None
    else:
        if not _public_ip(host):
            raise ValueError("在线发现不允许本机或内网地址。")


class PublicHTTPTransport(httpx2.AsyncHTTPTransport):
    def __init__(self, address: str):
        super().__init__(trust_env=False)
        self.address = address

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        # Preserve the logical Host and verify TLS against the original hostname.
        pinned = httpx2.Request(
            request.method, request.url.copy_with(host=self.address),
            headers=request.headers, stream=request.stream,
            extensions={**request.extensions, "sni_hostname": request.url.raw_host.decode("ascii")},
        )
        return await super().handle_async_request(pinned)


async def public_transport(url: httpx2.URL) -> PublicHTTPTransport:
    public_url(url)
    try:
        async with asyncio.timeout(5):
            records = await asyncio.get_running_loop().getaddrinfo(
                url.raw_host.decode("ascii"), 443, type=socket.SOCK_STREAM,
            )
    except (OSError, TimeoutError):
        raise ValueError("无法解析 MCP 服务地址。") from None
    addresses = [record[4][0] for record in records]
    if not addresses or not all(map(_public_ip, addresses)):
        raise ValueError("MCP 服务地址解析到非公网地址，已阻止连接。")
    return PublicHTTPTransport(next((address for address in addresses if ":" not in address), addresses[0]))
