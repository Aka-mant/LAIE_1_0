"""Границы сетевой доступности автономной установки LAIE."""

from __future__ import annotations

from ipaddress import ip_address


_LOCALHOST_NAMES = frozenset({"localhost"})


def require_loopback_host(value: str, setting_name: str) -> str:
    """Разрешает только localhost и IP-адреса loopback.

    Запрещает удалённые адреса, DNS-имена и wildcard-хосты для БД
    и списка HTTP-хостов автономной установки.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{setting_name}: значение хоста не должно быть пустым.")

    host = value.strip().lower()
    if host in _LOCALHOST_NAMES:
        return host

    # Скобки допустимы в записи IPv6 из конфигурации.
    candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host

    try:
        address = ip_address(candidate)
    except ValueError as exc:
        raise ValueError(
            f"{setting_name}: разрешены только localhost и loopback IP; "
            f"получено {value!r}."
        ) from exc

    if not address.is_loopback:
        raise ValueError(
            f"{setting_name}: удалённые адреса запрещены; "
            f"получено {value!r}."
        )

    return candidate


def parse_local_allowed_hosts(value: str) -> list[str]:
    """Разбирает ALLOWED_HOSTS и запрещает внешние имена и wildcard."""
    hosts = [item.strip() for item in value.split(",") if item.strip()]
    if not hosts:
        raise ValueError("DJANGO_ALLOWED_HOSTS: список локальных хостов пуст.")

    return [
        require_loopback_host(host, "DJANGO_ALLOWED_HOSTS")
        for host in hosts
    ]
