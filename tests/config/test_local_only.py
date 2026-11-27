import pytest

from config.local_only import parse_local_allowed_hosts, require_loopback_host


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "[::1]"])
def test_loopback_hosts_are_allowed(host):
    assert require_loopback_host(host, "DB_HOST") in {
        "localhost", "127.0.0.1", "::1"
    }


@pytest.mark.parametrize(
    "host",
    [
        "db.example.com",
        "192.168.1.10",
        "10.0.0.5",
        "0.0.0.0",
        "*",
        "",
    ],
)
def test_non_loopback_hosts_are_rejected(host):
    with pytest.raises(ValueError):
        require_loopback_host(host, "DB_HOST")


def test_allowed_hosts_are_parsed_and_normalized():
    assert parse_local_allowed_hosts("127.0.0.1, localhost, ::1") == [
        "127.0.0.1", "localhost", "::1"
    ]


@pytest.mark.parametrize("value", ["*", "example.com", "127.0.0.1,10.0.0.2"])
def test_remote_allowed_hosts_are_rejected(value):
    with pytest.raises(ValueError):
        parse_local_allowed_hosts(value)


def test_empty_allowed_hosts_are_rejected():
    with pytest.raises(ValueError):
        parse_local_allowed_hosts(" , ")
