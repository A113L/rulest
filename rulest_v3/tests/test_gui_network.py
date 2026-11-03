from pathlib import Path
import ipaddress
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


def test_lan_bind_provides_local_and_lan_urls():
    urls = app.build_access_urls("0.0.0.0", 8765, ["192.168.1.20", "10.0.0.8"])
    assert urls == [
        "http://127.0.0.1:8765",
        "http://192.168.1.20:8765",
        "http://10.0.0.8:8765",
    ]


def test_specific_host_only_displays_that_host():
    assert app.build_access_urls("127.0.0.1", 9000, ["192.168.1.20"]) == [
        "http://127.0.0.1:9000"
    ]


def test_lan_address_discovery_filters_loopback_and_link_local():
    # The address discovery routine should never advertise loopback or link-local addresses.
    addresses = app.discover_lan_ipv4_addresses()
    for value in addresses:
        ip = ipaddress.ip_address(value)
        assert ip.version == 4
        assert ip.is_private
        assert not ip.is_loopback
        assert not ip.is_link_local
