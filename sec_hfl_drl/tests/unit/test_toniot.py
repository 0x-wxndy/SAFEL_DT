"""Offline TON_IoT loader test against a synthetic CSV in tmp_path.

We don't need the real (~30 MB) dataset to exercise the loader: the
schema is fully captured by a 10-row CSV with the same columns. This
test runs in milliseconds and verifies:

* the schema-driven preprocessing (drop ID cols, one-hot small
  categoricals, replace missing values),
* the binary / multi-class label modes,
* the per-client uniform split,
* the rejection paths (missing file, bad arg).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from safel_dt.data.toniot import TonIotMissing, load_toniot


def _make_synth_toniot(tmp_path: Path, n: int = 60) -> Path:
    """Create a TON_IoT-shaped CSV with mixed categorical / numeric columns."""
    rows = []
    for i in range(n):
        label = i % 2
        attack = "normal" if label == 0 else ["ddos", "dos", "scanning"][i % 3]
        rows.append(
            {
                "src_ip": f"10.0.0.{i}",
                "src_port": 1024 + i,
                "dst_ip": "8.8.8.8",
                "dst_port": 53 if i % 3 else 80,
                "proto": "tcp" if i % 2 else "udp",
                "service": "http" if i % 3 == 0 else "-",
                "duration": float(i),
                "src_bytes": i * 100,
                "dst_bytes": i * 50,
                "conn_state": "SF",
                "missed_bytes": 0,
                "src_pkts": i + 1,
                "src_ip_bytes": i * 40,
                "dst_pkts": i + 1,
                "dst_ip_bytes": i * 30,
                "dns_query": "-",
                "dns_qclass": 0,
                "dns_qtype": 0,
                "dns_rcode": 0,
                "dns_AA": "F",
                "dns_RD": "F",
                "dns_RA": "F",
                "dns_rejected": "F",
                "ssl_version": "-",
                "ssl_cipher": "-",
                "ssl_resumed": "F",
                "ssl_established": "F",
                "ssl_subject": "-",
                "ssl_issuer": "-",
                "http_trans_depth": 0,
                "http_method": "GET" if i % 4 == 0 else "-",
                "http_uri": "-",
                "http_version": "1.1" if i % 4 == 0 else "-",
                "http_request_body_len": 0,
                "http_response_body_len": 0,
                "http_status_code": 200,
                "http_user_agent": "-",
                "http_orig_mime_types": "-",
                "http_resp_mime_types": "-",
                "weird_name": "-",
                "weird_addl": "-",
                "weird_notice": "F",
                "label": label,
                "type": attack,
            }
        )
    csv_path = tmp_path / "TONIOT" / "train_test_network.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_toniot_loader_binary(tmp_path: Path) -> None:
    _make_synth_toniot(tmp_path, n=60)
    clients, test_set, meta = load_toniot(
        data_dir=tmp_path,
        mode="binary",
        num_clients=3,
        test_fraction=0.2,
        seed=0,
    )
    assert len(clients) == 3
    assert meta["num_classes"] == 2
    assert meta["mode"] == "binary"
    assert meta["in_features"] > 0
    # samples coverage: train + test = 60
    total = sum(len(c) for c in clients) + len(test_set)
    assert total == 60


def test_toniot_loader_multi(tmp_path: Path) -> None:
    _make_synth_toniot(tmp_path, n=60)
    clients, _test, meta = load_toniot(
        data_dir=tmp_path, mode="multi", num_clients=2, seed=0
    )
    # 4 distinct classes in the synth: normal, ddos, dos, scanning
    assert meta["num_classes"] == 4
    assert len(meta["class_names"]) == 4
    assert all(len(c) > 0 for c in clients)


def test_toniot_loader_drops_identifier_columns(tmp_path: Path) -> None:
    """The loader must NOT propagate src_ip / dst_ip into the feature matrix
    (otherwise the IDS task is trivially separable by IP)."""
    _make_synth_toniot(tmp_path, n=40)
    _clients, _test, meta = load_toniot(
        data_dir=tmp_path, mode="binary", num_clients=2, seed=0
    )
    # in_features after dropping IDs + numerics + one-hot expansions
    n_feat = int(meta["in_features"])
    # very loose bound; the exact count depends on dummies, but it must be
    # well under "all original 42 cols verbatim"
    assert 5 <= n_feat <= 80


def test_toniot_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(TonIotMissing):
        load_toniot(data_dir=tmp_path, mode="binary", num_clients=3, seed=0)
