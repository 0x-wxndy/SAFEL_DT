"""Datasets and federated partitioning (IoT-focused: N-BaIoT, Edge-IIoTset)."""

from safel_dt.data.datasets import (
    SUPPORTED_DATASETS,
    EdgeIIoTsetDataset,
    NBaIoTDataset,
    TonIotDataset,
    load_dataset,
    load_edge_iiotset,
    load_nbaiot_per_device,
    load_toniot,
)
from safel_dt.data.partition import (
    assign_clients_to_fogs,
    dirichlet_partition,
    iid_partition,
    natural_per_device_partition,
)
from safel_dt.data.tabular import SyntheticTabularDataset

__all__ = [
    "SUPPORTED_DATASETS",
    "EdgeIIoTsetDataset",
    "NBaIoTDataset",
    "SyntheticTabularDataset",
    "TonIotDataset",
    "assign_clients_to_fogs",
    "dirichlet_partition",
    "iid_partition",
    "load_dataset",
    "load_edge_iiotset",
    "load_nbaiot_per_device",
    "load_toniot",
    "natural_per_device_partition",
]
