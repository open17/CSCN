from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import networkx as nx
import numpy as np


RESULTS_FILENAME = "results.npz"
MANIFEST_FILENAME = "run_manifest.json"
DEBUG_STATE_FILENAME = "debug_state.pkl"


@dataclass(frozen=True)
class CompactDagBatch:
    cell_ids: np.ndarray
    node_count: int
    cell_edge_offsets: np.ndarray
    node_ptrs: np.ndarray
    targets: np.ndarray
    topo_orders: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.cell_ids.shape[0])


@dataclass(frozen=True)
class ResultPaths:
    result_dir: Path
    results_path: Path
    manifest_path: Path
    debug_state_path: Path


def build_result_paths(result_dir: Path) -> ResultPaths:
    return ResultPaths(
        result_dir=result_dir,
        results_path=result_dir / RESULTS_FILENAME,
        manifest_path=result_dir / MANIFEST_FILENAME,
        debug_state_path=result_dir / DEBUG_STATE_FILENAME,
    )


def dag_to_compact_arrays(dag, node_count: int, integer_dtype, node_names: Sequence[object] | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if node_names is None:
        node_names = list(range(node_count))
    if len(node_names) != node_count:
        raise ValueError(f"node_names has length {len(node_names)}, expected {node_count}")
    node_to_idx = {node_name: idx for idx, node_name in enumerate(node_names)}
    dag_with_all_nodes = dag.copy()
    dag_with_all_nodes.add_nodes_from(node_names)
    adjacency_lists: List[List[int]] = []
    node_ptr = np.zeros(node_count + 1, dtype=np.int32)
    total_edges = 0
    for source_idx, source_name in enumerate(node_names):
        targets = sorted(node_to_idx[target] for target in dag_with_all_nodes.successors(source_name) if target in node_to_idx)
        adjacency_lists.append(targets)
        total_edges += len(targets)
        node_ptr[source_idx + 1] = total_edges
    targets = np.empty(total_edges, dtype=integer_dtype)
    cursor = 0
    for entries in adjacency_lists:
        width = len(entries)
        if width:
            targets[cursor: cursor + width] = np.asarray(entries, dtype=integer_dtype)
            cursor += width
    if nx.is_directed_acyclic_graph(dag_with_all_nodes):
        topo_named = list(nx.topological_sort(dag_with_all_nodes))
        topo_order = np.asarray([node_to_idx[node_name] for node_name in topo_named if node_name in node_to_idx], dtype=integer_dtype)
        if topo_order.shape[0] != node_count:
            raise ValueError(f"Topological order has {topo_order.shape[0]} nodes, expected {node_count}")
    else:
        topo_order = np.full(node_count, -1, dtype=integer_dtype)
    return node_ptr, topo_order, targets


def pack_dag_records(records: Sequence[Tuple[int, object]], node_count: int, node_names: Sequence[object] | None = None) -> CompactDagBatch:
    ordered = sorted(((int(task_id), dag) for task_id, dag in records), key=lambda item: item[0])
    integer_dtype = np.int16 if node_count <= np.iinfo(np.int16).max else np.int32
    cell_ids = np.asarray([task_id for task_id, _ in ordered], dtype=np.int32)
    node_ptrs = np.zeros((len(ordered), node_count + 1), dtype=np.int32)
    topo_orders = np.zeros((len(ordered), node_count), dtype=integer_dtype)
    cell_edge_offsets = np.zeros(len(ordered) + 1, dtype=np.int64)
    target_chunks: List[np.ndarray] = []
    edge_cursor = 0
    for idx, (_, dag) in enumerate(ordered):
        node_ptr, topo_order, targets = dag_to_compact_arrays(dag, node_count=node_count, integer_dtype=integer_dtype, node_names=node_names)
        node_ptrs[idx] = node_ptr
        topo_orders[idx] = topo_order
        target_chunks.append(targets)
        edge_cursor += int(targets.shape[0])
        cell_edge_offsets[idx + 1] = edge_cursor
    packed_targets = np.concatenate(target_chunks) if target_chunks else np.empty(0, dtype=integer_dtype)
    return CompactDagBatch(
        cell_ids=cell_ids,
        node_count=int(node_count),
        cell_edge_offsets=cell_edge_offsets,
        node_ptrs=node_ptrs,
        targets=packed_targets,
        topo_orders=topo_orders,
    )


def save_compact_results(result_dir: Path, records: Sequence[Tuple[int, object]], node_count: int, manifest: Dict[str, object], node_names: Sequence[object] | None = None) -> ResultPaths:
    result_dir.mkdir(parents=True, exist_ok=True)
    paths = build_result_paths(result_dir)
    batch = pack_dag_records(records, node_count=node_count, node_names=node_names)
    np.savez_compressed(
        paths.results_path,
        cell_ids=batch.cell_ids,
        node_count=np.int32(batch.node_count),
        cell_edge_offsets=batch.cell_edge_offsets,
        node_ptrs=batch.node_ptrs,
        targets=batch.targets,
        topo_orders=batch.topo_orders,
    )
    with paths.manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    return paths


def load_compact_results(result_dir: Path) -> CompactDagBatch:
    paths = build_result_paths(result_dir)
    if not paths.results_path.exists():
        raise FileNotFoundError(paths.results_path)
    payload = np.load(paths.results_path, allow_pickle=False)
    return CompactDagBatch(
        cell_ids=np.asarray(payload['cell_ids'], dtype=np.int32),
        node_count=int(np.asarray(payload['node_count']).item()),
        cell_edge_offsets=np.asarray(payload['cell_edge_offsets'], dtype=np.int64),
        node_ptrs=np.asarray(payload['node_ptrs'], dtype=np.int32),
        targets=np.asarray(payload['targets']),
        topo_orders=np.asarray(payload['topo_orders']),
    )


def load_manifest(result_dir: Path) -> Dict[str, object]:
    paths = build_result_paths(result_dir)
    with paths.manifest_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def cell_target_slice(batch: CompactDagBatch, cell_pos: int) -> np.ndarray:
    start = int(batch.cell_edge_offsets[cell_pos])
    stop = int(batch.cell_edge_offsets[cell_pos + 1])
    return batch.targets[start:stop]


def decode_cell_edges(batch: CompactDagBatch, cell_pos: int) -> List[Tuple[int, int]]:
    local_targets = cell_target_slice(batch, cell_pos)
    node_ptr = batch.node_ptrs[cell_pos]
    edges: List[Tuple[int, int]] = []
    for source in range(batch.node_count):
        start = int(node_ptr[source])
        stop = int(node_ptr[source + 1])
        if stop <= start:
            continue
        for target in local_targets[start:stop]:
            edges.append((source, int(target)))
    return edges


def iter_cell_compact(batch: CompactDagBatch) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    for cell_pos, cell_id in enumerate(batch.cell_ids.tolist()):
        yield (
            int(cell_id),
            batch.node_ptrs[cell_pos],
            cell_target_slice(batch, cell_pos),
            batch.topo_orders[cell_pos],
        )
