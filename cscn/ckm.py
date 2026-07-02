from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .results import iter_cell_compact, load_compact_results, load_manifest


BETA_MODES = {"blend", "clipped_blend", "expression", "rank_blend", "uniform"}
EDGE_WEIGHT_MODES = {"binary", "tf_prior"}
PROPAGATION_MODES = {"mean_out", "sum", "sym_degree"}


def _safe_column_mean_normalize(node_values: np.ndarray) -> np.ndarray:
    column_means = np.asarray(node_values.mean(axis=0), dtype=np.float32)
    column_means = np.where(np.abs(column_means) > 1e-8, column_means, 1.0).astype(np.float32, copy=False)
    return node_values / column_means


def _rank_normalize_columns(node_values: np.ndarray) -> np.ndarray:
    ranks = np.zeros_like(node_values, dtype=np.float32)
    n_cells = int(node_values.shape[0])
    if n_cells <= 1:
        return np.ones_like(node_values, dtype=np.float32)
    for col_idx in range(int(node_values.shape[1])):
        order = np.argsort(node_values[:, col_idx], kind="mergesort")
        ranks[order, col_idx] = np.arange(n_cells, dtype=np.float32)
    return 1.0 + (ranks / float(n_cells - 1))


def load_expression_from_modules(module_dir: Path) -> pd.DataFrame:
    merged = None
    for module_path in sorted(module_dir.glob('module_*_expression.csv')):
        df = pd.read_csv(module_path, index_col=0)
        merged = df if merged is None else merged.join(df, how='inner')
    if merged is None:
        raise ValueError(f'No module expression files found in {module_dir}')
    return merged


def _edge_coefficients(
    node_ptr: np.ndarray,
    targets: np.ndarray,
    edge_weight_map: Optional[Mapping[Tuple[int, int], float]],
    edge_weight_lambda: float,
    propagation_mode: str,
) -> Dict[Tuple[int, int], float]:
    if propagation_mode not in PROPAGATION_MODES:
        raise ValueError(f"Unsupported propagation_mode: {propagation_mode}")
    coeffs: Dict[Tuple[int, int], float] = {}
    source_totals = np.zeros(int(node_ptr.shape[0]) - 1, dtype=np.float32)
    target_totals = np.zeros(int(node_ptr.shape[0]) - 1, dtype=np.float32)
    for source in range(int(node_ptr.shape[0]) - 1):
        start = int(node_ptr[source])
        stop = int(node_ptr[source + 1])
        if stop <= start:
            continue
        for target in np.asarray(targets[start:stop], dtype=np.int64):
            target_idx = int(target)
            weight = _edge_weight(source, target_idx, edge_weight_map, edge_weight_lambda)
            coeffs[(int(source), target_idx)] = float(weight)
            source_totals[int(source)] += float(weight)
            target_totals[target_idx] += float(weight)
    if propagation_mode == "sum":
        return coeffs
    normalized: Dict[Tuple[int, int], float] = {}
    for (source, target), weight in coeffs.items():
        if propagation_mode == "mean_out":
            denom = float(source_totals[int(source)])
        else:
            denom = float(np.sqrt(float(source_totals[int(source)]) * float(target_totals[int(target)])))
        normalized[(int(source), int(target))] = float(weight) / denom if denom > 1e-8 else float(weight)
    return normalized


def solve_dag_katz(node_ptr: np.ndarray, targets: np.ndarray, topo_order: np.ndarray, beta: np.ndarray, alpha: float) -> np.ndarray:
    values = np.array(beta, dtype=np.float32, copy=True)
    for node in topo_order[::-1]:
        node = int(node)
        start = int(node_ptr[node])
        stop = int(node_ptr[node + 1])
        if stop <= start:
            continue
        values[node] = beta[node] + (alpha * np.sum(values[np.asarray(targets[start:stop], dtype=np.int64)]))
    return values


def _edge_weight(source: int, target: int, weight_map: Optional[Mapping[Tuple[int, int], float]], edge_weight_lambda: float) -> float:
    if not weight_map:
        return 1.0
    prior_weight = float(weight_map.get((int(source), int(target)), 0.0))
    if prior_weight <= 0.0:
        return 1.0
    return float(1.0 + (float(edge_weight_lambda) * prior_weight))


def solve_weighted_dag_katz(
    node_ptr: np.ndarray,
    targets: np.ndarray,
    topo_order: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    edge_weight_map: Optional[Mapping[Tuple[int, int], float]],
    edge_weight_lambda: float,
    propagation_mode: str = "sum",
) -> np.ndarray:
    values = np.array(beta, dtype=np.float32, copy=True)
    edge_coeffs = _edge_coefficients(
        node_ptr=node_ptr,
        targets=targets,
        edge_weight_map=edge_weight_map,
        edge_weight_lambda=edge_weight_lambda,
        propagation_mode=propagation_mode,
    )
    for node in topo_order[::-1]:
        node = int(node)
        start = int(node_ptr[node])
        stop = int(node_ptr[node + 1])
        if stop <= start:
            continue
        weighted_sum = 0.0
        for target in np.asarray(targets[start:stop], dtype=np.int64):
            target_idx = int(target)
            weighted_sum += float(edge_coeffs.get((node, target_idx), 0.0)) * float(values[target_idx])
        values[node] = float(beta[node]) + (float(alpha) * weighted_sum)
    return values


def solve_compact_katz(node_ptr: np.ndarray, targets: np.ndarray, topo_order: np.ndarray, beta: np.ndarray, alpha: float) -> np.ndarray:
    if topo_order.size and int(topo_order[0]) >= 0:
        return solve_dag_katz(node_ptr, targets, topo_order, beta, alpha)
    adjacency = np.zeros((beta.shape[0], beta.shape[0]), dtype=np.float32)
    for source in range(beta.shape[0]):
        start = int(node_ptr[source])
        stop = int(node_ptr[source + 1])
        if stop <= start:
            continue
        adjacency[source, np.asarray(targets[start:stop], dtype=np.int64)] = 1.0
    return np.linalg.solve(np.eye(beta.shape[0], dtype=np.float32) - (alpha * adjacency), beta.astype(np.float32, copy=False))


def solve_weighted_compact_katz(
    node_ptr: np.ndarray,
    targets: np.ndarray,
    topo_order: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    edge_weight_map: Optional[Mapping[Tuple[int, int], float]],
    edge_weight_lambda: float,
    propagation_mode: str = "sum",
) -> np.ndarray:
    if propagation_mode not in PROPAGATION_MODES:
        raise ValueError(f"Unsupported propagation_mode: {propagation_mode}")
    if propagation_mode == "sum" and (not edge_weight_map or float(edge_weight_lambda) == 0.0):
        return solve_compact_katz(node_ptr, targets, topo_order, beta, alpha)
    if topo_order.size and int(topo_order[0]) >= 0:
        return solve_weighted_dag_katz(
            node_ptr=node_ptr,
            targets=targets,
            topo_order=topo_order,
            beta=beta,
            alpha=alpha,
            edge_weight_map=edge_weight_map,
            edge_weight_lambda=edge_weight_lambda,
            propagation_mode=propagation_mode,
        )
    edge_coeffs = _edge_coefficients(
        node_ptr=node_ptr,
        targets=targets,
        edge_weight_map=edge_weight_map,
        edge_weight_lambda=edge_weight_lambda,
        propagation_mode=propagation_mode,
    )
    adjacency = np.zeros((beta.shape[0], beta.shape[0]), dtype=np.float32)
    for (source, target), coefficient in edge_coeffs.items():
        adjacency[int(source), int(target)] = float(coefficient)
    system = np.eye(beta.shape[0], dtype=np.float32) - (float(alpha) * adjacency)
    try:
        return np.linalg.solve(system, beta.astype(np.float32, copy=False))
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, beta.astype(np.float32, copy=False), rcond=None)[0].astype(np.float32, copy=False)


def load_node_values(module_path: Path, result_dir: Path) -> pd.DataFrame:
    manifest = load_manifest(result_dir)
    expr_df = pd.read_csv(module_path, index_col=0)
    node_values_file = manifest.get('node_values_file')
    if node_values_file:
        node_values = np.load(result_dir / str(node_values_file))
        node_names = manifest.get('node_names') or [f'factor_{idx:03d}' for idx in range(node_values.shape[1])]
        return pd.DataFrame(node_values, index=expr_df.index, columns=node_names)
    return expr_df


def _load_module_prior(module_path: Path, node_names: Sequence[str]) -> pd.DataFrame:
    prior_path = module_path.with_name(f"{module_path.stem}_tf_target_prior.csv")
    if not prior_path.exists():
        return pd.DataFrame()
    prior_df = pd.read_csv(prior_path)
    required = {"tf_module_column", "gene_module_column", "tf_target_score"}
    if not required.issubset(prior_df.columns):
        return pd.DataFrame()
    node_set = {str(name) for name in node_names}
    prior_df = prior_df.copy()
    prior_df["tf_module_column"] = prior_df["tf_module_column"].astype(str)
    prior_df["gene_module_column"] = prior_df["gene_module_column"].astype(str)
    prior_df["tf_target_score"] = pd.to_numeric(prior_df["tf_target_score"], errors="coerce").fillna(0.0)
    prior_df = prior_df[
        prior_df["tf_module_column"].isin(node_set)
        & prior_df["gene_module_column"].isin(node_set)
        & (prior_df["tf_module_column"] != prior_df["gene_module_column"])
        & (prior_df["tf_target_score"] > 0)
    ].copy()
    return prior_df


def _build_global_tf_prior_weight_map(
    prior_df: pd.DataFrame,
    node_to_idx: Mapping[str, int],
    tf_top_k: int,
) -> Dict[Tuple[int, int], float]:
    if prior_df.empty:
        return {}
    weight_map: Dict[Tuple[int, int], float] = {}
    for target, rows in prior_df.groupby("gene_module_column", sort=False):
        target_name = str(target)
        if target_name not in node_to_idx:
            continue
        target_rows = rows.sort_values(["tf_target_score", "tf_module_column"], ascending=[False, True]).head(int(tf_top_k))
        for row in target_rows.itertuples(index=False):
            source_name = str(row.tf_module_column)
            if source_name not in node_to_idx:
                continue
            pair = (int(node_to_idx[source_name]), int(node_to_idx[target_name]))
            weight_map[pair] = max(float(row.tf_target_score), float(weight_map.get(pair, 0.0)))
    return weight_map


def build_ckm(
    module_dir: Path,
    result_dir: Path,
    alpha: float = 0.05,
    out_path: Optional[Path] = None,
    beta_mode: str = "expression",
    beta_lambda: float = 1.0,
    edge_weight_mode: str = "binary",
    edge_weight_lambda: float = 1.0,
    tf_top_k: int = 20,
    propagation_mode: str = "sum",
) -> pd.DataFrame:
    if beta_mode not in BETA_MODES:
        raise ValueError(f"Unsupported beta_mode: {beta_mode}")
    if edge_weight_mode not in EDGE_WEIGHT_MODES:
        raise ValueError(f"Unsupported edge_weight_mode: {edge_weight_mode}")
    if propagation_mode not in PROPAGATION_MODES:
        raise ValueError(f"Unsupported propagation_mode: {propagation_mode}")
    if not 0.0 <= float(beta_lambda) <= 1.0:
        raise ValueError("beta_lambda must be within [0, 1]")
    if float(edge_weight_lambda) < 0.0:
        raise ValueError("edge_weight_lambda must be >= 0")
    if int(tf_top_k) < 1:
        raise ValueError("tf_top_k must be >= 1")
    module_frames = []
    for module_path in sorted(module_dir.glob('module_*_expression.csv')):
        module_name = module_path.stem
        batch = load_compact_results(result_dir / module_name)
        node_df = load_node_values(module_path, result_dir / module_name)
        if batch.n_cells != node_df.shape[0]:
            raise ValueError(f'Cell count mismatch for {module_name}: {batch.n_cells} vs {node_df.shape[0]}')
        if batch.node_count != node_df.shape[1]:
            raise ValueError(f'Node count mismatch for {module_name}: {batch.node_count} vs {node_df.shape[1]}')
        node_matrix = node_df.to_numpy(dtype=np.float32, copy=False)
        if beta_mode == "expression":
            beta_matrix = node_matrix
        elif beta_mode == "uniform":
            beta_matrix = np.ones_like(node_matrix, dtype=np.float32)
        elif beta_mode == "blend":
            normalized_matrix = _safe_column_mean_normalize(node_matrix)
            beta_matrix = (float(beta_lambda) * normalized_matrix) + ((1.0 - float(beta_lambda)) * np.ones_like(normalized_matrix, dtype=np.float32))
        elif beta_mode == "clipped_blend":
            normalized_matrix = np.clip(_safe_column_mean_normalize(node_matrix), 0.0, 3.0).astype(np.float32, copy=False)
            beta_matrix = (float(beta_lambda) * normalized_matrix) + ((1.0 - float(beta_lambda)) * np.ones_like(normalized_matrix, dtype=np.float32))
        else:
            normalized_matrix = _rank_normalize_columns(node_matrix)
            beta_matrix = (float(beta_lambda) * normalized_matrix) + ((1.0 - float(beta_lambda)) * np.ones_like(normalized_matrix, dtype=np.float32))
        node_names = [str(column) for column in node_df.columns]
        node_to_idx = {name: idx for idx, name in enumerate(node_names)}
        global_edge_weight_map: Dict[Tuple[int, int], float] = {}
        if edge_weight_mode == "tf_prior":
            prior_df = _load_module_prior(module_path, node_names)
            global_edge_weight_map = _build_global_tf_prior_weight_map(prior_df, node_to_idx, tf_top_k=tf_top_k)
        rows = []
        for row_idx, (_, node_ptr, targets, topo_order) in enumerate(iter_cell_compact(batch)):
            beta = beta_matrix[row_idx]
            edge_weight_map: Optional[Mapping[Tuple[int, int], float]]
            if edge_weight_mode == "binary":
                edge_weight_map = None
            else:
                edge_weight_map = global_edge_weight_map
            rows.append(
                solve_weighted_compact_katz(
                    node_ptr=node_ptr,
                    targets=targets,
                    topo_order=topo_order,
                    beta=beta,
                    alpha=alpha,
                    edge_weight_map=edge_weight_map,
                    edge_weight_lambda=edge_weight_lambda,
                    propagation_mode=propagation_mode,
                )
            )
        module_frames.append(pd.DataFrame(np.vstack(rows), index=node_df.index, columns=node_df.columns))
    merged = pd.concat(module_frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path)
    return merged
