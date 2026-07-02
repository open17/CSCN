from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from .tf_motif import (
    DEFAULT_MOTIF_REL_SCORE_THRESHOLD,
    JASPAR_RELEASE,
    canonical_gene_token,
    ensure_jaspar_cache,
    load_jaspar_motif_records,
    load_peak_sequences,
    load_tf_tokens,
    map_tf_tokens_to_symbols,
    scan_peak_sequence_for_tf_hits,
)


DEFAULT_STRONG_PERCENT = 0.03
DEFAULT_AMBIGUOUS_PERCENT = 0.15
DEFAULT_MODULE_MIN_SIZE = 20
DEFAULT_MODULE_MAX_SIZE = 120


@dataclass(frozen=True)
class Interval:
    chrom: str
    start: int
    end: int

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2


@dataclass
class FeatureTable:
    names: np.ndarray
    ids: np.ndarray
    feature_types: np.ndarray
    intervals: np.ndarray
    genomes: np.ndarray


@dataclass(frozen=True)
class GtfMetadata:
    tss_map: Dict[str, Interval]
    alias_to_symbol: Dict[str, str]


def make_unique(names: Sequence[str], ids: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for name, feature_id in zip(names, ids):
        if name not in seen:
            seen[name] = 0
            unique.append(name)
            continue
        seen[name] += 1
        unique.append(f"{name}|{feature_id}")
    return unique


def parse_interval(text: str) -> Optional[Interval]:
    if not text or ":" not in text or "-" not in text:
        return None
    chrom, coords = text.split(":", 1)
    start_text, end_text = coords.split("-", 1)
    return Interval(chrom=chrom, start=int(start_text), end=int(end_text))


def parse_gtf_attributes(text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for chunk in text.strip().split(";"):
        chunk = chunk.strip()
        if not chunk or " " not in chunk:
            continue
        key, value = chunk.split(" ", 1)
        attrs[key] = value.strip().strip('"')
    return attrs


def strip_id_version(text: object) -> str:
    token = str(text or "").strip()
    if not token:
        return ""
    return token.split(".", 1)[0]


def load_gtf_metadata(path: Optional[Path]) -> GtfMetadata:
    if path is None:
        return GtfMetadata(tss_map={}, alias_to_symbol={})
    tss_map: Dict[str, Interval] = {}
    alias_to_symbol: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom, _, _, start_text, end_text, _, strand, _, attr_text = fields
            attrs = parse_gtf_attributes(attr_text)
            start = int(start_text)
            end = int(end_text)
            tss = start if strand == "+" else end
            interval = Interval(chrom=chrom, start=tss, end=tss)
            gene_name = attrs.get("gene_name")
            gene_id = attrs.get("gene_id")
            for key in (gene_name, gene_id, strip_id_version(gene_id)):
                if key:
                    tss_map[key] = interval
            symbol = gene_name or gene_id
            if symbol:
                for alias in (gene_name, gene_id, strip_id_version(gene_id)):
                    canonical = canonical_gene_token(alias)
                    if canonical:
                        alias_to_symbol[canonical] = str(symbol)
    return GtfMetadata(tss_map=tss_map, alias_to_symbol=alias_to_symbol)


def load_gene_tss_from_gtf(path: Optional[Path]) -> Dict[str, Interval]:
    return load_gtf_metadata(path).tss_map


def resolve_gene_symbols(
    gene_names: Sequence[str],
    gene_ids: Sequence[str],
    alias_to_symbol: Mapping[str, str],
) -> np.ndarray:
    symbols: List[str] = []
    for name, gene_id in zip(gene_names, gene_ids):
        candidates = (
            canonical_gene_token(name),
            canonical_gene_token(gene_id),
            canonical_gene_token(strip_id_version(gene_id)),
        )
        symbol = None
        for candidate in candidates:
            if candidate in alias_to_symbol:
                symbol = alias_to_symbol[candidate]
                break
        symbols.append(str(symbol or name or gene_id))
    return np.asarray(symbols, dtype=object)


def load_10x_h5(path: Path) -> Tuple[sparse.csc_matrix, FeatureTable, np.ndarray]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        shape = tuple(matrix["shape"][:])
        feature_group = matrix["features"]
        features = FeatureTable(
            names=np.array([value.decode("utf-8") for value in feature_group["name"][:]]),
            ids=np.array([value.decode("utf-8") for value in feature_group["id"][:]]),
            feature_types=np.array([value.decode("utf-8") for value in feature_group["feature_type"][:]]),
            intervals=np.array([value.decode("utf-8") for value in feature_group["interval"][:]]),
            genomes=np.array([value.decode("utf-8") for value in feature_group["genome"][:]]),
        )
        barcodes = np.array([value.decode("utf-8") for value in matrix["barcodes"][:]])
    feature_by_barcode = sparse.csc_matrix((data, indices, indptr), shape=shape)
    return feature_by_barcode, features, barcodes


def subset_feature_type(
    feature_by_barcode: sparse.csc_matrix,
    feature_table: FeatureTable,
    feature_type: str,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    mask = feature_table.feature_types == feature_type
    matrix = feature_by_barcode[mask, :].transpose().tocsr().astype(np.float32)
    return matrix, feature_table.names[mask], feature_table.ids[mask], feature_table.intervals[mask]


def log_normalize_sparse(matrix: sparse.csr_matrix) -> Tuple[sparse.csr_matrix, np.ndarray]:
    cell_sums = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1e4, cell_sums, out=np.zeros_like(cell_sums), where=cell_sums > 0)
    normalized = matrix.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized, cell_sums


def auto_log1p_dense(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    values = df.to_numpy(dtype=np.float32, copy=True)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values, "empty"
    rounded = np.isclose(finite, np.round(finite))
    looks_like_counts = rounded.mean() > 0.95 and finite.max() > 20
    if not looks_like_counts:
        return values, "already_log_like"
    cell_sums = values.sum(axis=1)
    scale = np.divide(1e4, cell_sums, out=np.zeros_like(cell_sums), where=cell_sums > 0)
    values *= scale[:, None]
    values = np.log1p(values)
    return values.astype(np.float32, copy=False), "counts_to_log1p"


def select_top_variable_sparse(
    matrix: sparse.csr_matrix,
    names: np.ndarray,
    ids: np.ndarray,
    intervals: np.ndarray,
    top_n: int,
    min_cells: int,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    valid_mask = detected >= min_cells
    if not np.any(valid_mask):
        raise ValueError("No features passed the min_cells filter")
    filtered = matrix[:, valid_mask]
    filtered_names = names[valid_mask]
    filtered_ids = ids[valid_mask]
    filtered_intervals = intervals[valid_mask]
    mean = np.asarray(filtered.mean(axis=0)).ravel()
    mean_sq = np.asarray(filtered.power(2).mean(axis=0)).ravel()
    variance = mean_sq - mean**2
    order = np.argsort(variance)[::-1]
    top = order[: min(top_n, order.size)]
    return filtered[:, top], filtered_names[top], filtered_ids[top], filtered_intervals[top]


def select_top_cells(
    matrices: Sequence[sparse.csr_matrix],
    barcodes: np.ndarray,
    scores: np.ndarray,
    top_n_cells: Optional[int],
) -> Tuple[List[sparse.csr_matrix], np.ndarray]:
    if top_n_cells is None or top_n_cells >= len(barcodes):
        return [matrix for matrix in matrices], barcodes
    order = np.argsort(scores)[::-1][:top_n_cells]
    return [matrix[order] for matrix in matrices], barcodes[order]


def binary_copy(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    out = matrix.copy().tocsr()
    if out.nnz:
        out.data = np.ones_like(out.data, dtype=np.float32)
    return out


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.array([], dtype=np.float32)
    return matrix[np.triu_indices(matrix.shape[0], k=1)].astype(np.float32, copy=False)


def percentile_threshold(values: np.ndarray, keep_fraction: float) -> float:
    if values.size == 0:
        return float("inf")
    return float(np.quantile(values, max(0.0, min(1.0, 1.0 - keep_fraction))))


def safe_corr_1d(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float32, copy=False) - np.mean(left)
    right = right.astype(np.float32, copy=False) - np.mean(right)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


def columnwise_corr(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=False)
    matrix = matrix.astype(np.float32, copy=False)
    centered_vector = vector - vector.mean()
    centered_matrix = matrix - matrix.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(centered_vector) * np.linalg.norm(centered_matrix, axis=0)
    numer = centered_vector @ centered_matrix
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-8)


def build_interval_index(parsed_intervals: Sequence[Optional[Interval]]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    by_chr: Dict[str, List[Tuple[int, int, int]]] = {}
    for idx, interval in enumerate(parsed_intervals):
        if interval is None:
            continue
        by_chr.setdefault(interval.chrom, []).append((interval.start, interval.end, idx))
    index: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for chrom, entries in by_chr.items():
        entries.sort(key=lambda item: item[0])
        index[chrom] = (
            np.array([item[0] for item in entries], dtype=np.int64),
            np.array([item[1] for item in entries], dtype=np.int64),
            np.array([item[2] for item in entries], dtype=np.int64),
        )
    return index


def query_interval_ids(
    interval_index: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    chrom: str,
    left: int,
    right: int,
) -> np.ndarray:
    payload = interval_index.get(chrom)
    if payload is None:
        return np.array([], dtype=np.int64)
    starts, ends, ids = payload
    stop = np.searchsorted(starts, right, side="right")
    if stop == 0:
        return np.array([], dtype=np.int64)
    mask = ends[:stop] >= left
    return ids[:stop][mask]


def build_neighbor_graph(embedding: np.ndarray, n_neighbors: int) -> sparse.csr_matrix:
    n_cells = embedding.shape[0]
    if n_cells <= 1:
        return sparse.eye(n_cells, format="csr", dtype=np.float32)
    k = min(max(1, n_neighbors), n_cells - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neighbors.fit(embedding)
    indices = neighbors.kneighbors(return_distance=False)
    rows: List[int] = []
    cols: List[int] = []
    for cell_idx, row in enumerate(indices):
        chosen = row[1:]
        rows.extend([cell_idx] * (len(chosen) + 1))
        cols.extend([cell_idx, *chosen.tolist()])
    graph = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_cells, n_cells))
    graph = graph.maximum(graph.transpose()).tocsr()
    graph = graph + sparse.eye(n_cells, dtype=np.float32, format="csr")
    graph.data = np.ones_like(graph.data, dtype=np.float32)
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    inv_row = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    return sparse.diags(inv_row).dot(graph).tocsr()


def build_shared_neighbor_graph(
    log_rna: np.ndarray,
    atac_for_lsi: sparse.csr_matrix,
    n_neighbors: int,
    rna_components: int,
    atac_components: int,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    n_cells = log_rna.shape[0]
    rna_components = max(1, min(rna_components, n_cells - 1, log_rna.shape[1]))
    atac_components = max(1, min(atac_components, n_cells - 1, atac_for_lsi.shape[1]))
    rna_emb = PCA(n_components=rna_components, random_state=42).fit_transform(log_rna)
    binary_peaks = binary_copy(atac_for_lsi)
    row_sums = np.asarray(binary_peaks.sum(axis=1)).ravel()
    inv_row = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    tf = sparse.diags(inv_row).dot(binary_peaks)
    peak_df = np.asarray((binary_peaks > 0).sum(axis=0)).ravel()
    idf = np.log1p(binary_peaks.shape[0] / np.maximum(peak_df, 1))
    atac_emb = TruncatedSVD(n_components=atac_components, random_state=42).fit_transform(tf.multiply(idf))
    shared_embedding = np.hstack([rna_emb, atac_emb]).astype(np.float32, copy=False)
    return build_neighbor_graph(shared_embedding, n_neighbors), rna_emb, atac_emb


def column_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[1] == 1:
        return np.array([[1.0]], dtype=np.float32)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=0, keepdims=True)
    scale[scale <= 1e-8] = 1.0
    normalized = centered / scale
    corr = normalized.T @ normalized
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32, copy=False)


def adjacency_from_corr(corr: np.ndarray, soft_power: int) -> np.ndarray:
    adjacency = np.abs(corr) ** soft_power
    np.fill_diagonal(adjacency, 0.0)
    return adjacency.astype(np.float32, copy=False)


def tom_similarity(adjacency: np.ndarray) -> np.ndarray:
    if adjacency.shape[0] == 1:
        return np.array([[1.0]], dtype=np.float32)
    degree = adjacency.sum(axis=1)
    shared = adjacency @ adjacency
    denom = np.minimum.outer(degree, degree) + 1.0 - adjacency
    tom = np.divide(shared + adjacency, denom, out=np.zeros_like(shared), where=denom > 1e-8)
    np.fill_diagonal(tom, 1.0)
    return tom.astype(np.float32, copy=False)


def cluster_from_similarity(
    similarity: np.ndarray,
    cut_height: Optional[float] = None,
    max_clusters: Optional[int] = None,
) -> np.ndarray:
    n_features = similarity.shape[0]
    if n_features <= 1:
        return np.ones(n_features, dtype=int)
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    if max_clusters is not None:
        return fcluster(linkage_matrix, t=max_clusters, criterion="maxclust").astype(int)
    if cut_height is None:
        raise ValueError("cut_height must be provided when max_clusters is None")
    return fcluster(linkage_matrix, t=cut_height, criterion="distance").astype(int)


def recursively_split_indices(
    expression: np.ndarray,
    indices: np.ndarray,
    module_max: int,
    soft_power: int,
    cut_height: float,
) -> List[np.ndarray]:
    if indices.size <= module_max:
        return [indices]
    local_expression = expression[:, indices]
    similarity = tom_similarity(adjacency_from_corr(column_correlation(local_expression), soft_power))
    labels = cluster_from_similarity(similarity, cut_height=cut_height)
    unique_labels = np.unique(labels)
    if unique_labels.size == 1:
        forced = int(np.ceil(indices.size / module_max))
        labels = cluster_from_similarity(similarity, max_clusters=forced)
        unique_labels = np.unique(labels)
        if unique_labels.size == 1:
            order = np.argsort(local_expression.var(axis=0))[::-1]
            return [indices[order[i : i + module_max]] for i in range(0, indices.size, module_max)]
    out: List[np.ndarray] = []
    next_cut = max(0.35, cut_height - 0.05)
    for label in unique_labels:
        chunk = indices[labels == label]
        out.extend(recursively_split_indices(expression, chunk, module_max, soft_power, next_cut))
    return out


def enforce_module_size_policy(
    expression: np.ndarray,
    modules: Sequence[np.ndarray],
    module_min: int,
    module_max: int,
    soft_power: int,
    cut_height: float,
) -> List[np.ndarray]:
    split_modules: List[np.ndarray] = []
    for module in modules:
        split_modules.extend(recursively_split_indices(expression, np.asarray(module, dtype=np.int64), module_max, soft_power, cut_height))
    keepers = [module for module in split_modules if module.size >= module_min]
    small_modules = [module for module in split_modules if module.size < module_min]
    if small_modules:
        small_modules = sorted((np.asarray(module, dtype=np.int64) for module in small_modules), key=len, reverse=True)
        centroids = [expression[:, module].mean(axis=1) for module in small_modules]
        used = [False] * len(small_modules)
        for idx, module in enumerate(small_modules):
            if used[idx]:
                continue
            used[idx] = True
            combined = module.copy()
            centroid = centroids[idx].copy()
            while combined.size < module_min:
                best_idx = None
                best_score = -np.inf
                for candidate_idx, candidate in enumerate(small_modules):
                    if used[candidate_idx] or combined.size + candidate.size > module_max:
                        continue
                    score = safe_corr_1d(centroid, centroids[candidate_idx])
                    if score > best_score:
                        best_score = score
                        best_idx = candidate_idx
                if best_idx is None:
                    break
                used[best_idx] = True
                combined = np.concatenate([combined, small_modules[best_idx]])
                centroid = expression[:, combined].mean(axis=1)
            if combined.size >= module_min:
                keepers.append(np.unique(combined))
    final_modules = [module for module in keepers if module_min <= module.size <= module_max]
    final_modules.sort(key=lambda module: (-module.size, int(module[0])))
    return final_modules


def build_wgcna_modules(
    expression: np.ndarray,
    soft_power: int,
    cut_height: float,
    module_min: int,
    module_max: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    global_corr = column_correlation(expression)
    global_adj = adjacency_from_corr(global_corr, soft_power)
    global_tom = tom_similarity(global_adj)
    labels = cluster_from_similarity(global_tom, cut_height=cut_height)
    raw_modules = [np.where(labels == label)[0] for label in np.unique(labels)]
    modules = enforce_module_size_policy(expression, raw_modules, module_min, module_max, soft_power, cut_height)
    if not modules:
        modules = enforce_module_size_policy(expression, [np.arange(expression.shape[1], dtype=np.int64)], module_min, module_max, soft_power, cut_height)
    return modules, global_adj, global_tom


def build_gene_anchors(
    gene_names: np.ndarray,
    gene_ids: np.ndarray,
    gene_intervals: np.ndarray,
    gtf_path: Optional[Path],
) -> Tuple[List[Optional[Interval]], List[str]]:
    tss_map = load_gene_tss_from_gtf(gtf_path)
    anchors: List[Optional[Interval]] = []
    sources: List[str] = []
    for name, gene_id, interval_text in zip(gene_names, gene_ids, gene_intervals):
        anchor = tss_map.get(name) or tss_map.get(gene_id) or tss_map.get(strip_id_version(gene_id))
        if anchor is not None:
            anchors.append(anchor)
            sources.append("gtf_tss")
            continue
        fallback = parse_interval(interval_text)
        if fallback is None:
            anchors.append(None)
            sources.append("missing")
            continue
        anchors.append(Interval(chrom=fallback.chrom, start=fallback.start, end=fallback.start))
        sources.append("h5_interval_start")
    return anchors, sources


def collect_gene_peak_candidates(
    gene_anchors: Sequence[Optional[Interval]],
    peak_index: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    promoter_window: int,
    distal_window: int,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], np.ndarray]:
    candidates: Dict[int, Dict[str, np.ndarray]] = {}
    all_peak_ids: set[int] = set()
    for gene_idx, anchor in enumerate(gene_anchors):
        if anchor is None:
            candidates[gene_idx] = {"promoter_global": np.array([], dtype=np.int64), "distal_global": np.array([], dtype=np.int64)}
            continue
        promoter = query_interval_ids(peak_index, anchor.chrom, anchor.start - promoter_window, anchor.start + promoter_window)
        distal = query_interval_ids(peak_index, anchor.chrom, anchor.start - distal_window, anchor.start + distal_window)
        if promoter.size:
            distal = distal[~np.isin(distal, promoter)]
        candidates[gene_idx] = {
            "promoter_global": promoter.astype(np.int64, copy=False),
            "distal_global": distal.astype(np.int64, copy=False),
        }
        all_peak_ids.update(promoter.tolist())
        all_peak_ids.update(distal.tolist())
    return candidates, np.array(sorted(all_peak_ids), dtype=np.int64)


def map_candidates_to_local(
    gene_candidates: Dict[int, Dict[str, np.ndarray]],
    global_to_local: Dict[int, int],
) -> Dict[int, Dict[str, np.ndarray]]:
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for gene_idx, payload in gene_candidates.items():
        promoter_local = [global_to_local[peak_id] for peak_id in payload["promoter_global"] if peak_id in global_to_local]
        distal_local = [global_to_local[peak_id] for peak_id in payload["distal_global"] if peak_id in global_to_local]
        out[gene_idx] = {
            "promoter_local": np.array(promoter_local, dtype=np.int64),
            "distal_local": np.array(distal_local, dtype=np.int64),
            "promoter_global": payload["promoter_global"],
            "distal_global": payload["distal_global"],
        }
    return out


def interval_distance(anchor: Optional[Interval], peak_interval: Optional[Interval]) -> float:
    if anchor is None or peak_interval is None:
        return float("nan")
    return float(abs(peak_interval.center - anchor.start))


def compute_gene_atac_support(
    smoothed_rna: np.ndarray,
    smoothed_atac_local: sparse.csr_matrix,
    gene_names: np.ndarray,
    gene_ids: np.ndarray,
    gene_anchors: Sequence[Optional[Interval]],
    anchor_sources: Sequence[str],
    peak_names_local: np.ndarray,
    peak_intervals_local: Sequence[Optional[Interval]],
    local_peak_ids: np.ndarray,
    gene_candidates_local: Dict[int, Dict[str, np.ndarray]],
    max_distal_links: int,
) -> Tuple[np.ndarray, List[set], pd.DataFrame]:
    n_cells, n_genes = smoothed_rna.shape
    gene_activity = np.zeros((n_cells, n_genes), dtype=np.float32)
    distal_peak_sets: List[set] = []
    rows: List[Dict[str, object]] = []
    for gene_idx in range(n_genes):
        gene_vector = smoothed_rna[:, gene_idx].astype(np.float32, copy=False)
        payload = gene_candidates_local[gene_idx]
        promoter_local = payload["promoter_local"]
        distal_local = payload["distal_local"]
        anchor = gene_anchors[gene_idx]
        promoter_signal = np.zeros(n_cells, dtype=np.float32)
        if promoter_local.size:
            promoter_signal = np.asarray(smoothed_atac_local[:, promoter_local].mean(axis=1)).ravel().astype(np.float32)
            for peak_local in promoter_local:
                peak_interval = peak_intervals_local[peak_local]
                rows.append({
                    "gene": gene_names[gene_idx],
                    "gene_id": gene_ids[gene_idx],
                    "peak_name": peak_names_local[peak_local],
                    "peak_interval": "" if peak_interval is None else f"{peak_interval.chrom}:{peak_interval.start}-{peak_interval.end}",
                    "link_kind": "promoter",
                    "distance_to_anchor": interval_distance(anchor, peak_interval),
                    "link_score": 1.0,
                    "retained_rank": 0,
                    "anchor_source": anchor_sources[gene_idx],
                })
        distal_signal = np.zeros(n_cells, dtype=np.float32)
        kept_distal_global: set = set()
        if distal_local.size:
            peak_dense = smoothed_atac_local[:, distal_local].toarray().astype(np.float32, copy=False)
            gene_rank = rankdata(gene_vector, method="average").astype(np.float32, copy=False)
            peak_rank = np.column_stack([rankdata(peak_dense[:, idx], method="average") for idx in range(peak_dense.shape[1])]).astype(np.float32, copy=False)
            corr = columnwise_corr(gene_rank, peak_rank)
            distances = np.array([interval_distance(anchor, peak_intervals_local[peak_idx]) for peak_idx in distal_local], dtype=np.float32)
            scores = np.abs(corr) * np.exp(-distances / 200000.0)
            order = np.argsort(scores)[::-1]
            keep = [idx for idx in order[:max_distal_links] if scores[idx] > 0]
            if keep:
                keep_local = distal_local[np.array(keep, dtype=np.int64)]
                keep_weights = scores[np.array(keep, dtype=np.int64)]
                normalized = keep_weights / np.maximum(keep_weights.sum(), 1e-8)
                distal_signal = peak_dense[:, keep] @ normalized
                kept_distal_global = set(local_peak_ids[keep_local].tolist())
                for retained_rank, (peak_local, score) in enumerate(zip(keep_local.tolist(), keep_weights.tolist()), start=1):
                    peak_interval = peak_intervals_local[peak_local]
                    rows.append({
                        "gene": gene_names[gene_idx],
                        "gene_id": gene_ids[gene_idx],
                        "peak_name": peak_names_local[peak_local],
                        "peak_interval": "" if peak_interval is None else f"{peak_interval.chrom}:{peak_interval.start}-{peak_interval.end}",
                        "link_kind": "distal",
                        "distance_to_anchor": interval_distance(anchor, peak_interval),
                        "link_score": float(score),
                        "retained_rank": retained_rank,
                        "anchor_source": anchor_sources[gene_idx],
                    })
        gene_activity[:, gene_idx] = promoter_signal + distal_signal
        distal_peak_sets.append(kept_distal_global)
    links_df = pd.DataFrame(rows)
    if links_df.empty:
        links_df = pd.DataFrame(columns=["gene", "gene_id", "peak_name", "peak_interval", "link_kind", "distance_to_anchor", "link_score", "retained_rank", "anchor_source"])
    return gene_activity, distal_peak_sets, links_df


def modules_to_label_array(modules: Sequence[np.ndarray], n_genes: int) -> np.ndarray:
    labels = np.full(n_genes, -1, dtype=int)
    for module_idx, module in enumerate(modules, start=1):
        labels[np.asarray(module, dtype=np.int64)] = module_idx
    return labels


def summarize_modules(modules: Sequence[np.ndarray]) -> Dict[str, float]:
    sizes = np.array([module.size for module in modules], dtype=np.int64)
    if sizes.size == 0:
        return {"module_count": 0, "min_size": 0, "median_size": 0, "max_size": 0, "estimated_pairwise_load": 0}
    return {
        "module_count": int(sizes.size),
        "min_size": int(sizes.min()),
        "median_size": float(np.median(sizes)),
        "max_size": int(sizes.max()),
        "estimated_pairwise_load": int(np.sum(sizes * np.maximum(sizes - 1, 0) // 2)),
    }


def build_module_allowed_pair_priors(
    modules: Sequence[np.ndarray],
    smoothed_rna: np.ndarray,
    gene_activity: np.ndarray,
    distal_peak_sets: Sequence[set],
    gene_names: np.ndarray,
    soft_power: int,
    strong_percent: float,
    ambiguous_percent: float,
    global_adj: Optional[np.ndarray],
    scope: str = "local",
) -> Dict[str, pd.DataFrame]:
    priors: Dict[str, pd.DataFrame] = {}
    global_values = upper_triangle_values(global_adj) if global_adj is not None else np.array([], dtype=np.float32)
    global_strong = percentile_threshold(global_values, strong_percent)
    global_ambiguous = percentile_threshold(global_values, ambiguous_percent)
    columns = [
        "gene_left",
        "gene_right",
        "local_adjacency",
        "strong_edge",
        "ambiguous_edge",
        "promoter_support",
        "distal_support",
        "support_rule",
    ]
    for module_idx, module in enumerate(modules, start=1):
        module_id = f"M{module_idx:03d}"
        module = np.asarray(module, dtype=np.int64)
        rows: List[Dict[str, object]] = []
        if module.size >= 2:
            local_expr = smoothed_rna[:, module]
            local_adj = adjacency_from_corr(column_correlation(local_expr), soft_power)
            local_values = upper_triangle_values(local_adj)
            if scope == "global" and global_values.size:
                strong_threshold = global_strong
                ambiguous_threshold = global_ambiguous
            else:
                strong_threshold = percentile_threshold(local_values, strong_percent)
                ambiguous_threshold = percentile_threshold(local_values, ambiguous_percent)
            local_activity = gene_activity[:, module]
            activity_corr = np.abs(column_correlation(local_activity))
            promoter_values = upper_triangle_values(activity_corr)
            promoter_threshold = percentile_threshold(promoter_values, 0.20)
            promoter_support = activity_corr >= promoter_threshold
            np.fill_diagonal(promoter_support, False)
            distal_support = np.zeros((module.size, module.size), dtype=bool)
            for left_idx in range(module.size):
                left_peaks = distal_peak_sets[module[left_idx]]
                if not left_peaks:
                    continue
                for right_idx in range(left_idx + 1, module.size):
                    if left_peaks.intersection(distal_peak_sets[module[right_idx]]):
                        distal_support[left_idx, right_idx] = True
                        distal_support[right_idx, left_idx] = True
            strong_mask = local_adj >= strong_threshold
            ambiguous_mask = (local_adj >= ambiguous_threshold) & ~strong_mask
            allow_mask = strong_mask | (ambiguous_mask & (promoter_support | distal_support))
            np.fill_diagonal(allow_mask, False)
            for left_idx in range(module.size):
                for right_idx in range(left_idx + 1, module.size):
                    if not allow_mask[left_idx, right_idx]:
                        continue
                    support_tokens: List[str] = []
                    if strong_mask[left_idx, right_idx]:
                        support_tokens.append("strong_rna")
                    if ambiguous_mask[left_idx, right_idx] and promoter_support[left_idx, right_idx]:
                        support_tokens.append("ambiguous_with_promoter")
                    if ambiguous_mask[left_idx, right_idx] and distal_support[left_idx, right_idx]:
                        support_tokens.append("ambiguous_with_distal")
                    if not support_tokens:
                        support_tokens.append("supported")
                    rows.append(
                        {
                            "gene_left": str(gene_names[module[left_idx]]),
                            "gene_right": str(gene_names[module[right_idx]]),
                            "local_adjacency": float(local_adj[left_idx, right_idx]),
                            "strong_edge": bool(strong_mask[left_idx, right_idx]),
                            "ambiguous_edge": bool(ambiguous_mask[left_idx, right_idx]),
                            "promoter_support": bool(promoter_support[left_idx, right_idx]),
                            "distal_support": bool(distal_support[left_idx, right_idx]),
                            "support_rule": "|".join(support_tokens),
                        }
                    )
        priors[module_id] = pd.DataFrame(rows, columns=columns)
    return priors


def minmax_normalize(values: Sequence[float], fill_value: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.shape, float(fill_value), dtype=np.float32)
    valid = arr[finite]
    lo = float(valid.min())
    hi = float(valid.max())
    if hi - lo <= 1e-8:
        out = np.full(arr.shape, float(fill_value), dtype=np.float32)
        out[~finite] = 0.0
        return out
    out = np.zeros(arr.shape, dtype=np.float32)
    out[finite] = (valid - lo) / (hi - lo)
    return out


def build_gene_alias_rows(
    modules: Sequence[np.ndarray],
    gene_names: np.ndarray,
    gene_ids: np.ndarray,
    gene_symbols: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for module_idx, module in enumerate(modules, start=1):
        module_id = f"M{module_idx:03d}"
        for order_in_module, gene_idx in enumerate(np.asarray(module, dtype=np.int64).tolist(), start=1):
            rows.append(
                {
                    "module_id": module_id,
                    "module_column": str(gene_names[gene_idx]),
                    "gene_id": str(gene_ids[gene_idx]),
                    "gene_symbol": str(gene_symbols[gene_idx]),
                    "order_in_module": int(order_in_module),
                }
            )
    return pd.DataFrame(rows, columns=["module_id", "module_column", "gene_id", "gene_symbol", "order_in_module"])


def build_peak_gene_links(
    *,
    expression: np.ndarray,
    peak_values: np.ndarray,
    gene_names: np.ndarray,
    gene_ids: np.ndarray,
    gene_symbols: np.ndarray,
    gene_anchors: Sequence[Optional[Interval]],
    anchor_sources: Sequence[str],
    peak_names: np.ndarray,
    peak_intervals: Sequence[Optional[Interval]],
    peak_gene_window: int,
    corr_threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    peak_index = build_interval_index(peak_intervals)
    peak_medians = np.median(peak_values, axis=0).astype(np.float32, copy=False) if peak_values.size else np.array([], dtype=np.float32)
    for gene_idx, anchor in enumerate(gene_anchors):
        if anchor is None:
            continue
        candidate_peak_ids = query_interval_ids(
            peak_index,
            anchor.chrom,
            anchor.start - int(peak_gene_window),
            anchor.start + int(peak_gene_window),
        )
        if candidate_peak_ids.size == 0:
            continue
        gene_rank = rankdata(expression[:, gene_idx], method="average").astype(np.float32, copy=False)
        peak_rank = np.column_stack(
            [rankdata(peak_values[:, peak_idx], method="average").astype(np.float32, copy=False) for peak_idx in candidate_peak_ids]
        )
        corr = columnwise_corr(gene_rank, peak_rank)
        keep_positions = np.where(corr > float(corr_threshold))[0]
        for pos in keep_positions.tolist():
            peak_idx = int(candidate_peak_ids[pos])
            peak_interval = peak_intervals[peak_idx]
            rows.append(
                {
                    "gene_module_column": str(gene_names[gene_idx]),
                    "gene_gene_id": str(gene_ids[gene_idx]),
                    "gene_symbol": str(gene_symbols[gene_idx]),
                    "peak_name": str(peak_names[peak_idx]),
                    "peak_interval": "" if peak_interval is None else f"{peak_interval.chrom}:{peak_interval.start}-{peak_interval.end}",
                    "peak_gene_corr": float(corr[pos]),
                    "distance_to_tss": interval_distance(anchor, peak_interval),
                    "peak_accessibility_median": float(peak_medians[peak_idx]) if peak_medians.size else 0.0,
                    "association_pass": True,
                    "anchor_source": str(anchor_sources[gene_idx]),
                }
            )
    columns = [
        "gene_module_column",
        "gene_gene_id",
        "gene_symbol",
        "peak_name",
        "peak_interval",
        "peak_gene_corr",
        "distance_to_tss",
        "peak_accessibility_median",
        "association_pass",
        "anchor_source",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_tf_motif_target_priors(
    *,
    modules: Sequence[np.ndarray],
    gene_names: np.ndarray,
    gene_ids: np.ndarray,
    gene_symbols: np.ndarray,
    gene_peak_links: pd.DataFrame,
    peak_names: np.ndarray,
    peak_intervals: Sequence[Optional[Interval]],
    peak_values: np.ndarray,
    tf_list_path: Optional[Path],
    genome_fasta: Path,
    outdir: Path,
    gtf_alias_to_symbol: Mapping[str, str],
    motif_rel_threshold: float = DEFAULT_MOTIF_REL_SCORE_THRESHOLD,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    empty_prior = pd.DataFrame(
        columns=[
            "module_id",
            "tf_module_column",
            "tf_gene_id",
            "tf_symbol",
            "gene_module_column",
            "gene_gene_id",
            "gene_symbol",
            "tf_target_score",
            "supporting_peak_name",
            "supporting_peak_interval",
            "peak_accessibility_score",
            "peak_gene_corr_score",
            "motif_rel_score",
            "motif_id",
            "motif_name",
            "motif_strand",
            "motif_position",
        ]
    )
    empty_triplets = pd.DataFrame(
        columns=[
            "module_id",
            "tf_module_column",
            "tf_gene_id",
            "tf_symbol",
            "gene_module_column",
            "gene_gene_id",
            "gene_symbol",
            "peak_name",
            "peak_interval",
            "peak_accessibility_score",
            "peak_gene_corr_score",
            "motif_rel_score",
            "triplet_score",
            "motif_id",
            "motif_name",
            "motif_strand",
            "motif_position",
        ]
    )
    if gene_peak_links.empty:
        return {f"M{idx:03d}": empty_prior.copy() for idx in range(1, len(modules) + 1)}, empty_triplets

    dataset_tf_symbols = {canonical_gene_token(symbol) for symbol in gene_symbols if canonical_gene_token(symbol)}
    tf_filter_tokens = load_tf_tokens(tf_list_path)
    tf_filter_symbols = map_tf_tokens_to_symbols(tf_filter_tokens, gtf_alias_to_symbol) if tf_filter_tokens else None
    motif_symbol_filter = dataset_tf_symbols if tf_filter_symbols is None else (dataset_tf_symbols & tf_filter_symbols)
    jaspar_path = ensure_jaspar_cache(Path(outdir) / ".runtime" / "jaspar")
    motif_records_by_symbol = load_jaspar_motif_records(jaspar_path, motif_symbol_filter)
    if not motif_records_by_symbol:
        return {f"M{idx:03d}": empty_prior.copy() for idx in range(1, len(modules) + 1)}, empty_triplets

    peak_interval_by_name = {str(name): interval for name, interval in zip(peak_names.tolist(), peak_intervals) if interval is not None}
    relevant_peak_names = sorted({str(name) for name in gene_peak_links["peak_name"].dropna().astype(str)})
    peak_sequences = load_peak_sequences(
        genome_fasta=genome_fasta,
        peak_intervals={name: peak_interval_by_name.get(name) for name in relevant_peak_names},
    )
    if not peak_sequences:
        return {f"M{idx:03d}": empty_prior.copy() for idx in range(1, len(modules) + 1)}, empty_triplets

    peak_name_to_idx = {str(name): idx for idx, name in enumerate(peak_names.tolist())}
    peak_medians = {
        peak_name: float(np.median(peak_values[:, peak_idx]))
        for peak_name, peak_idx in peak_name_to_idx.items()
    }
    peak_hits = {
        peak_name: scan_peak_sequence_for_tf_hits(peak_sequences.get(peak_name, ""), motif_records_by_symbol, threshold_rel=motif_rel_threshold)
        for peak_name in relevant_peak_names
        if peak_name in peak_sequences
    }

    triplet_rows: List[Dict[str, object]] = []
    module_priors: Dict[str, pd.DataFrame] = {}
    gene_id_lookup = {str(name): str(gene_id) for name, gene_id in zip(gene_names.tolist(), gene_ids.tolist())}
    gene_symbol_lookup = {str(name): str(symbol) for name, symbol in zip(gene_names.tolist(), gene_symbols.tolist())}
    for module_idx, module in enumerate(modules, start=1):
        module_id = f"M{module_idx:03d}"
        module_columns = [str(gene_names[idx]) for idx in np.asarray(module, dtype=np.int64).tolist()]
        symbol_to_columns: Dict[str, List[str]] = {}
        for module_column in module_columns:
            symbol_to_columns.setdefault(canonical_gene_token(gene_symbol_lookup.get(module_column, module_column)), []).append(module_column)
        module_links = gene_peak_links[gene_peak_links["gene_module_column"].astype(str).isin(module_columns)].copy()
        prior_rows: List[Dict[str, object]] = []
        if not module_links.empty:
            module_links["peak_accessibility_score"] = minmax_normalize(
                [peak_medians.get(str(name), 0.0) for name in module_links["peak_name"].astype(str).tolist()],
                fill_value=1.0,
            )
            module_links["peak_gene_corr_score"] = 0.0
            for _, gene_rows in module_links.groupby("gene_module_column", sort=False):
                normalized = minmax_normalize(gene_rows["peak_gene_corr"].astype(float).tolist(), fill_value=1.0)
                module_links.loc[gene_rows.index, "peak_gene_corr_score"] = normalized
            best_by_pair: Dict[Tuple[str, str], Dict[str, object]] = {}
            for link_row in module_links.itertuples(index=False):
                gene_column = str(link_row.gene_module_column)
                gene_symbol = str(link_row.gene_symbol)
                peak_name = str(link_row.peak_name)
                motif_hit_by_tf = peak_hits.get(peak_name, {})
                if not motif_hit_by_tf:
                    continue
                for tf_symbol_token, hit in motif_hit_by_tf.items():
                    tf_columns = symbol_to_columns.get(str(tf_symbol_token), [])
                    if not tf_columns:
                        continue
                    for tf_column in tf_columns:
                        if tf_column == gene_column:
                            continue
                        tf_symbol = gene_symbol_lookup.get(tf_column, tf_column)
                        triplet_score = float(hit.rel_score) * float(link_row.peak_accessibility_score) * float(link_row.peak_gene_corr_score)
                        if triplet_score <= 0:
                            continue
                        row = {
                            "module_id": module_id,
                            "tf_module_column": str(tf_column),
                            "tf_gene_id": gene_id_lookup.get(str(tf_column), str(tf_column)),
                            "tf_symbol": str(tf_symbol),
                            "gene_module_column": gene_column,
                            "gene_gene_id": gene_id_lookup.get(gene_column, gene_column),
                            "gene_symbol": gene_symbol,
                            "peak_name": peak_name,
                            "peak_interval": str(getattr(link_row, "peak_interval", "")),
                            "peak_accessibility_score": float(link_row.peak_accessibility_score),
                            "peak_gene_corr_score": float(link_row.peak_gene_corr_score),
                            "motif_rel_score": float(hit.rel_score),
                            "triplet_score": float(triplet_score),
                            "motif_id": str(hit.motif_id),
                            "motif_name": str(hit.motif_name),
                            "motif_strand": str(hit.strand),
                            "motif_position": int(hit.position),
                        }
                        triplet_rows.append(row)
                        pair_key = (str(tf_column), gene_column)
                        if pair_key not in best_by_pair or row["triplet_score"] > best_by_pair[pair_key]["tf_target_score"]:
                            best_by_pair[pair_key] = {
                                "module_id": module_id,
                                "tf_module_column": str(tf_column),
                                "tf_gene_id": gene_id_lookup.get(str(tf_column), str(tf_column)),
                                "tf_symbol": str(tf_symbol),
                                "gene_module_column": gene_column,
                                "gene_gene_id": gene_id_lookup.get(gene_column, gene_column),
                                "gene_symbol": gene_symbol,
                                "tf_target_score": float(row["triplet_score"]),
                                "supporting_peak_name": peak_name,
                                "supporting_peak_interval": row["peak_interval"],
                                "peak_accessibility_score": float(row["peak_accessibility_score"]),
                                "peak_gene_corr_score": float(row["peak_gene_corr_score"]),
                                "motif_rel_score": float(row["motif_rel_score"]),
                                "motif_id": str(row["motif_id"]),
                                "motif_name": str(row["motif_name"]),
                                "motif_strand": str(row["motif_strand"]),
                                "motif_position": int(row["motif_position"]),
                            }
            prior_rows = list(best_by_pair.values())
        module_priors[module_id] = pd.DataFrame(prior_rows, columns=empty_prior.columns)

    triplets_df = pd.DataFrame(triplet_rows, columns=empty_triplets.columns)
    return module_priors, triplets_df


def export_outputs(
    outdir: Path,
    prefix: str,
    barcodes: np.ndarray,
    unsmoothed_log_rna: np.ndarray,
    gene_names: np.ndarray,
    gene_ids: np.ndarray,
    gene_symbols: np.ndarray,
    modules: Sequence[np.ndarray],
    gene_peak_links: pd.DataFrame,
    module_backend: str,
    input_modality: str,
    analysis_mode: str,
    manifest: Dict[str, object],
    sensitivity_report: Sequence[Dict[str, object]],
    module_pair_priors: Optional[Dict[str, pd.DataFrame]] = None,
    module_tf_target_priors: Optional[Dict[str, pd.DataFrame]] = None,
    tf_peak_gene_triplets: Optional[pd.DataFrame] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    mapping_rows: List[Dict[str, object]] = []
    module_rows: List[Dict[str, object]] = []
    for module_idx, module in enumerate(modules, start=1):
        module_id = f"M{module_idx:03d}"
        module_names = gene_names[module]
        module_ids = gene_ids[module]
        module_df = pd.DataFrame(unsmoothed_log_rna[:, module], index=barcodes, columns=module_names)
        module_df.index.name = "barcode"
        module_path = outdir / f"module_{module_id}_expression.csv"
        module_df.to_csv(module_path)
        if module_pair_priors is not None and module_id in module_pair_priors:
            module_pair_priors[module_id].to_csv(outdir / f"{module_path.stem}_allowed_pairs.csv", index=False)
        if module_tf_target_priors is not None and module_id in module_tf_target_priors:
            module_tf_target_priors[module_id].to_csv(outdir / f"{module_path.stem}_tf_target_prior.csv", index=False)
        module_rows.append({
            "module_id": module_id,
            "gene_count": int(module.size),
            "input_modality": input_modality,
            "analysis_mode": analysis_mode,
            "module_backend": module_backend,
            "cscn_input": "unsmoothed_log1p_rna",
            "module_construction_view": "log1p_rna",
        })
        for order_in_module, (gene_name, gene_id) in enumerate(zip(module_names, module_ids), start=1):
            mapping_rows.append({
                "gene": gene_name,
                "gene_id": gene_id,
                "module_id": module_id,
                "order_in_module": order_in_module,
                "input_modality": input_modality,
                "analysis_mode": analysis_mode,
            })
    pd.DataFrame(mapping_rows).to_csv(outdir / "gene_module_mapping.csv", index=False)
    build_gene_alias_rows(modules=modules, gene_names=gene_names, gene_ids=gene_ids, gene_symbols=gene_symbols).to_csv(
        outdir / "gene_alias_mapping.csv",
        index=False,
    )
    pd.DataFrame(module_rows).to_csv(outdir / "module_metadata.csv", index=False)
    gene_peak_links.to_csv(outdir / "gene_peak_links.csv", index=False)
    if tf_peak_gene_triplets is not None:
        tf_peak_gene_triplets.to_csv(outdir / "tf_peak_gene_triplets.csv", index=False)
    with (outdir / "sensitivity_report.json").open("w", encoding="utf-8") as handle:
        json.dump(list(sensitivity_report), handle, indent=2)
    with (outdir / "preprocess_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    if analysis_mode == "representation":
        representation_df = pd.DataFrame(unsmoothed_log_rna, index=barcodes, columns=gene_names)
        representation_df.index.name = "barcode"
        representation_df.to_csv(outdir / f"{prefix}_representation_input.csv")


def prepare_scrna(args) -> Dict[str, object]:
    if args.module_backend != "wgcna":
        raise ValueError("single scRNA currently supports only module_backend='wgcna'")
    df = pd.read_csv(args.input_csv, index_col=0)
    barcodes = df.index.astype(str).to_numpy()
    gene_names = np.array(df.columns.astype(str))
    gene_ids = gene_names.copy()
    gene_symbols = gene_names.copy()
    log_rna, normalization_mode = auto_log1p_dense(df)
    modules, _, _ = build_wgcna_modules(
        expression=log_rna,
        soft_power=args.soft_power,
        cut_height=args.cut_height,
        module_min=args.module_min_size,
        module_max=args.module_max_size,
    )
    manifest = {
        "dataset_prefix": args.prefix,
        "input_modality": "scrna",
        "analysis_mode": args.analysis_mode,
        "module_backend": args.module_backend,
        "source": str(args.input_csv),
        "normalization_mode": normalization_mode,
        "allow_nmf_after_preprocess": args.analysis_mode == "representation",
        "module_min_size": args.module_min_size,
        "module_max_size": args.module_max_size,
        "selected_shape_cells_by_genes": [int(log_rna.shape[0]), int(log_rna.shape[1])],
        "soft_power": args.soft_power,
        "cut_height": args.cut_height,
        "modules": summarize_modules(modules),
    }
    export_outputs(
        outdir=args.outdir,
        prefix=args.prefix,
        barcodes=barcodes,
        unsmoothed_log_rna=log_rna,
        gene_names=gene_names,
        gene_ids=gene_ids,
        gene_symbols=gene_symbols,
        modules=modules,
        gene_peak_links=pd.DataFrame(
            columns=[
                "gene_module_column",
                "gene_gene_id",
                "gene_symbol",
                "peak_name",
                "peak_interval",
                "peak_gene_corr",
                "distance_to_tss",
                "peak_accessibility_median",
                "association_pass",
                "anchor_source",
            ]
        ),
        module_backend=args.module_backend,
        input_modality="scrna",
        analysis_mode=args.analysis_mode,
        manifest=manifest,
        sensitivity_report=[{"setting": "wgcna_default", **summarize_modules(modules)}],
        module_pair_priors=None,
        module_tf_target_priors=None,
        tf_peak_gene_triplets=None,
    )
    return manifest


def prepare_multiome(args) -> Dict[str, object]:
    if args.module_backend != "multiome_refined_wgcna":
        raise ValueError("paired_multiome currently supports only module_backend='multiome_refined_wgcna'")
    feature_by_barcode, feature_table, barcodes = load_10x_h5(args.input_h5)
    gex_matrix, gene_names_all, gene_ids_all, gene_intervals_all = subset_feature_type(feature_by_barcode, feature_table, "Gene Expression")
    peak_matrix, peak_names_all, peak_ids_all, peak_intervals_all = subset_feature_type(feature_by_barcode, feature_table, "Peaks")
    log_gex_all, cell_sums = log_normalize_sparse(gex_matrix)
    selected_rna_sparse, selected_gene_names, selected_gene_ids, selected_gene_intervals = select_top_variable_sparse(
        log_gex_all,
        gene_names_all,
        gene_ids_all,
        gene_intervals_all,
        args.top_genes,
        args.min_gene_cells,
    )
    [selected_rna_sparse, peak_matrix], barcodes = select_top_cells(
        [selected_rna_sparse, peak_matrix],
        barcodes,
        cell_sums,
        args.top_cells,
    )
    gtf_metadata = load_gtf_metadata(args.gtf)
    unsmoothed_log_rna = selected_rna_sparse.toarray().astype(np.float32, copy=False)
    unique_gene_names = np.array(make_unique(selected_gene_names.tolist(), selected_gene_ids.tolist()))
    gene_symbols = resolve_gene_symbols(selected_gene_names, selected_gene_ids, gtf_metadata.alias_to_symbol)
    top_peak_sparse, top_peak_names, top_peak_ids, top_peak_intervals = select_top_variable_sparse(
        peak_matrix,
        peak_names_all,
        peak_ids_all,
        peak_intervals_all,
        args.top_peaks,
        args.min_peak_cells,
    )
    peak_values = top_peak_sparse.toarray().astype(np.float32, copy=False)
    peak_intervals_parsed = [parse_interval(text) for text in top_peak_intervals]
    gene_anchors, anchor_sources = build_gene_anchors(selected_gene_names, selected_gene_ids, selected_gene_intervals, args.gtf)
    gene_peak_links = build_peak_gene_links(
        expression=unsmoothed_log_rna,
        peak_values=peak_values,
        gene_names=unique_gene_names,
        gene_ids=selected_gene_ids,
        gene_symbols=gene_symbols,
        gene_anchors=gene_anchors,
        anchor_sources=anchor_sources,
        peak_names=top_peak_names,
        peak_intervals=peak_intervals_parsed,
        peak_gene_window=args.peak_gene_window,
        corr_threshold=args.peak_gene_corr_threshold,
    )
    modules, global_adj, _ = build_wgcna_modules(
        expression=unsmoothed_log_rna,
        soft_power=args.soft_power,
        cut_height=args.cut_height,
        module_min=args.module_min_size,
        module_max=args.module_max_size,
    )
    module_pair_priors = None
    module_tf_target_priors: Optional[Dict[str, pd.DataFrame]] = None
    tf_peak_gene_triplets: Optional[pd.DataFrame] = None
    if args.enable_tf_motif_prior and len(top_peak_names):
        module_tf_target_priors, tf_peak_gene_triplets = build_tf_motif_target_priors(
            modules=modules,
            gene_names=unique_gene_names,
            gene_ids=selected_gene_ids,
            gene_symbols=gene_symbols,
            gene_peak_links=gene_peak_links,
            peak_names=top_peak_names,
            peak_intervals=peak_intervals_parsed,
            peak_values=peak_values,
            tf_list_path=args.tf_list_path,
            genome_fasta=args.genome_fasta,
            outdir=args.outdir,
            gtf_alias_to_symbol=gtf_metadata.alias_to_symbol,
        )
    anchor_source_counts: Dict[str, int] = {}
    for source in anchor_sources:
        anchor_source_counts[source] = anchor_source_counts.get(source, 0) + 1
    manifest = {
        "dataset_prefix": args.prefix,
        "input_modality": "paired_multiome",
        "analysis_mode": args.analysis_mode,
        "module_backend": args.module_backend,
        "source_h5": str(args.input_h5),
        "gtf": None if args.gtf is None else str(args.gtf),
        "allow_nmf_after_preprocess": args.analysis_mode == "representation",
        "final_cscn_input": "unsmoothed_log1p_rna",
        "module_min_size": args.module_min_size,
        "module_max_size": args.module_max_size,
        "soft_power": args.soft_power,
        "cut_height": args.cut_height,
        "peak_gene_window": int(args.peak_gene_window),
        "peak_gene_corr_threshold": float(args.peak_gene_corr_threshold),
        "selected_shape_cells_by_genes": [int(unsmoothed_log_rna.shape[0]), int(unsmoothed_log_rna.shape[1])],
        "top_peaks_selected": int(top_peak_sparse.shape[1]),
        "anchor_source_counts": anchor_source_counts,
        "modules": summarize_modules(modules),
        "peak_gene_link_count": int(len(gene_peak_links)),
        "enable_tf_motif_prior": bool(args.enable_tf_motif_prior),
        "jaspar_release": JASPAR_RELEASE if args.enable_tf_motif_prior else None,
        "tf_motif_triplet_count": int(0 if tf_peak_gene_triplets is None else len(tf_peak_gene_triplets)),
        "tf_motif_target_pair_count": int(0 if module_tf_target_priors is None else sum(len(df) for df in module_tf_target_priors.values())),
        "tf_motif_symbol_count": int(
            0
            if module_tf_target_priors is None
            else len({str(row.tf_symbol) for df in module_tf_target_priors.values() for row in df.itertuples(index=False)})
        ),
    }
    export_outputs(
        outdir=args.outdir,
        prefix=args.prefix,
        barcodes=barcodes,
        unsmoothed_log_rna=unsmoothed_log_rna,
        gene_names=unique_gene_names,
        gene_ids=selected_gene_ids,
        gene_symbols=gene_symbols,
        modules=modules,
        gene_peak_links=gene_peak_links,
        module_backend=args.module_backend,
        input_modality="paired_multiome",
        analysis_mode=args.analysis_mode,
        manifest=manifest,
        sensitivity_report=[{"setting": "wgcna_default", **summarize_modules(modules)}],
        module_pair_priors=module_pair_priors,
        module_tf_target_priors=module_tf_target_priors,
        tf_peak_gene_triplets=tf_peak_gene_triplets,
    )
    return manifest


from dataclasses import dataclass
from types import SimpleNamespace

from .config import PreprocessConfig


@dataclass(frozen=True)
class PreprocessResult:
    output_dir: Path
    manifest: Dict[str, object]


def build_modules(config: PreprocessConfig, outdir: Path) -> PreprocessResult:
    config = config.validate()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(**config.__dict__, outdir=outdir)
    if config.input_modality == 'scrna':
        manifest = prepare_scrna(args)
    else:
        manifest = prepare_multiome(args)
    return PreprocessResult(output_dir=outdir, manifest=manifest)
