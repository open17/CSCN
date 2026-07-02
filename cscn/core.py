from __future__ import annotations

from pathlib import Path

import os
import random
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from bitarray import bitarray, frozenbitarray
from scipy.stats import norm
from sklearn.decomposition import NMF

UNBOUNDED_RANGE = (1, 0)
QUERY_ENGINES = {"bitmap", "hybrid", "kdt_debug"}
CELL_BACKENDS = {"serial", "process"}
TF_WEIGHTED_PRIOR_MODES = {
    "atac_prior_cscn",
}
TF_SKELETON_PRIOR_MODES = {
    "atac_prior_cscn",
}
ATAC_CI_MODES = {"none", "joint_rna_atac_conditioned_cscn"}
ATAC_CI_PROFILE_MODES = {"max", "weighted_sum"}


@dataclass(frozen=True)
class ExpertKnowledge:
    forbidden_edges: Tuple[Tuple[object, object], ...] = ()
    required_edges: Tuple[Tuple[object, object], ...] = ()
    temporal_ordering: Mapping[object, int] = None
    temporal_order: Tuple[Tuple[object, ...], ...] = ((),)

    def __init__(
        self,
        forbidden_edges: Optional[Iterable[Tuple[object, object]]] = None,
        required_edges: Optional[Iterable[Tuple[object, object]]] = None,
        temporal_ordering: Optional[Mapping[object, int]] = None,
        temporal_order: Optional[Sequence[Sequence[object]]] = None,
    ):
        object.__setattr__(self, "forbidden_edges", tuple(forbidden_edges or ()))
        object.__setattr__(self, "required_edges", tuple(required_edges or ()))
        object.__setattr__(self, "temporal_ordering", dict(temporal_ordering or {}))
        object.__setattr__(self, "temporal_order", tuple(tuple(group) for group in (temporal_order or ((),))))


class KDT_Node:
    __slots__ = ("point", "cell_id", "left", "right", "low_bounds", "up_bounds", "size")

    def __init__(self, point, cell_id, left, right):
        self.point = point
        self.cell_id = cell_id
        self.left = left
        self.right = right

    def push_up(self):
        self.low_bounds = self.point.copy()
        self.up_bounds = self.point.copy()
        self.size = 1
        for child in (self.left, self.right):
            if child is None:
                continue
            self.size += child.size
            for axis in range(len(self.point)):
                self.low_bounds[axis] = min(self.low_bounds[axis], child.low_bounds[axis])
                self.up_bounds[axis] = max(self.up_bounds[axis], child.up_bounds[axis])


def qnth_element(records, left, right, k, axis):
    if left == right:
        return records[left]

    pivot_idx = random.randint(left, right)
    pivot_value = records[pivot_idx][0][axis]
    i = left - 1
    j = right + 1
    while i < j:
        i += 1
        j -= 1
        while i <= right and records[i][0][axis] < pivot_value:
            i += 1
        while j >= left and records[j][0][axis] > pivot_value:
            j -= 1
        if i < j:
            records[i], records[j] = records[j], records[i]
    if k <= j:
        return qnth_element(records, left, j, k, axis)
    return qnth_element(records, j + 1, right, k, axis)


class KDT:
    __slots__ = ("points", "K", "root")

    def __init__(self, points, cell_ids=None):
        if len(points) == 0:
            raise ValueError("KD-tree requires at least one point")
        if cell_ids is None:
            cell_ids = range(len(points))
        self.points = [(list(point), int(cell_id)) for point, cell_id in zip(points, cell_ids)]
        self.K = len(self.points[0][0])
        self.root = self.build(0, len(self.points) - 1, 0)

    def build(self, left, right, axis):
        if left > right:
            return None
        axis %= self.K
        idx = left + (right - left + 1) // 2
        point, cell_id = qnth_element(self.points, left, right, idx, axis)
        node = KDT_Node(
            point,
            cell_id,
            self.build(left, idx - 1, axis + 1),
            self.build(idx + 1, right, axis + 1),
        )
        node.push_up()
        return node

    def _subtree_fully_inside(self, node, query_ranges):
        for axis in range(self.K):
            low, up = query_ranges[axis]
            if up < low:
                continue
            if not (low <= node.low_bounds[axis] and up >= node.up_bounds[axis]):
                return False
        return True

    def _subtree_disjoint(self, node, query_ranges):
        for axis in range(self.K):
            low, up = query_ranges[axis]
            if up < low:
                continue
            if node.low_bounds[axis] > up or node.up_bounds[axis] < low:
                return True
        return False

    def _point_inside(self, point, query_ranges):
        for axis in range(self.K):
            low, up = query_ranges[axis]
            if up < low:
                continue
            if point[axis] < low or point[axis] > up:
                return False
        return True

    def _collect_subtree_ids(self, node, results, limit):
        if node is None:
            return
        if limit is not None and len(results) >= limit:
            return
        results.append(node.cell_id)
        self._collect_subtree_ids(node.left, results, limit)
        self._collect_subtree_ids(node.right, results, limit)

    def _query_count(self, node, query_ranges):
        if node is None:
            return 0
        if self._subtree_fully_inside(node, query_ranges):
            return node.size
        if self._subtree_disjoint(node, query_ranges):
            return 0
        count = 1 if self._point_inside(node.point, query_ranges) else 0
        count += self._query_count(node.left, query_ranges)
        count += self._query_count(node.right, query_ranges)
        return count

    def _query_ids(self, node, query_ranges, results, limit):
        if node is None:
            return
        if limit is not None and len(results) >= limit:
            return
        if self._subtree_disjoint(node, query_ranges):
            return
        if self._subtree_fully_inside(node, query_ranges):
            self._collect_subtree_ids(node, results, limit)
            return
        if self._point_inside(node.point, query_ranges):
            results.append(node.cell_id)
            if limit is not None and len(results) >= limit:
                return
        self._query_ids(node.left, query_ranges, results, limit)
        self._query_ids(node.right, query_ranges, results, limit)

    def query_cnt(self, query_ranges):
        return self._query_count(self.root, query_ranges)

    def query_ids(self, query_ranges, limit=None):
        results = []
        self._query_ids(self.root, query_ranges, results, limit)
        return results


_PC_WORKER = None


def _current_cpu_affinity() -> Tuple[int, ...]:
    try:
        return tuple(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return tuple(range(os.cpu_count() or 1))


def _restore_worker_affinity(cscn_instance) -> None:
    target = getattr(cscn_instance, "_parent_cpu_affinity", None)
    if not target:
        return
    try:
        os.sched_setaffinity(0, set(int(cpu) for cpu in target))
    except (AttributeError, OSError, ValueError):
        return


def _init_pc_worker(cscn_instance):
    global _PC_WORKER
    _restore_worker_affinity(cscn_instance)
    _PC_WORKER = cscn_instance


def _run_pc_worker_batch(task_ids):
    if _PC_WORKER is None:
        raise RuntimeError("PC worker is not initialized")
    return [_PC_WORKER.run_pc_for_task(task_id) for task_id in task_ids]


class CSCN:
    def __init__(
        self,
        sigmoid_score=0.1,
        pc_var="stable",
        significance_level=0.01,
        max_cond_vars=20,
        use_bitmap=None,
        debug=False,
        query_engine="hybrid",
        hybrid_subspace_dim=6,
        hybrid_subspace_stride=4,
        hybrid_min_overlap=2,
        hybrid_candidate_limit=None,
        hybrid_fallback_ratio=0.15,
        enable_query_stats=False,
        max_bits_cache_entries=8192,
        tf_prior_mode="none",
        allowed_undirected_pairs=None,
        allowed_undirected_pair_weights=None,
        external_prior_mode="hard",
        external_prior_alpha=0.20,
        external_prior_min_strength=0.0,
        tf_list_path=None,
        tf_top_k=5,
        tf_skeleton_alpha=0.20,
        tf_skeleton_min_strength=0.0,
        tf_skeleton_max_conditioning_vars=-1,
        tf_skeleton_strength_gamma=1.0,
        atac_ci_mode="none",
        atac_ci_open_threshold=0.0,
        atac_ci_profile_mode="max",
        variable_names=None,
    ):
        if query_engine not in QUERY_ENGINES:
            raise ValueError(f"Unsupported query_engine: {query_engine}")
        if atac_ci_mode not in ATAC_CI_MODES:
            raise ValueError(f"Unsupported atac_ci_mode: {atac_ci_mode}")
        if atac_ci_profile_mode not in ATAC_CI_PROFILE_MODES:
            raise ValueError(f"Unsupported atac_ci_profile_mode: {atac_ci_profile_mode}")
        if use_bitmap is not None:
            query_engine = "bitmap" if use_bitmap else "kdt_debug"
        self.sigmoid_score = sigmoid_score
        self.significance_level = significance_level
        self.max_cond_vars = max_cond_vars
        self.pc_var = pc_var
        self.query_engine = query_engine
        self.use_bitmap = query_engine == "bitmap"
        self.debug = debug
        self.hybrid_subspace_dim = max(1, int(hybrid_subspace_dim))
        self.hybrid_subspace_stride = max(1, int(hybrid_subspace_stride))
        self.hybrid_min_overlap = max(1, int(hybrid_min_overlap))
        self.hybrid_candidate_limit = hybrid_candidate_limit
        self.hybrid_fallback_ratio = float(hybrid_fallback_ratio)
        self.enable_query_stats = enable_query_stats
        self.max_bits_cache_entries = int(max_bits_cache_entries)
        self.ran_cache = {}
        self.bits_cache = OrderedDict()
        self.count_cache = {}
        self.ci_cache = {}
        self.hybrid_candidate_cache = {}
        self.query_stats = self._new_query_stats()
        self.data = None
        self.df = None
        self.kdtree = None
        self.loadings = None
        self.n_cells = 0
        self.n_genes = 0
        self.sorted_indices = None
        self.sorted_expressions = None
        self.sorted_positions = None
        self.sorted_nonpositive_counts = None
        self.hybrid_candidate_limit_value = None
        self.gene_rank_by_variance = None
        self.subspace_trees = []
        self.gene_to_subspaces = {}
        self.pc_estimator = None
        self.use_nmf_applied = False
        self.nmf_components_used = None
        self.tf_prior_mode = str(tf_prior_mode)
        self.tf_list_path = None if tf_list_path is None else str(tf_list_path)
        self.tf_top_k = max(1, int(tf_top_k))
        self.tf_skeleton_alpha = float(tf_skeleton_alpha)
        self.tf_skeleton_min_strength = float(tf_skeleton_min_strength)
        self.tf_skeleton_max_conditioning_vars = int(tf_skeleton_max_conditioning_vars)
        self.tf_skeleton_strength_gamma = float(tf_skeleton_strength_gamma)
        self.atac_ci_mode = str(atac_ci_mode)
        self.atac_ci_open_threshold = float(atac_ci_open_threshold)
        self.atac_ci_profile_mode = str(atac_ci_profile_mode)
        self.variable_names = None if variable_names is None else [str(name) for name in variable_names]
        self._parent_cpu_affinity = _current_cpu_affinity()
        self.extra_forbidden_edges: Set[Tuple[str, str]] = set()
        self.external_allowed_pairs_provided = (
            allowed_undirected_pairs is not None or allowed_undirected_pair_weights is not None
        )
        self.external_prior_mode = str(external_prior_mode)
        self.external_prior_alpha = float(external_prior_alpha)
        self.external_prior_min_strength = float(external_prior_min_strength)
        self.external_prior_weight_map: Dict[frozenset[str], float] = {}
        self.tf_local_candidate_pairs: Set[Tuple[str, str]] = set()
        self.tf_direct_supported_pairs: Set[Tuple[str, str]] = set()
        self.tf_prior_weight_map: Dict[Tuple[str, str], float] = {}
        self.atac_ci_peak_access_by_name: Dict[str, np.ndarray] = {}
        self.atac_ci_target_peaks_by_id: Dict[int, List[str]] = {}
        self.atac_ci_target_peak_weights_by_id: Dict[int, Dict[str, float]] = {}
        self.atac_ci_profiles_by_id: Dict[int, np.ndarray] = {}
        self.atac_ci_sorted_indices_by_id: Dict[int, np.ndarray] = {}
        self.atac_ci_sorted_values_by_id: Dict[int, np.ndarray] = {}
        self.atac_ci_sorted_positions_by_id: Dict[int, np.ndarray] = {}
        self.atac_ci_nonpositive_counts_by_id: Dict[int, int] = {}
        self.atac_ci_joint_bits_cache: OrderedDict[Tuple[object, ...], frozenbitarray] = OrderedDict()
        self.atac_ci_source_h5 = None
        self.atac_ci_peak_count = 0
        self.atac_ci_missing_peak_count = 0
        self.atac_ci_target_count = 0
        self.atac_ci_profile_mode_used = str(atac_ci_profile_mode)
        self._reset_atac_ci_stats()
        self._reset_task_atac_ci_stats()
        self.tf_local_candidate_edge_count = 0
        self.tf_local_target_count = 0
        self.tf_direct_edge_count = 0
        self.tf_weighted_prior_pair_count = 0
        self.tf_weighted_prior_weight_total = 0.0
        self._reset_tf_weighted_orientation_stats()
        self._reset_tf_skeleton_stats()
        self._reset_task_tf_skeleton_stats()
        self.external_allowed_pairs: Set[frozenset[str]] = set()
        if allowed_undirected_pair_weights is not None:
            self.external_prior_weight_map = {
                frozenset(str(node) for node in pair): float(np.clip(weight, 0.0, 1.0))
                for pair, weight in allowed_undirected_pair_weights.items()
                if len(pair) == 2 and float(weight) > 0.0
            }
            if float(self.external_prior_min_strength) > 0.0:
                self.external_prior_weight_map = {
                    pair: weight
                    for pair, weight in self.external_prior_weight_map.items()
                    if float(weight) >= float(self.external_prior_min_strength)
                }
            self.external_allowed_pairs = set(self.external_prior_weight_map)
        elif allowed_undirected_pairs is not None:
            self.external_allowed_pairs = {
                frozenset(str(node) for node in pair)
                for pair in allowed_undirected_pairs
                if len(pair) == 2
            }
            self.external_prior_weight_map = {pair: 1.0 for pair in self.external_allowed_pairs}
        self.combined_expert_knowledge = None
        self.tf_prior_active_tf_count = 0
        self.tf_prior_allowed_edge_count = 0
        self.external_prior_allowed_edge_count = len(self.external_allowed_pairs)
        self.combined_prior_allowed_edge_count = 0
        self.variable_name_to_idx: Dict[str, int] = {}

    def _new_query_stats(self):
        return {
            "count_queries": 0,
            "count_cache_hits": 0,
            "ci_queries": 0,
            "ci_cache_hits": 0,
            "bit_cache_hits": 0,
            "range_cache_hits": 0,
            "bits_cache_evictions": 0,
            "hybrid_queries": 0,
            "hybrid_bitmap_shortcut": 0,
            "hybrid_no_subspace": 0,
            "hybrid_low_overlap": 0,
            "hybrid_candidate_queries": 0,
            "hybrid_candidate_cache_hits": 0,
            "hybrid_candidate_accepted": 0,
            "hybrid_candidate_overflow": 0,
            "hybrid_candidate_total_size": 0,
            "hybrid_candidate_max_size": 0,
        }

    def _bump_stat(self, key, value=1):
        if self.enable_query_stats:
            self.query_stats[key] += value

    def reset_query_stats(self):
        self.query_stats = self._new_query_stats()

    def get_query_stats(self):
        stats = dict(self.query_stats)
        accepted = stats["hybrid_candidate_accepted"]
        stats["hybrid_candidate_avg_size"] = stats["hybrid_candidate_total_size"] / accepted if accepted else 0.0
        stats["count_cache_hit_rate"] = stats["count_cache_hits"] / stats["count_queries"] if stats["count_queries"] else 0.0
        stats["ci_cache_hit_rate"] = stats["ci_cache_hits"] / stats["ci_queries"] if stats["ci_queries"] else 0.0
        stats["hybrid_candidate_cache_hit_rate"] = stats["hybrid_candidate_cache_hits"] / stats["hybrid_candidate_queries"] if stats["hybrid_candidate_queries"] else 0.0
        stats["hybrid_fallbacks"] = stats["hybrid_bitmap_shortcut"] + stats["hybrid_no_subspace"] + stats["hybrid_low_overlap"] + stats["hybrid_candidate_overflow"]
        stats["hybrid_success_rate"] = accepted / stats["hybrid_queries"] if stats["hybrid_queries"] else 0.0
        return stats

    def _reset_tf_weighted_orientation_stats(self):
        self.tf_weighted_candidate_edge_count = 0
        self.tf_weighted_oriented_edge_count = 0
        self.tf_weighted_satisfied_weight = 0.0
        self.tf_weighted_available_weight = 0.0
        self.tf_weighted_satisfaction_rate = 0.0
        self.tf_weighted_reversed_pc_edge_count = 0
        self.tf_weighted_cycle_avoidance_flip_count = 0
        self.tf_weighted_forced_cycle_edge_count = 0
        self.tf_weighted_prior_conflict_edge_count = 0

    def _reset_tf_skeleton_stats(self):
        self.tf_skeleton_prior_pair_count = 0
        self.tf_skeleton_ci_query_count = 0
        self.tf_skeleton_rescued_ci_count = 0
        self.external_skeleton_ci_query_count = 0
        self.external_skeleton_rescued_ci_count = 0

    def _reset_atac_ci_stats(self):
        self.atac_ci_gate_query_count = 0
        self.atac_ci_gate_applied_count = 0
        self.atac_ci_gate_zero_key_count = 0
        self.atac_ci_gate_total_cells = 0
        self.atac_ci_joint_query_count = 0
        self.atac_ci_joint_gene_applied_count = 0

    def _reset_task_tf_skeleton_stats(self):
        self._task_tf_skeleton_ci_query_count = 0
        self._task_tf_skeleton_rescued_ci_count = 0
        self._task_external_skeleton_ci_query_count = 0
        self._task_external_skeleton_rescued_ci_count = 0

    def _reset_task_atac_ci_stats(self):
        self._task_atac_ci_gate_query_count = 0
        self._task_atac_ci_gate_applied_count = 0
        self._task_atac_ci_gate_zero_key_count = 0
        self._task_atac_ci_gate_total_cells = 0
        self._task_atac_ci_joint_query_count = 0
        self._task_atac_ci_joint_gene_applied_count = 0

    def clear_cache(self):
        self.clear_stable_cache()
        self.clear_task_cache()

    def clear_stable_cache(self):
        self.ran_cache.clear()
        self.bits_cache.clear()
        self.atac_ci_joint_bits_cache.clear()

    def clear_task_cache(self):
        self.count_cache.clear()
        self.ci_cache.clear()
        self.hybrid_candidate_cache.clear()
        self._reset_task_tf_skeleton_stats()
        self._reset_task_atac_ci_stats()

    def _coerce_gene_id(self, gene) -> int:
        if isinstance(gene, (int, np.integer)):
            return int(gene)
        gene_text = str(gene)
        if gene_text in self.variable_name_to_idx:
            return int(self.variable_name_to_idx[gene_text])
        return int(gene)

    def _normalize_gene_key(self, genes: Iterable[int]) -> Tuple[int, ...]:
        return tuple(sorted(self._coerce_gene_id(gene) for gene in genes))

    def _prepare_sorted_views(self):
        self.n_cells, self.n_genes = self.data.shape
        self.sorted_indices = np.argsort(self.data, axis=0, kind="mergesort")
        self.sorted_expressions = np.take_along_axis(self.data, self.sorted_indices, axis=0)
        self.sorted_positions = np.empty_like(self.sorted_indices)
        row_order = np.arange(self.n_cells)[:, None]
        col_order = np.arange(self.n_genes)[None, :]
        self.sorted_positions[self.sorted_indices, col_order] = row_order
        self.sorted_nonpositive_counts = np.count_nonzero(self.sorted_expressions <= 0, axis=0)

    def _resolve_hybrid_candidate_limit(self):
        if self.hybrid_candidate_limit is None:
            return max(256, self.n_cells // 64)
        return int(self.hybrid_candidate_limit)

    def _indices_to_bitset(self, indices):
        bitset = bitarray(self.n_cells)
        bitset.setall(False)
        idx_arr = np.asarray(indices, dtype=np.int64)
        if idx_arr.size == 0:
            return frozenbitarray(bitset)
        bad_mask = (idx_arr < 0) | (idx_arr >= self.n_cells)
        if np.any(bad_mask):
            bad_idx = int(idx_arr[bad_mask][0])
            raise ValueError(f"Cell index {bad_idx} is out of range [0, {self.n_cells - 1}]")
        bitset[idx_arr.tolist()] = True
        return frozenbitarray(bitset)

    def _bitset_intersection_count(self, bitsets) -> int:
        if not bitsets:
            return self.n_cells
        result = bitarray(bitsets[0])
        for bitset in bitsets[1:]:
            result &= bitset
        return int(result.count())

    def _get_hybrid_threshold(self):
        ratio_limit = int(np.ceil(self.hybrid_fallback_ratio * self.n_cells))
        return max(self.hybrid_candidate_limit_value, ratio_limit)

    def _build_query_ranges(self, genes, key_cell_idx, sigmoid_score, total_dims, gene_to_axis):
        ranges = [UNBOUNDED_RANGE] * total_dims
        for gene_id in genes:
            axis = gene_to_axis[gene_id]
            ranges[axis] = self.get_ran_with_indices(gene_id, key_cell_idx, sigmoid_score)["expression_range"]
        return ranges

    def _get_tie_block(self, gene_id, key_cell_idx):
        sorted_expressions = self.sorted_expressions[:, gene_id]
        sorted_idx = int(self.sorted_positions[key_cell_idx, gene_id])
        current_expr = sorted_expressions[sorted_idx]
        tie_start = int(np.searchsorted(sorted_expressions, current_expr, side="left"))
        tie_stop = int(np.searchsorted(sorted_expressions, current_expr, side="right"))
        return tie_start, tie_stop

    def _neighborhood_cache_key(self, gene_id, key_cell_idx, sigmoid_score):
        tie_start, tie_stop = self._get_tie_block(gene_id, key_cell_idx)
        return int(gene_id), tie_start, tie_stop, float(sigmoid_score)

    def _build_subspace_windows(self):
        if self.n_genes == 0:
            return []
        window_size = min(self.hybrid_subspace_dim, self.n_genes)
        ranked_genes = self.gene_rank_by_variance
        if self.n_genes <= window_size:
            return [tuple(int(gene) for gene in ranked_genes)]
        windows = []
        start = 0
        while start + window_size <= self.n_genes:
            windows.append(tuple(int(gene) for gene in ranked_genes[start:start + window_size]))
            start += self.hybrid_subspace_stride
        tail_start = self.n_genes - window_size
        tail_window = tuple(int(gene) for gene in ranked_genes[tail_start:tail_start + window_size])
        if not windows or windows[-1] != tail_window:
            windows.append(tail_window)
        deduped = []
        seen = set()
        for window in windows:
            if window in seen:
                continue
            seen.add(window)
            deduped.append(window)
        return deduped

    def _prepare_subspace_trees(self):
        variances = np.var(self.data, axis=0)
        self.gene_rank_by_variance = np.argsort(variances)[::-1]
        self.subspace_trees = []
        self.gene_to_subspaces = {gene_id: [] for gene_id in range(self.n_genes)}
        for subspace_id, genes in enumerate(self._build_subspace_windows()):
            gene_to_axis = {gene_id: axis for axis, gene_id in enumerate(genes)}
            spec = {
                "id": subspace_id,
                "genes": genes,
                "gene_set": frozenset(genes),
                "gene_to_axis": gene_to_axis,
                "tree": KDT(self.data[:, genes].tolist(), cell_ids=np.arange(self.n_cells)),
            }
            self.subspace_trees.append(spec)
            for gene_id in genes:
                self.gene_to_subspaces[gene_id].append(subspace_id)

    def _select_hybrid_subspace(self, gene_key):
        overlap_counts = {}
        for gene_id in gene_key:
            for subspace_id in self.gene_to_subspaces.get(gene_id, ()): 
                overlap_counts[subspace_id] = overlap_counts.get(subspace_id, 0) + 1
        if not overlap_counts:
            return None, ()
        best_subspace_id = max(overlap_counts.items(), key=lambda item: (item[1], -item[0]))[0]
        spec = self.subspace_trees[best_subspace_id]
        overlap_genes = tuple(gene_id for gene_id in gene_key if gene_id in spec["gene_set"])
        return spec, overlap_genes

    def _get_hybrid_candidate(self, spec, seed_genes, key_cell_idx, sigmoid_score):
        cache_key = (spec["id"], key_cell_idx, seed_genes, sigmoid_score)
        if cache_key in self.hybrid_candidate_cache:
            self._bump_stat("hybrid_candidate_cache_hits")
            return self.hybrid_candidate_cache[cache_key]
        self._bump_stat("hybrid_candidate_queries")
        threshold = self._get_hybrid_threshold()
        limit = threshold + 1
        ranges = self._build_query_ranges(seed_genes, key_cell_idx, sigmoid_score, spec["tree"].K, spec["gene_to_axis"])
        candidate_ids = spec["tree"].query_ids(ranges, limit=limit)
        if len(candidate_ids) > threshold:
            self._bump_stat("hybrid_candidate_overflow")
            result = (None, len(candidate_ids), True)
        else:
            candidate_bitset = self._indices_to_bitset(candidate_ids)
            self._bump_stat("hybrid_candidate_accepted")
            self._bump_stat("hybrid_candidate_total_size", len(candidate_ids))
            if self.enable_query_stats:
                self.query_stats["hybrid_candidate_max_size"] = max(self.query_stats["hybrid_candidate_max_size"], len(candidate_ids))
            result = (candidate_bitset, len(candidate_ids), False)
        self.hybrid_candidate_cache[cache_key] = result
        return result

    def get_ran_with_indices(self, gene_id, key_cell_idx, sigmoid_score=None):
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        cache_key = self._neighborhood_cache_key(gene_id, key_cell_idx, sigmoid_score)
        if cache_key in self.ran_cache:
            self._bump_stat("range_cache_hits")
            return self.ran_cache[cache_key]
        sorted_indices = self.sorted_indices[:, gene_id]
        sorted_expressions = self.sorted_expressions[:, gene_id]
        window_size = int(sigmoid_score * self.n_cells)
        _, tie_start, tie_stop, _ = cache_key
        tie_size = tie_stop - tie_start
        if tie_size > window_size:
            lower_bound = tie_start
            upper_bound = tie_stop - 1
        else:
            # Mirror c-CSN's tie-block handling: expand from the tie block boundary
            # instead of the individual cell rank, and avoid falling back into a
            # large zero-valued block when a positive neighborhood is available.
            lower_floor = 0
            nonpositive_count = int(self.sorted_nonpositive_counts[gene_id])
            if nonpositive_count > window_size and nonpositive_count < self.n_cells:
                lower_floor = nonpositive_count
            lower_bound = max(lower_floor, tie_start - window_size)
            upper_bound = min(self.n_cells - 1, (tie_stop - 1) + window_size)
        low_expr = sorted_expressions[lower_bound]
        high_expr = sorted_expressions[upper_bound]
        value_lower = np.searchsorted(sorted_expressions, low_expr, side="left")
        value_upper = np.searchsorted(sorted_expressions, high_expr, side="right")
        result = {
            "expression_range": (low_expr, high_expr),
            "original_indices": sorted_indices[value_lower:value_upper],
        }
        self.ran_cache[cache_key] = result
        return result

    def get_kdt_counts(self, genes, key_cell_idx, sigmoid_score=None):
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        if not genes:
            return self.n_cells
        if self.kdtree is None:
            raise RuntimeError("Full KD-tree is only available in kdt_debug mode")
        gene_key = self._normalize_gene_key(genes)
        gene_to_axis = {gene_id: gene_id for gene_id in range(self.n_genes)}
        ranges = self._build_query_ranges(gene_key, key_cell_idx, sigmoid_score, self.n_genes, gene_to_axis)
        return self.kdtree.query_cnt(ranges)

    def get_bits(self, gene_id, key_cell_idx, sigmoid_score=None):
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        cache_key = self._neighborhood_cache_key(gene_id, key_cell_idx, sigmoid_score)
        if cache_key in self.bits_cache:
            self._bump_stat("bit_cache_hits")
            bitset = self.bits_cache.pop(cache_key)
            self.bits_cache[cache_key] = bitset
            return bitset
        original_indices = self.get_ran_with_indices(gene_id, key_cell_idx, sigmoid_score)["original_indices"]
        bitset = self._indices_to_bitset(original_indices)
        if self.max_bits_cache_entries != 0:
            self.bits_cache[cache_key] = bitset
            if self.max_bits_cache_entries > 0:
                while len(self.bits_cache) > self.max_bits_cache_entries:
                    self.bits_cache.popitem(last=False)
                    self._bump_stat("bits_cache_evictions")
        return bitset

    def get_bits_counts(self, genes, key_cell_idx, sigmoid_score=None):
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        gene_key = self._normalize_gene_key(genes)
        if not gene_key:
            return self.n_cells
        bitsets = [self.get_bits(gene_id, key_cell_idx, sigmoid_score) for gene_id in gene_key]
        return self._bitset_intersection_count(bitsets)

    def _atac_tie_block(self, gene_id: int, key_cell_idx: int) -> Tuple[int, int]:
        sorted_values = self.atac_ci_sorted_values_by_id[int(gene_id)]
        sorted_idx = int(self.atac_ci_sorted_positions_by_id[int(gene_id)][int(key_cell_idx)])
        current_value = sorted_values[sorted_idx]
        tie_start = int(np.searchsorted(sorted_values, current_value, side="left"))
        tie_stop = int(np.searchsorted(sorted_values, current_value, side="right"))
        return tie_start, tie_stop

    def _atac_neighborhood_cache_key(self, gene_id: int, key_cell_idx: int, sigmoid_score: float) -> Tuple[object, ...]:
        tie_start, tie_stop = self._atac_tie_block(gene_id, key_cell_idx)
        return ("atac", int(gene_id), tie_start, tie_stop, float(sigmoid_score))

    def _joint_neighborhood_cache_key(self, gene_id: int, key_cell_idx: int, sigmoid_score: float) -> Tuple[object, ...]:
        rna_key = self._neighborhood_cache_key(gene_id, key_cell_idx, sigmoid_score)
        atac_key = self._atac_neighborhood_cache_key(gene_id, key_cell_idx, sigmoid_score)
        return (
            "joint_rna_atac_conditioned_cscn",
            int(gene_id),
            int(rna_key[1]),
            int(rna_key[2]),
            int(atac_key[2]),
            int(atac_key[3]),
            float(sigmoid_score),
        )

    def _cache_atac_ci_bitset(self, cache_key: Tuple[object, ...], bitset: frozenbitarray) -> None:
        if self.max_bits_cache_entries == 0:
            return
        self.atac_ci_joint_bits_cache[cache_key] = bitset
        if self.max_bits_cache_entries > 0:
            while len(self.atac_ci_joint_bits_cache) > self.max_bits_cache_entries:
                self.atac_ci_joint_bits_cache.popitem(last=False)
                self._bump_stat("bits_cache_evictions")

    def _get_atac_bits(self, gene_id: int, key_cell_idx: int, sigmoid_score=None) -> frozenbitarray:
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        gene_id = int(gene_id)
        cache_key = self._atac_neighborhood_cache_key(gene_id, key_cell_idx, sigmoid_score)
        if cache_key in self.atac_ci_joint_bits_cache:
            self._bump_stat("bit_cache_hits")
            bitset = self.atac_ci_joint_bits_cache.pop(cache_key)
            self.atac_ci_joint_bits_cache[cache_key] = bitset
            return bitset
        sorted_indices = self.atac_ci_sorted_indices_by_id[gene_id]
        sorted_values = self.atac_ci_sorted_values_by_id[gene_id]
        window_size = int(float(sigmoid_score) * self.n_cells)
        _, _, tie_start, tie_stop, _ = cache_key
        tie_size = int(tie_stop) - int(tie_start)
        if tie_size > window_size:
            lower_bound = int(tie_start)
            upper_bound = int(tie_stop) - 1
        else:
            lower_floor = 0
            nonpositive_count = int(self.atac_ci_nonpositive_counts_by_id.get(gene_id, 0))
            if nonpositive_count > window_size and nonpositive_count < self.n_cells:
                lower_floor = nonpositive_count
            lower_bound = max(lower_floor, int(tie_start) - window_size)
            upper_bound = min(self.n_cells - 1, (int(tie_stop) - 1) + window_size)
        low_value = sorted_values[lower_bound]
        high_value = sorted_values[upper_bound]
        value_lower = int(np.searchsorted(sorted_values, low_value, side="left"))
        value_upper = int(np.searchsorted(sorted_values, high_value, side="right"))
        bitset = self._indices_to_bitset(sorted_indices[value_lower:value_upper])
        self._cache_atac_ci_bitset(cache_key, bitset)
        return bitset

    def _atac_ci_joint_enabled(self) -> bool:
        return self.atac_ci_mode == "joint_rna_atac_conditioned_cscn" and bool(self.atac_ci_profiles_by_id)

    def _get_count_bits(self, gene_id: int, key_cell_idx: int, sigmoid_score=None) -> frozenbitarray:
        if self._atac_ci_joint_enabled():
            return self.get_joint_rna_atac_bits(gene_id, key_cell_idx, sigmoid_score)
        return self.get_bits(gene_id, key_cell_idx, sigmoid_score)

    def get_joint_rna_atac_bits(self, gene_id: int, key_cell_idx: int, sigmoid_score=None) -> frozenbitarray:
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        gene_id = int(gene_id)
        if gene_id not in self.atac_ci_profiles_by_id:
            return self.get_bits(gene_id, key_cell_idx, sigmoid_score)
        cache_key = self._joint_neighborhood_cache_key(gene_id, key_cell_idx, sigmoid_score)
        if cache_key in self.atac_ci_joint_bits_cache:
            self._bump_stat("bit_cache_hits")
            bitset = self.atac_ci_joint_bits_cache.pop(cache_key)
            self.atac_ci_joint_bits_cache[cache_key] = bitset
            return bitset
        joint_bits = bitarray(self.get_bits(gene_id, key_cell_idx, sigmoid_score))
        joint_bits &= self._get_atac_bits(gene_id, key_cell_idx, sigmoid_score)
        frozen_bits = frozenbitarray(joint_bits)
        self._cache_atac_ci_bitset(cache_key, frozen_bits)
        return frozen_bits

    def get_joint_rna_atac_counts(self, genes, key_cell_idx, sigmoid_score=None):
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        gene_key = self._normalize_gene_key(genes)
        if not gene_key:
            return self.n_cells
        applied = sum(1 for gene_id in gene_key if int(gene_id) in self.atac_ci_profiles_by_id)
        self._task_atac_ci_joint_query_count += 1
        self._task_atac_ci_joint_gene_applied_count += int(applied)
        bitsets = [self.get_joint_rna_atac_bits(gene_id, key_cell_idx, sigmoid_score) for gene_id in gene_key]
        return self._bitset_intersection_count(bitsets)

    def get_hybrid_counts(self, genes, key_cell_idx, sigmoid_score=None):
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        gene_key = self._normalize_gene_key(genes)
        if not gene_key:
            return self.n_cells
        self._bump_stat("hybrid_queries")
        if len(gene_key) <= 2 or not self.subspace_trees:
            self._bump_stat("hybrid_bitmap_shortcut")
            return self.get_bits_counts(gene_key, key_cell_idx, sigmoid_score)
        spec, seed_genes = self._select_hybrid_subspace(gene_key)
        if spec is None:
            self._bump_stat("hybrid_no_subspace")
            return self.get_bits_counts(gene_key, key_cell_idx, sigmoid_score)
        if len(seed_genes) < self.hybrid_min_overlap:
            self._bump_stat("hybrid_low_overlap")
            return self.get_bits_counts(gene_key, key_cell_idx, sigmoid_score)
        candidate_bitset, _, overflow = self._get_hybrid_candidate(spec, seed_genes, key_cell_idx, sigmoid_score)
        if overflow or candidate_bitset is None:
            return self.get_bits_counts(gene_key, key_cell_idx, sigmoid_score)
        result = bitarray(candidate_bitset)
        for gene_id in gene_key:
            result &= self.get_bits(gene_id, key_cell_idx, sigmoid_score)
        return int(result.count())

    def _dispatch_counts(self, gene_key, key_cell_idx, sigmoid_score):
        if self.query_engine == "bitmap":
            return self.get_bits_counts(gene_key, key_cell_idx, sigmoid_score)
        if self.query_engine == "hybrid":
            return self.get_hybrid_counts(gene_key, key_cell_idx, sigmoid_score)
        return self.get_kdt_counts(gene_key, key_cell_idx, sigmoid_score)

    def get_conditional_counts(self, genes, key_cell_idx, sigmoid_score=None, gate_key=None, gate_bitset=None):
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        self._bump_stat("count_queries")
        gene_key = self._normalize_gene_key(genes)
        normalized_gate_key = None if gate_key is None else tuple(gate_key)
        joint_context = (
            "joint_rna_atac_conditioned_cscn",
            tuple(sorted(int(gene_id) for gene_id in self.atac_ci_profiles_by_id)),
        ) if self._atac_ci_joint_enabled() else None
        cache_key = (key_cell_idx, gene_key, sigmoid_score, self.query_engine, joint_context, normalized_gate_key)
        if cache_key in self.count_cache:
            self._bump_stat("count_cache_hits")
            return self.count_cache[cache_key]
        if gate_bitset is None and self._atac_ci_joint_enabled():
            result = self.get_joint_rna_atac_counts(gene_key, key_cell_idx, sigmoid_score)
        elif gate_bitset is None:
            result = self._dispatch_counts(gene_key, key_cell_idx, sigmoid_score)
        else:
            result_bits = bitarray(gate_bitset)
            for gene_id in gene_key:
                result_bits &= self._get_count_bits(gene_id, key_cell_idx, sigmoid_score)
            result = int(result_bits.count())
        if self.debug and self.query_engine != "bitmap":
            if gate_bitset is None and self._atac_ci_joint_enabled():
                bitmap_result = self.get_joint_rna_atac_counts(gene_key, key_cell_idx, sigmoid_score)
            elif gate_bitset is None:
                bitmap_result = self.get_bits_counts(gene_key, key_cell_idx, sigmoid_score)
            else:
                bitmap_bits = bitarray(gate_bitset)
                for gene_id in gene_key:
                    bitmap_bits &= self._get_count_bits(gene_id, key_cell_idx, sigmoid_score)
                bitmap_result = int(bitmap_bits.count())
            if result != bitmap_result:
                print(f"query_engine mismatch: {self.query_engine}={result}, bitmap={bitmap_result}, genes={gene_key}, cell={key_cell_idx}")
        self.count_cache[cache_key] = result
        return result

    def prepare_target_atac_ci(
        self,
        gene_peak_links: pd.DataFrame,
        peak_access_by_name: Mapping[str, np.ndarray],
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.atac_ci_peak_access_by_name = {}
        self.atac_ci_target_peaks_by_id = {}
        self.atac_ci_target_peak_weights_by_id = {}
        self.atac_ci_profiles_by_id = {}
        self.atac_ci_sorted_indices_by_id = {}
        self.atac_ci_sorted_values_by_id = {}
        self.atac_ci_sorted_positions_by_id = {}
        self.atac_ci_nonpositive_counts_by_id = {}
        self.atac_ci_joint_bits_cache.clear()
        self.atac_ci_source_h5 = None
        self.atac_ci_peak_count = 0
        self.atac_ci_missing_peak_count = 0
        self.atac_ci_target_count = 0
        self.atac_ci_profile_mode_used = str(self.atac_ci_profile_mode)
        self.count_cache.clear()
        self.ci_cache.clear()
        self._reset_atac_ci_stats()
        self._reset_task_atac_ci_stats()
        if self.atac_ci_mode != "joint_rna_atac_conditioned_cscn" or self.df is None or self.df.empty or self.use_nmf_applied:
            return
        required = {"gene_module_column", "peak_name"}
        if not required.issubset(gene_peak_links.columns):
            raise ValueError(f"gene peak links must contain columns: {sorted(required)}")
        links = gene_peak_links.copy()
        links["gene_module_column"] = links["gene_module_column"].astype(str)
        links["peak_name"] = links["peak_name"].astype(str).str.strip()
        if "association_pass" in links.columns:
            links = links[links["association_pass"].astype(bool)].copy()
        if "peak_gene_corr" in links.columns:
            links["peak_gene_corr"] = pd.to_numeric(links["peak_gene_corr"], errors="coerce").fillna(0.0)
            links = links[links["peak_gene_corr"] > 0].copy()
        if "peak_gene_weight" in links.columns:
            links["peak_gene_weight"] = pd.to_numeric(links["peak_gene_weight"], errors="coerce").fillna(0.0)
        columns = {str(column) for column in self.df.columns}
        links = links[links["gene_module_column"].isin(columns) & links["peak_name"].ne("")].copy()
        if links.empty:
            return
        if "peak_gene_corr" in links.columns:
            links = links.sort_values(["gene_module_column", "peak_gene_corr", "peak_name"], ascending=[True, False, True])
        else:
            links = links.sort_values(["gene_module_column", "peak_name"], ascending=[True, True])
        access_by_name = {
            str(peak_name): np.asarray(values, dtype=np.float32)
            for peak_name, values in peak_access_by_name.items()
        }
        target_peaks: Dict[int, List[str]] = {}
        target_peak_weights: Dict[int, Dict[str, float]] = {}
        for target_name, rows in links.groupby("gene_module_column", sort=False):
            gene_id = self._coerce_gene_id(str(target_name))
            peaks = []
            peak_weights: Dict[str, float] = {}
            seen = set()
            for row in rows.itertuples(index=False):
                peak_name = str(getattr(row, "peak_name"))
                if peak_name in seen or peak_name not in access_by_name:
                    continue
                seen.add(peak_name)
                peaks.append(peak_name)
                if "peak_gene_weight" in links.columns:
                    weight = float(getattr(row, "peak_gene_weight", 0.0))
                elif "peak_gene_corr" in links.columns:
                    weight = float(getattr(row, "peak_gene_corr", 0.0))
                else:
                    weight = 1.0
                peak_weights[peak_name] = max(0.0, float(weight))
            if peaks:
                target_peaks[int(gene_id)] = peaks
                target_peak_weights[int(gene_id)] = peak_weights
        metadata_dict = dict(metadata or {})
        if "atac_ci_profile_mode" in metadata_dict:
            mode = str(metadata_dict.get("atac_ci_profile_mode"))
            if mode not in ATAC_CI_PROFILE_MODES:
                raise ValueError(f"Unsupported atac_ci_profile_mode: {mode}")
            self.atac_ci_profile_mode = mode
        self.atac_ci_target_peak_weights_by_id = target_peak_weights
        self.atac_ci_profile_mode_used = str(self.atac_ci_profile_mode)
        self.atac_ci_peak_access_by_name = access_by_name
        self.atac_ci_target_peaks_by_id = target_peaks
        self.atac_ci_source_h5 = metadata_dict.get("source_h5")
        self.atac_ci_peak_count = int(metadata_dict.get("peak_count", len(access_by_name)) or 0)
        self.atac_ci_missing_peak_count = int(metadata_dict.get("missing_peak_count", 0) or 0)
        self.atac_ci_target_count = int(len(target_peaks))
        if self.atac_ci_mode == "joint_rna_atac_conditioned_cscn":
            self._prepare_joint_atac_profiles()

    def _prepare_joint_atac_profiles(self) -> None:
        self.atac_ci_profiles_by_id = {}
        self.atac_ci_sorted_indices_by_id = {}
        self.atac_ci_sorted_values_by_id = {}
        self.atac_ci_sorted_positions_by_id = {}
        self.atac_ci_nonpositive_counts_by_id = {}
        threshold = float(self.atac_ci_open_threshold)
        profile_mode = str(getattr(self, "atac_ci_profile_mode", "max"))
        for gene_id, peak_names in self.atac_ci_target_peaks_by_id.items():
            arrays = []
            weights = []
            peak_weights = getattr(self, "atac_ci_target_peak_weights_by_id", {}).get(int(gene_id), {})
            for peak_name in peak_names:
                if str(peak_name) not in self.atac_ci_peak_access_by_name:
                    continue
                values = np.asarray(self.atac_ci_peak_access_by_name[str(peak_name)], dtype=np.float32)
                if int(values.shape[0]) != int(self.n_cells):
                    continue
                arrays.append(values)
                weights.append(float(peak_weights.get(str(peak_name), 1.0)))
            if not arrays:
                continue
            if profile_mode == "weighted_sum":
                profile = np.zeros(self.n_cells, dtype=np.float32)
                for values, weight in zip(arrays, weights):
                    if weight <= 0.0:
                        continue
                    profile += values.astype(np.float32, copy=False) * float(weight)
                profile = profile.astype(np.float32, copy=False)
            else:
                profile = np.maximum.reduce(arrays).astype(np.float32, copy=False)
            if threshold > 0.0:
                profile = profile.copy()
                profile[profile <= threshold] = 0.0
            sorted_indices = np.argsort(profile, kind="mergesort").astype(np.int64, copy=False)
            sorted_values = np.ascontiguousarray(profile[sorted_indices], dtype=np.float32)
            sorted_positions = np.empty(self.n_cells, dtype=np.int64)
            sorted_positions[sorted_indices] = np.arange(self.n_cells, dtype=np.int64)
            self.atac_ci_profiles_by_id[int(gene_id)] = np.ascontiguousarray(profile, dtype=np.float32)
            self.atac_ci_sorted_indices_by_id[int(gene_id)] = sorted_indices
            self.atac_ci_sorted_values_by_id[int(gene_id)] = sorted_values
            self.atac_ci_sorted_positions_by_id[int(gene_id)] = sorted_positions
            self.atac_ci_nonpositive_counts_by_id[int(gene_id)] = int(np.count_nonzero(sorted_values <= threshold))
        self.atac_ci_target_count = int(len(self.atac_ci_profiles_by_id))

    def _atac_ci_pair_gate(self, x_id: int, y_id: int, key_cell_idx: int):
        return None, None

    def conditional_independence_test(self, X, Y, Z, data, independencies, key_cell_idx=0, significance_level=None, sigmoid_score=None) -> bool:
        if significance_level is None:
            significance_level = self.significance_level
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        try:
            self._bump_stat("ci_queries")
            x_id, y_id = sorted((self._coerce_gene_id(X), self._coerce_gene_id(Y)))
            condition_set = {self._coerce_gene_id(gene) for gene in Z}
            effective_alpha, tf_prior_strength, external_prior_strength = self._effective_prior_alpha(
                x_id,
                y_id,
                int(key_cell_idx),
                float(significance_level),
                conditioning_size=len(condition_set),
            )
            atac_gate_key, atac_gate_bitset = self._atac_ci_pair_gate(x_id, y_id, int(key_cell_idx))
            ci_key = (
                key_cell_idx,
                x_id,
                y_id,
                self._normalize_gene_key(condition_set),
                sigmoid_score,
                effective_alpha,
                atac_gate_key,
            )
            if ci_key in self.ci_cache:
                self._bump_stat("ci_cache_hits")
                return self.ci_cache[ci_key]
            if tf_prior_strength > 0.0:
                self._task_tf_skeleton_ci_query_count += 1
            if external_prior_strength > 0.0:
                self._task_external_skeleton_ci_query_count += 1

            def count_with_optional_gate(genes):
                if atac_gate_bitset is None:
                    return self.get_conditional_counts(genes, key_cell_idx, sigmoid_score)
                return self.get_conditional_counts(
                    genes,
                    key_cell_idx,
                    sigmoid_score,
                    gate_key=atac_gate_key,
                    gate_bitset=atac_gate_bitset,
                )

            count_z = count_with_optional_gate(condition_set)
            if count_z <= 1:
                self.ci_cache[ci_key] = False
                return False
            x_condition_set = condition_set.copy(); x_condition_set.add(x_id)
            count_x_z = count_with_optional_gate(x_condition_set)
            y_condition_set = condition_set.copy(); y_condition_set.add(y_id)
            count_y_z = count_with_optional_gate(y_condition_set)
            xy_condition_set = condition_set.copy(); xy_condition_set.add(x_id); xy_condition_set.add(y_id)
            count_xy_z = count_with_optional_gate(xy_condition_set)
            rho = ((count_z * count_xy_z) - (count_x_z * count_y_z)) / (count_z ** 2)
            epsilon = 1e-10
            variance_numerator = count_x_z * count_y_z * (count_z - count_x_z) * (count_z - count_y_z)
            variance_numerator = np.clip(variance_numerator, epsilon, None)
            variance_denominator = (count_z ** 4) * (count_z - 1)
            std_deviation = np.sqrt(variance_numerator / (variance_denominator + epsilon))
            z_score = np.divide(rho, std_deviation, out=np.zeros_like(rho), where=(std_deviation > epsilon))
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            if std_deviation <= epsilon and np.abs(rho) > epsilon:
                p_value = 0
            result = bool(p_value > effective_alpha)
            if tf_prior_strength > 0.0 and p_value > float(significance_level) and not result:
                self._task_tf_skeleton_rescued_ci_count += 1
            if external_prior_strength > 0.0 and p_value > float(significance_level) and not result:
                self._task_external_skeleton_rescued_ci_count += 1
            self.ci_cache[ci_key] = result
            return result
        except Exception as exc:
            print(f"ICT ERROR: {exc}")
            return True


    def _build_expert_knowledge_from_allowed_pairs(
        self,
        allowed_pairs: Set[frozenset[str]],
        extra_forbidden_edges: Optional[Iterable[Tuple[str, str]]] = None,
        restrict_to_allowed_pairs: bool = False,
    ) -> Optional[ExpertKnowledge]:
        if self.df is None or self.df.empty:
            return None
        columns = [str(col) for col in self.df.columns]
        if len(columns) < 2:
            return None
        forbidden_edges: Set[Tuple[str, str]] = set(extra_forbidden_edges or ())
        if allowed_pairs or restrict_to_allowed_pairs:
            all_pairs = {frozenset((columns[i], columns[j])) for i in range(len(columns)) for j in range(i + 1, len(columns))}
            for pair in all_pairs - allowed_pairs:
                left, right = tuple(pair)
                forbidden_edges.add((left, right))
                forbidden_edges.add((right, left))
        if not forbidden_edges:
            return None
        return ExpertKnowledge(forbidden_edges=sorted(forbidden_edges))

    def _pair_unconditioned_p_value(self, x_id: int, y_id: int, key_cell_idx: int, sigmoid_score: Optional[float] = None) -> float:
        if sigmoid_score is None:
            sigmoid_score = self.sigmoid_score
        count_z = self.n_cells
        if count_z <= 1:
            return 1.0
        x_bits = self.get_bits(x_id, key_cell_idx, sigmoid_score)
        y_bits = self.get_bits(y_id, key_cell_idx, sigmoid_score)
        count_x = int(x_bits.count())
        count_y = int(y_bits.count())
        count_xy = self._bitset_intersection_count((x_bits, y_bits))
        rho = ((count_z * count_xy) - (count_x * count_y)) / (count_z ** 2)
        epsilon = 1e-10
        variance_numerator = count_x * count_y * (count_z - count_x) * (count_z - count_y)
        variance_numerator = np.clip(variance_numerator, epsilon, None)
        variance_denominator = (count_z ** 4) * (count_z - 1)
        std_deviation = np.sqrt(variance_numerator / (variance_denominator + epsilon))
        z_score = np.divide(rho, std_deviation, out=np.zeros_like(rho), where=(std_deviation > epsilon))
        p_value = float(2 * (1 - norm.cdf(abs(z_score))))
        if std_deviation <= epsilon and np.abs(rho) > epsilon:
            return 0.0
        return max(0.0, min(1.0, p_value))

    def _tf_skeleton_prior_enabled(self) -> bool:
        return self.tf_prior_mode in TF_SKELETON_PRIOR_MODES and bool(self.tf_prior_weight_map)

    def _node_name_for_id(self, gene_id: int) -> str:
        if self.df is not None and 0 <= int(gene_id) < len(self.df.columns):
            return str(self.df.columns[int(gene_id)])
        return str(int(gene_id))

    def _get_tf_skeleton_weight_map(self, key_cell_idx: int) -> Mapping[Tuple[str, str], float]:
        return self.tf_prior_weight_map

    def _tf_skeleton_pair_strength(self, x_id: int, y_id: int, key_cell_idx: int) -> float:
        if not self._tf_skeleton_prior_enabled():
            return 0.0
        source = self._node_name_for_id(x_id)
        target = self._node_name_for_id(y_id)
        weight_map = self._get_tf_skeleton_weight_map(key_cell_idx)
        strength = max(
            float(weight_map.get((source, target), 0.0)),
            float(weight_map.get((target, source), 0.0)),
        )
        return float(np.clip(strength, 0.0, 1.0))

    def _external_skeleton_pair_strength(self, x_id: int, y_id: int) -> float:
        if self.external_prior_mode != "soft" or not self.external_prior_weight_map:
            return 0.0
        source = self._node_name_for_id(x_id)
        target = self._node_name_for_id(y_id)
        strength = float(self.external_prior_weight_map.get(frozenset((source, target)), 0.0))
        if strength <= 0.0 or strength < float(self.external_prior_min_strength):
            return 0.0
        return float(np.clip(strength, 0.0, 1.0))

    def _effective_prior_alpha(
        self,
        x_id: int,
        y_id: int,
        key_cell_idx: int,
        base_alpha: float,
        conditioning_size: int,
    ) -> Tuple[float, float, float]:
        effective_alpha = float(base_alpha)
        tf_strength = 0.0
        external_strength = self._external_skeleton_pair_strength(x_id, y_id)
        tf_allowed_for_conditioning = not (
            int(self.tf_skeleton_max_conditioning_vars) >= 0
            and int(conditioning_size) > int(self.tf_skeleton_max_conditioning_vars)
        )
        if tf_allowed_for_conditioning:
            tf_strength = self._tf_skeleton_pair_strength(x_id, y_id, key_cell_idx)
            if tf_strength > 0.0 and tf_strength >= float(self.tf_skeleton_min_strength):
                adjusted_strength = float(np.clip(tf_strength, 0.0, 1.0)) ** float(self.tf_skeleton_strength_gamma)
                tf_alpha = float(base_alpha) + (
                    adjusted_strength * (float(self.tf_skeleton_alpha) - float(base_alpha))
                )
                effective_alpha = max(float(effective_alpha), float(tf_alpha))
        if external_strength > 0.0:
            external_alpha = float(base_alpha) + (
                external_strength * (float(self.external_prior_alpha) - float(base_alpha))
            )
            effective_alpha = max(float(effective_alpha), float(external_alpha))
        return float(np.clip(effective_alpha, 0.0, 1.0)), float(tf_strength), float(external_strength)

    def _effective_tf_skeleton_alpha(
        self,
        x_id: int,
        y_id: int,
        key_cell_idx: int,
        base_alpha: float,
        conditioning_size: int = 0,
    ) -> Tuple[float, float]:
        effective_alpha, tf_strength, _ = self._effective_prior_alpha(
            x_id=x_id,
            y_id=y_id,
            key_cell_idx=key_cell_idx,
            base_alpha=base_alpha,
            conditioning_size=conditioning_size,
        )
        return effective_alpha, tf_strength

    def _annotate_tf_skeleton_stats(self, graph) -> None:
        graph.graph["tf_skeleton_ci_query_count"] = int(self._task_tf_skeleton_ci_query_count)
        graph.graph["tf_skeleton_rescued_ci_count"] = int(self._task_tf_skeleton_rescued_ci_count)
        graph.graph["external_skeleton_ci_query_count"] = int(self._task_external_skeleton_ci_query_count)
        graph.graph["external_skeleton_rescued_ci_count"] = int(self._task_external_skeleton_rescued_ci_count)

    def _annotate_atac_ci_stats(self, graph) -> None:
        graph.graph["atac_ci_gate_query_count"] = int(self._task_atac_ci_gate_query_count)
        graph.graph["atac_ci_gate_applied_count"] = int(self._task_atac_ci_gate_applied_count)
        graph.graph["atac_ci_gate_zero_key_count"] = int(self._task_atac_ci_gate_zero_key_count)
        graph.graph["atac_ci_gate_total_cells"] = int(self._task_atac_ci_gate_total_cells)
        graph.graph["atac_ci_joint_query_count"] = int(self._task_atac_ci_joint_query_count)
        graph.graph["atac_ci_joint_gene_applied_count"] = int(self._task_atac_ci_joint_gene_applied_count)

    def prepare_atac_prior(
        self,
        tf_target_prior: pd.DataFrame,
        mode: str = "atac_prior_cscn",
    ) -> None:
        self.tf_prior_mode = str(mode)
        self.extra_forbidden_edges = set()
        self.tf_local_candidate_pairs = set()
        self.tf_direct_supported_pairs = set()
        self.tf_prior_weight_map = {}
        self.tf_local_candidate_edge_count = 0
        self.tf_local_target_count = 0
        self.tf_direct_edge_count = 0
        self.tf_prior_active_tf_count = 0
        self.tf_prior_allowed_edge_count = 0
        self.tf_weighted_prior_pair_count = 0
        self.tf_weighted_prior_weight_total = 0.0
        self._reset_tf_weighted_orientation_stats()
        self._reset_tf_skeleton_stats()
        if self.df is None or self.df.empty or self.use_nmf_applied:
            self._prepare_combined_expert_knowledge()
            return
        prior_df = tf_target_prior.copy()
        required = {"tf_module_column", "gene_module_column", "tf_target_score"}
        if not required.issubset(prior_df.columns):
            raise ValueError(f"ATAC prior must contain columns: {sorted(required)}")
        prior_df["tf_module_column"] = prior_df["tf_module_column"].astype(str)
        prior_df["gene_module_column"] = prior_df["gene_module_column"].astype(str)
        prior_df["tf_target_score"] = prior_df["tf_target_score"].astype(float)
        columns = {str(col) for col in self.df.columns}
        prior_df = prior_df[
            prior_df["tf_module_column"].isin(columns)
            & prior_df["gene_module_column"].isin(columns)
            & (prior_df["tf_module_column"] != prior_df["gene_module_column"])
            & (prior_df["tf_target_score"] > 0)
        ].copy()
        if prior_df.empty:
            self._prepare_combined_expert_knowledge()
            return
        shortlist: Set[Tuple[str, str]] = set()
        direct_pairs: Set[Tuple[str, str]] = set()
        weight_map: Dict[Tuple[str, str], float] = {}
        targets_with_hits: Set[str] = set()
        tf_columns = sorted(prior_df["tf_module_column"].drop_duplicates().astype(str).tolist())
        for target, rows in prior_df.groupby("gene_module_column", sort=False):
            target = str(target)
            target_rows = rows.sort_values(["tf_target_score", "tf_module_column"], ascending=[False, True]).head(self.tf_top_k)
            if target_rows.empty:
                continue
            targets_with_hits.add(target)
            for row in target_rows.itertuples(index=False):
                pair = (str(row.tf_module_column), target)
                score = float(row.tf_target_score)
                shortlist.add(pair)
                direct_pairs.add(pair)
                weight_map[pair] = max(score, weight_map.get(pair, 0.0))
        self.tf_prior_active_tf_count = len(tf_columns)
        self.tf_local_candidate_pairs = shortlist
        self.tf_direct_supported_pairs = direct_pairs
        self.tf_prior_weight_map = weight_map
        self.tf_local_candidate_edge_count = len(shortlist)
        self.tf_local_target_count = len(targets_with_hits)
        self.tf_prior_allowed_edge_count = len(shortlist)
        self.tf_weighted_prior_pair_count = len(weight_map)
        self.tf_weighted_prior_weight_total = float(sum(weight_map.values()))
        self.tf_skeleton_prior_pair_count = len(weight_map) if mode in TF_SKELETON_PRIOR_MODES else 0
        forbidden_edges: Set[Tuple[str, str]] = set()
        if mode not in TF_WEIGHTED_PRIOR_MODES:
            for target in targets_with_hits:
                for tf_column in tf_columns:
                    if tf_column == target:
                        continue
                    forbidden_edges.add((target, tf_column))
                    if mode == "atac_prior_cscn" and (tf_column, target) not in shortlist:
                        forbidden_edges.add((tf_column, target))
        self.extra_forbidden_edges = forbidden_edges
        self._prepare_combined_expert_knowledge()

    def _tf_prior_weight(self, source, target, weight_map: Optional[Mapping[Tuple[str, str], float]] = None) -> float:
        active_weight_map = self.tf_prior_weight_map if weight_map is None else weight_map
        return float(active_weight_map.get((str(source), str(target)), 0.0))

    def _ordered_graph_nodes(self, dag) -> List[object]:
        candidates: List[object] = []
        if self.df is not None:
            candidates.extend(self.df.columns.tolist())
        candidates.extend(list(dag.nodes()))
        nodes: List[object] = []
        seen: Set[str] = set()
        for node in candidates:
            key = str(node)
            if key in seen:
                continue
            nodes.append(node)
            seen.add(key)
        return nodes

    def _orient_dag_with_weighted_tf_prior(
        self,
        dag,
        weight_map: Optional[Mapping[Tuple[str, str], float]] = None,
        allow_cycles: bool = False,
    ):
        nodes = self._ordered_graph_nodes(dag)
        node_order = {str(node): idx for idx, node in enumerate(nodes)}

        def node_sort_key(node):
            return (node_order.get(str(node), len(node_order)), str(node))

        skeleton: Dict[frozenset, Tuple[object, object]] = {}
        for source, target in dag.edges():
            if source == target:
                continue
            pair_key = frozenset((source, target))
            if pair_key not in skeleton:
                skeleton[pair_key] = (source, target)

        edge_infos = []
        for source, target in skeleton.values():
            forward_weight = self._tf_prior_weight(source, target, weight_map=weight_map)
            reverse_weight = self._tf_prior_weight(target, source, weight_map=weight_map)
            max_weight = max(forward_weight, reverse_weight)
            if reverse_weight > forward_weight:
                preferred = (target, source)
            else:
                preferred = (source, target)
            pair_key = tuple(sorted((source, target), key=node_sort_key))
            edge_infos.append(
                {
                    "source": source,
                    "target": target,
                    "preferred": preferred,
                    "priority_weight": max_weight,
                    "forward_weight": forward_weight,
                    "reverse_weight": reverse_weight,
                    "pair_key": pair_key,
                }
            )
        edge_infos.sort(
            key=lambda item: (
                -float(item["priority_weight"]),
                node_sort_key(item["pair_key"][0]),
                node_sort_key(item["pair_key"][1]),
                node_sort_key(item["source"]),
                node_sort_key(item["target"]),
            )
        )

        oriented = nx.DiGraph()
        oriented.add_nodes_from(nodes)
        stats = {
            "candidate_edge_count": 0,
            "oriented_edge_count": 0,
            "satisfied_weight": 0.0,
            "available_weight": 0.0,
            "reversed_pc_edge_count": 0,
            "cycle_avoidance_flip_count": 0,
            "forced_cycle_edge_count": 0,
            "prior_conflict_edge_count": 0,
        }

        def would_create_cycle(source, target) -> bool:
            return source == target or nx.has_path(oriented, target, source)

        for info in edge_infos:
            source = info["source"]
            target = info["target"]
            preferred_source, preferred_target = info["preferred"]
            reverse_source, reverse_target = preferred_target, preferred_source
            priority_weight = float(info["priority_weight"])
            if priority_weight > 0:
                stats["candidate_edge_count"] += 1
                stats["available_weight"] += priority_weight
                if info["forward_weight"] > 0 and info["reverse_weight"] > 0:
                    stats["prior_conflict_edge_count"] += 1

            final_source, final_target = preferred_source, preferred_target
            if not allow_cycles and would_create_cycle(final_source, final_target):
                if not would_create_cycle(reverse_source, reverse_target):
                    final_source, final_target = reverse_source, reverse_target
                    stats["cycle_avoidance_flip_count"] += 1
                else:
                    stats["forced_cycle_edge_count"] += 1
            oriented.add_edge(final_source, final_target)

            if (final_source, final_target) != (source, target):
                stats["reversed_pc_edge_count"] += 1
            final_weight = self._tf_prior_weight(final_source, final_target, weight_map=weight_map)
            if priority_weight > 0 and final_weight > 0:
                stats["oriented_edge_count"] += 1
            stats["satisfied_weight"] += final_weight

        return oriented, stats

    def _project_bidirected_edges_with_weighted_tf_prior(
        self,
        dag,
        weight_map: Optional[Mapping[Tuple[str, str], float]] = None,
    ):
        nodes = self._ordered_graph_nodes(dag)
        node_order = {str(node): idx for idx, node in enumerate(nodes)}

        def node_sort_key(node):
            return (node_order.get(str(node), len(node_order)), str(node))

        pairs: Dict[frozenset, Tuple[object, object]] = {}
        directed_edges: Set[Tuple[object, object]] = set()
        for source, target in dag.edges():
            if source == target:
                continue
            pair_key = frozenset((source, target))
            if pair_key not in pairs:
                left, right = sorted((source, target), key=node_sort_key)
                pairs[pair_key] = (left, right)
            directed_edges.add((source, target))

        projected = nx.DiGraph()
        projected.add_nodes_from(nodes)
        stats = {
            "candidate_edge_count": 0,
            "oriented_edge_count": 0,
            "satisfied_weight": 0.0,
            "available_weight": 0.0,
            "reversed_pc_edge_count": 0,
            "cycle_avoidance_flip_count": 0,
            "forced_cycle_edge_count": 0,
            "prior_conflict_edge_count": 0,
        }

        for _, (left, right) in sorted(pairs.items(), key=lambda item: (node_sort_key(item[1][0]), node_sort_key(item[1][1]))):
            left_to_right = (left, right) in directed_edges
            right_to_left = (right, left) in directed_edges
            if left_to_right and right_to_left:
                forward_weight = self._tf_prior_weight(left, right, weight_map=weight_map)
                reverse_weight = self._tf_prior_weight(right, left, weight_map=weight_map)
                priority_weight = max(forward_weight, reverse_weight)
                if priority_weight > 0:
                    stats["candidate_edge_count"] += 1
                    stats["available_weight"] += priority_weight
                if forward_weight > 0 and reverse_weight > 0:
                    stats["prior_conflict_edge_count"] += 1
                if forward_weight > reverse_weight:
                    projected.add_edge(left, right)
                    stats["oriented_edge_count"] += 1
                    stats["satisfied_weight"] += forward_weight
                    stats["reversed_pc_edge_count"] += 1
                elif reverse_weight > forward_weight:
                    projected.add_edge(right, left)
                    stats["oriented_edge_count"] += 1
                    stats["satisfied_weight"] += reverse_weight
                    stats["reversed_pc_edge_count"] += 1
                else:
                    projected.add_edge(left, right)
                    projected.add_edge(right, left)
                    stats["satisfied_weight"] += priority_weight
            elif left_to_right:
                projected.add_edge(left, right)
            elif right_to_left:
                projected.add_edge(right, left)

        projected.graph.update(getattr(dag, "graph", {}))
        projected.graph["tf_weighted_projection_mode"] = "bidirected_only"
        return projected, stats

    def apply_tf_weighted_orientation(
        self,
        records: Sequence[Tuple[int, object]],
        allow_cycles: bool = False,
        projection_only: bool = False,
    ) -> List[Tuple[int, object]]:
        self._reset_tf_weighted_orientation_stats()
        if not self.tf_prior_weight_map:
            return list(records)
        oriented_records: List[Tuple[int, object]] = []
        total_candidate_edges = 0
        total_oriented_edges = 0
        total_satisfied_weight = 0.0
        total_available_weight = 0.0
        total_reversed_pc_edges = 0
        total_cycle_flips = 0
        total_forced_cycles = 0
        total_prior_conflicts = 0
        for task_id, dag in records:
            if projection_only:
                oriented_dag, stats = self._project_bidirected_edges_with_weighted_tf_prior(
                    dag,
                )
            else:
                oriented_dag, stats = self._orient_dag_with_weighted_tf_prior(
                    dag,
                    allow_cycles=allow_cycles,
                )
            oriented_records.append((int(task_id), oriented_dag))
            total_candidate_edges += int(stats["candidate_edge_count"])
            total_oriented_edges += int(stats["oriented_edge_count"])
            total_satisfied_weight += float(stats["satisfied_weight"])
            total_available_weight += float(stats["available_weight"])
            total_reversed_pc_edges += int(stats["reversed_pc_edge_count"])
            total_cycle_flips += int(stats["cycle_avoidance_flip_count"])
            total_forced_cycles += int(stats["forced_cycle_edge_count"])
            total_prior_conflicts += int(stats["prior_conflict_edge_count"])
        self.tf_weighted_candidate_edge_count = int(total_candidate_edges)
        self.tf_weighted_oriented_edge_count = int(total_oriented_edges)
        self.tf_weighted_satisfied_weight = float(total_satisfied_weight)
        self.tf_weighted_available_weight = float(total_available_weight)
        self.tf_weighted_satisfaction_rate = (
            float(total_satisfied_weight / total_available_weight) if total_available_weight > 0 else 0.0
        )
        self.tf_weighted_reversed_pc_edge_count = int(total_reversed_pc_edges)
        self.tf_weighted_cycle_avoidance_flip_count = int(total_cycle_flips)
        self.tf_weighted_forced_cycle_edge_count = int(total_forced_cycles)
        self.tf_weighted_prior_conflict_edge_count = int(total_prior_conflicts)
        return oriented_records

    def _prepare_combined_expert_knowledge(self) -> None:
        self.combined_expert_knowledge = None
        self.combined_prior_allowed_edge_count = 0
        if self.df is None or self.df.empty or self.use_nmf_applied:
            return
        allowed_pairs: Set[frozenset[str]] = set()
        if self.external_allowed_pairs:
            allowed_pairs.update(self.external_allowed_pairs)
        self.combined_prior_allowed_edge_count = len(allowed_pairs)
        restrict_to_allowed_pairs = self.external_allowed_pairs_provided
        if self.external_prior_mode == "soft":
            allowed_pairs = set()
            restrict_to_allowed_pairs = False
        if not restrict_to_allowed_pairs and not allowed_pairs and not self.extra_forbidden_edges:
            return
        self.combined_expert_knowledge = self._build_expert_knowledge_from_allowed_pairs(
            allowed_pairs,
            extra_forbidden_edges=self.extra_forbidden_edges,
            restrict_to_allowed_pairs=restrict_to_allowed_pairs,
        )

    def _node_sort_key(self, node) -> Tuple[int, int, str]:
        idx = self.variable_name_to_idx.get(str(node))
        if idx is None:
            return (1, 0, str(node))
        return (0, int(idx), str(node))

    def _sorted_nodes(self, nodes: Iterable[object]) -> List[object]:
        return sorted(nodes, key=self._node_sort_key)

    def _sorted_undirected_edges(self, graph: nx.Graph) -> List[Tuple[object, object]]:
        edges: List[Tuple[object, object]] = []
        for left, right in graph.edges():
            ordered = tuple(sorted((left, right), key=self._node_sort_key))
            edges.append((ordered[0], ordered[1]))
        return sorted(edges, key=lambda edge: (self._node_sort_key(edge[0]), self._node_sort_key(edge[1])))

    def _required_edge_pairs(self, expert_knowledge: Optional[ExpertKnowledge]) -> Set[frozenset[object]]:
        if expert_knowledge is None:
            return set()
        return {
            frozenset((left, right))
            for left, right in getattr(expert_knowledge, "required_edges", ())
            if left != right
        }

    def _iter_potential_sepsets(
        self,
        left,
        right,
        neighbors_by_node: Mapping[object, Sequence[object]],
        temporal_ordering: Mapping[object, int],
        lim_neighbors: int,
    ):
        left_neighbors = [node for node in neighbors_by_node.get(left, ()) if node != right]
        right_neighbors = [node for node in neighbors_by_node.get(right, ()) if node != left]
        if temporal_ordering:
            max_order = min(int(temporal_ordering[left]), int(temporal_ordering[right]))
            left_neighbors = [node for node in left_neighbors if int(temporal_ordering[node]) <= max_order]
            right_neighbors = [node for node in right_neighbors if int(temporal_ordering[node]) <= max_order]
        left_neighbors = self._sorted_nodes(left_neighbors)
        right_neighbors = self._sorted_nodes(right_neighbors)
        seen: Set[Tuple[object, ...]] = set()
        for pool in (left_neighbors, right_neighbors):
            if len(pool) < int(lim_neighbors):
                continue
            for separating_set in combinations(pool, int(lim_neighbors)):
                if separating_set in seen:
                    continue
                seen.add(separating_set)
                yield separating_set

    def _build_deterministic_pc_skeleton(
        self,
        key_cell_idx: int,
        expert_knowledge: Optional[ExpertKnowledge],
        enforce_expert_knowledge: bool,
    ) -> Tuple[nx.Graph, Dict[frozenset[object], Tuple[object, ...]]]:
        if self.df is None or self.df.empty:
            raise ValueError("PC skeleton estimation requires self.df to be initialized")
        nodes = self._sorted_nodes(self.df.columns.tolist())
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for left_idx, left in enumerate(nodes):
            for right in nodes[left_idx + 1:]:
                graph.add_edge(left, right)
        temporal_ordering = {}
        if expert_knowledge is not None:
            temporal_ordering = dict(getattr(expert_knowledge, "temporal_ordering", {}) or {})
        if enforce_expert_knowledge and expert_knowledge is not None:
            for left, right in getattr(expert_knowledge, "forbidden_edges", ()):
                if graph.has_edge(left, right):
                    graph.remove_edge(left, right)
        required_pairs = self._required_edge_pairs(expert_knowledge) if enforce_expert_knowledge else set()
        separating_sets: Dict[frozenset[object], Tuple[object, ...]] = {}
        lim_neighbors = 0
        while not all(len(list(graph.neighbors(node))) < lim_neighbors for node in nodes):
            if lim_neighbors > int(self.max_cond_vars):
                break
            stable_snapshot = {
                node: tuple(self._sorted_nodes(graph.neighbors(node)))
                for node in nodes
            }
            edges = self._sorted_undirected_edges(graph)
            pending_removals: List[Tuple[object, object]] = []
            for left, right in edges:
                if frozenset((left, right)) in required_pairs:
                    continue
                if self.pc_var == "stable":
                    neighbor_source = stable_snapshot
                else:
                    neighbor_source = {
                        left: tuple(self._sorted_nodes(graph.neighbors(left))),
                        right: tuple(self._sorted_nodes(graph.neighbors(right))),
                    }
                for separating_set in self._iter_potential_sepsets(
                    left,
                    right,
                    neighbor_source,
                    temporal_ordering,
                    lim_neighbors,
                ):
                    independent = self.conditional_independence_test(
                        left,
                        right,
                        separating_set,
                        data=self.df,
                        independencies=None,
                        key_cell_idx=int(key_cell_idx),
                        significance_level=self.significance_level,
                        sigmoid_score=self.sigmoid_score,
                    )
                    if not independent:
                        continue
                    separating_sets[frozenset((left, right))] = tuple(separating_set)
                    if self.pc_var == "stable":
                        pending_removals.append((left, right))
                    elif graph.has_edge(left, right):
                        graph.remove_edge(left, right)
                    break
            if pending_removals:
                graph.remove_edges_from(pending_removals)
            if lim_neighbors >= int(self.max_cond_vars):
                break
            lim_neighbors += 1
        return graph, separating_sets

    def _orient_colliders_deterministic(
        self,
        skeleton: nx.Graph,
        separating_sets: Mapping[frozenset[object], Sequence[object]],
        temporal_ordering: Optional[Mapping[object, int]] = None,
    ) -> nx.DiGraph:
        temporal_ordering = dict(temporal_ordering or {})
        nodes = self._sorted_nodes(skeleton.nodes())
        pdag = nx.DiGraph()
        pdag.add_nodes_from(nodes)
        for left, right in self._sorted_undirected_edges(skeleton):
            pdag.add_edge(left, right)
            pdag.add_edge(right, left)
        for left in nodes:
            for right in nodes:
                if left == right or skeleton.has_edge(left, right):
                    continue
                separating_set = set(separating_sets.get(frozenset((left, right)), ()))
                shared = self._sorted_nodes(set(skeleton.neighbors(left)) & set(skeleton.neighbors(right)))
                for center in shared:
                    if center in separating_set:
                        continue
                    if temporal_ordering:
                        if int(temporal_ordering[center]) < int(temporal_ordering[left]):
                            continue
                        if int(temporal_ordering[center]) < int(temporal_ordering[right]):
                            continue
                    if pdag.has_edge(center, left):
                        pdag.remove_edge(center, left)
                    if pdag.has_edge(center, right):
                        pdag.remove_edge(center, right)
        return pdag

    def _check_incoming_edges_deterministic(self, pdag: nx.DiGraph, source, target) -> bool:
        for predecessor in self._sorted_nodes(pdag.predecessors(target)):
            if pdag.has_edge(target, predecessor):
                continue
            if pdag.has_edge(predecessor, source) or pdag.has_edge(source, predecessor):
                continue
            return True
        return False

    def _has_directed_path(self, pdag: nx.DiGraph, source, target) -> bool:
        directed_only = nx.DiGraph()
        directed_only.add_nodes_from(pdag.nodes())
        directed_only.add_edges_from(
            (left, right)
            for left, right in pdag.edges()
            if not pdag.has_edge(right, left)
        )
        return bool(nx.has_path(directed_only, source, target))

    def _apply_orientation_rules_deterministic(self, pdag: nx.DiGraph, apply_r4: bool = False) -> nx.DiGraph:
        nodes = self._sorted_nodes(pdag.nodes())
        progress = True
        while progress:
            num_edges = pdag.number_of_edges()
            for left in nodes:
                for right in nodes:
                    if left == right:
                        continue
                    if pdag.has_edge(left, right) or pdag.has_edge(right, left):
                        continue
                    left_directed = set(pdag.successors(left)) - set(pdag.predecessors(left))
                    right_undirected = set(pdag.successors(right)) & set(pdag.predecessors(right))
                    for center in self._sorted_nodes(left_directed & right_undirected):
                        if self._check_incoming_edges_deterministic(pdag, center, right):
                            continue
                        if self._has_directed_path(pdag, right, center):
                            continue
                        if pdag.has_edge(right, center):
                            pdag.remove_edge(right, center)
            for left in nodes:
                for right in nodes:
                    if left == right:
                        continue
                    if not (pdag.has_edge(left, right) and pdag.has_edge(right, left)):
                        continue
                    if self._has_directed_path(pdag, left, right) and pdag.has_edge(right, left):
                        pdag.remove_edge(right, left)
            for left in nodes:
                for right in nodes:
                    if left == right:
                        continue
                    left_undirected = set(pdag.successors(left)) & set(pdag.predecessors(left))
                    right_undirected = set(pdag.successors(right)) & set(pdag.predecessors(right))
                    shared_undirected = left_undirected & right_undirected
                    shared_directed = (set(pdag.successors(left)) - set(pdag.predecessors(left))) & (
                        set(pdag.successors(right)) - set(pdag.predecessors(right))
                    )
                    for center in self._sorted_nodes(shared_undirected):
                        for target in self._sorted_nodes(shared_directed & (set(pdag.successors(center)) & set(pdag.predecessors(center)))):
                            if pdag.has_edge(target, center):
                                pdag.remove_edge(target, center)
            if apply_r4:
                for left in nodes:
                    for right in nodes:
                        if left == right:
                            continue
                        left_undirected = set(pdag.successors(left)) & set(pdag.predecessors(left))
                        right_undirected = set(pdag.successors(right)) & set(pdag.predecessors(right))
                        for center in self._sorted_nodes(left_undirected & set(pdag.predecessors(right)) & set(pdag.successors(right))):
                            center_neighbors = set(pdag.predecessors(center)) | set(pdag.successors(center))
                            right_directed = set(pdag.successors(right)) - set(pdag.predecessors(right))
                            left_parents = set(pdag.predecessors(left))
                            for target in self._sorted_nodes(right_directed & center_neighbors & left_parents):
                                if pdag.has_edge(left, center):
                                    pdag.remove_edge(left, center)
            progress = num_edges > pdag.number_of_edges()
        return pdag

    def _pdag_to_dag_deterministic(self, pdag: nx.DiGraph) -> nx.DiGraph:
        dag = nx.DiGraph()
        dag.add_nodes_from(self._sorted_nodes(pdag.nodes()))
        directed_edges = [
            (left, right)
            for left, right in pdag.edges()
            if not pdag.has_edge(right, left)
        ]
        for left, right in sorted(directed_edges, key=lambda edge: (self._node_sort_key(edge[0]), self._node_sort_key(edge[1]))):
            dag.add_edge(left, right)
        working = pdag.copy()
        while working.number_of_nodes() > 0:
            found = False
            for node in self._sorted_nodes(working.nodes()):
                directed_outgoing = set(working.successors(node)) - set(working.predecessors(node))
                undirected_neighbors = set(working.successors(node)) & set(working.predecessors(node))
                neighbors_are_clique = all(
                    working.has_edge(left, right)
                    for right in working.predecessors(node)
                    for left in undirected_neighbors
                    if left != right
                )
                if directed_outgoing:
                    continue
                if undirected_neighbors and not neighbors_are_clique:
                    continue
                found = True
                for parent in self._sorted_nodes(working.predecessors(node)):
                    dag.add_edge(parent, node)
                working.remove_node(node)
                break
            if found:
                continue
            remaining_edges = sorted(
                working.edges(),
                key=lambda edge: (self._node_sort_key(edge[0]), self._node_sort_key(edge[1])),
            )
            for left, right in remaining_edges:
                dag.add_edge(left, right)
            dag.graph["pc_orientation_fallback"] = True
            break
        return dag

    def _pc_search_kwargs(self, key_cell_idx):
        return dict(
            variant=self.pc_var,
            ci_test=self.conditional_independence_test,
            significance_level=self.significance_level,
            max_cond_vars=self.max_cond_vars,
            key_cell_idx=key_cell_idx,
            sigmoid_score=self.sigmoid_score,
            show_progress=False,
        )

    def _estimate_pc_skeleton(self, key_cell_idx):
        expert_knowledge = self.combined_expert_knowledge
        return self._build_deterministic_pc_skeleton(
            key_cell_idx=int(key_cell_idx),
            expert_knowledge=expert_knowledge,
            enforce_expert_knowledge=expert_knowledge is not None,
        )

    def _fallback_pc_skeleton_digraph(self, key_cell_idx):
        skeleton, _ = self._estimate_pc_skeleton(key_cell_idx)
        directed = nx.DiGraph()
        if self.df is not None:
            directed.add_nodes_from(self.df.columns.tolist())
        directed.add_nodes_from(skeleton.nodes())
        for source, target in skeleton.edges():
            if source == target:
                continue
            directed.add_edge(source, target)
            directed.add_edge(target, source)
        directed.graph["pc_orientation_fallback"] = True
        return directed

    def run_pc(self, key_cell_idx):
        expert_knowledge = self.combined_expert_knowledge
        skeleton, separating_sets = self._build_deterministic_pc_skeleton(
            key_cell_idx=int(key_cell_idx),
            expert_knowledge=expert_knowledge,
            enforce_expert_knowledge=expert_knowledge is not None,
        )
        temporal_ordering = {}
        apply_r4 = False
        if expert_knowledge is not None:
            temporal_ordering = dict(getattr(expert_knowledge, "temporal_ordering", {}) or {})
            apply_r4 = bool(getattr(expert_knowledge, "temporal_order", [[]]) != [[]])
        pdag = self._orient_colliders_deterministic(
            skeleton,
            separating_sets,
            temporal_ordering=temporal_ordering,
        )
        pdag = self._apply_orientation_rules_deterministic(pdag, apply_r4=apply_r4)
        return self._pdag_to_dag_deterministic(pdag)

    def run_pc_for_task(self, task_id):
        self.clear_task_cache()
        graph = self.run_pc(int(task_id))
        self._annotate_tf_skeleton_stats(graph)
        self._annotate_atac_ci_stats(graph)
        return int(task_id), graph

    def run_pc_tasks(self, max_workers=None, backend="serial", task_ids=None):
        if backend not in CELL_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")
        if task_ids is None:
            task_ids = range(len(self.df))
        task_ids = [int(task_id) for task_id in task_ids]
        if backend == "serial" or len(task_ids) <= 1:
            return [self.run_pc_for_task(task_id) for task_id in task_ids]
        worker_count = max_workers or os.cpu_count() or 1
        chunk_size = max(1, (len(task_ids) + (worker_count * 4) - 1) // (worker_count * 4))
        task_batches = [task_ids[idx: idx + chunk_size] for idx in range(0, len(task_ids), chunk_size)]
        outputs = []
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_pc_worker, initargs=(self,)) as executor:
            futures = [executor.submit(_run_pc_worker_batch, batch) for batch in task_batches]
            for future in futures:
                outputs.extend(future.result())
        outputs.sort(key=lambda item: item[0])
        return outputs

    def __getstate__(self):
        state = self.__dict__.copy()
        state["pc_estimator"] = None
        return state

    def run_core(self, data, use_nmf=False, nmf_components=100, nmf_max_iter=50000):
        _, col = data.shape
        if use_nmf:
            n_components = min(col, int(nmf_components))
            nmf = NMF(n_components=n_components, random_state=42, max_iter=int(nmf_max_iter))
            factors = nmf.fit_transform(data)
            self.data = np.ascontiguousarray(factors)
            self.loadings = nmf.components_.T
            self.use_nmf_applied = True
            self.nmf_components_used = int(n_components)
        else:
            self.data = np.ascontiguousarray(data)
            self.loadings = np.eye(col)
            self.use_nmf_applied = False
            self.nmf_components_used = int(col)
        self._prepare_sorted_views()
        self.hybrid_candidate_limit_value = self._resolve_hybrid_candidate_limit()
        if self.query_engine == "hybrid":
            self._prepare_subspace_trees()
        else:
            self.gene_rank_by_variance = None
            self.subspace_trees = []
            self.gene_to_subspaces = {}
        columns = self.variable_names if (self.variable_names is not None and len(self.variable_names) == self.data.shape[1]) else list(range(self.data.shape[1]))
        self.df = pd.DataFrame(self.data, index=range(self.n_cells), columns=columns)
        self.variable_name_to_idx = {str(name): idx for idx, name in enumerate(columns)}
        self.pc_estimator = None
        self.reset_query_stats()
        self.kdtree = KDT(self.data.tolist(), cell_ids=np.arange(self.n_cells)) if self.query_engine == "kdt_debug" else None
        self.clear_cache()
        self._prepare_combined_expert_knowledge()
