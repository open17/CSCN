from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .config import RunConfig
from .core import CSCN
from .results import build_result_paths, save_compact_results


@dataclass(frozen=True)
class RunResult:
    module_name: str
    result_dir: Path
    results_path: Path
    manifest_path: Path
    cell_count: int
    node_count: int
    used_nmf: bool
    nmf_components_used: int
    tf_prior_mode: str


@dataclass(frozen=True)
class RunSummary:
    result_root: Path
    modules: Tuple[RunResult, ...]


def should_use_nmf(config: RunConfig, n_genes: int) -> bool:
    config.validate()
    if config.nmf_mode == 'off':
        return False
    if config.nmf_mode == 'force':
        return True
    return config.analysis_mode == 'representation' and n_genes > 150


def module_result_exists(result_dir: Path) -> bool:
    paths = build_result_paths(result_dir)
    return paths.results_path.exists() and paths.manifest_path.exists()


def load_existing_run_result(result_dir: Path) -> RunResult:
    paths = build_result_paths(result_dir)
    with paths.manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    module_name = str(manifest.get('module_name') or result_dir.name)
    return RunResult(
        module_name=module_name,
        result_dir=paths.result_dir,
        results_path=paths.results_path,
        manifest_path=paths.manifest_path,
        cell_count=int(manifest.get('barcodes_count', 0) or 0),
        node_count=int(manifest.get('node_count', 0) or 0),
        used_nmf=bool(manifest.get('use_nmf_applied', False)),
        nmf_components_used=int(manifest.get('nmf_components_used', 0) or 0),
        tf_prior_mode=str(manifest.get('tf_prior_mode', 'none') or 'none'),
    )


def _node_names_for_run(cscn: CSCN, expression_df: pd.DataFrame) -> List[str]:
    if cscn.use_nmf_applied:
        width = int(cscn.data.shape[1])
        return [f'factor_{idx:03d}' for idx in range(width)]
    return expression_df.columns.astype(str).tolist()


def _build_manifest(module_name: str, expression_df: pd.DataFrame, config: RunConfig, cscn: CSCN, node_values_file: Optional[str]) -> Dict[str, object]:
    effective_tf_mode = str(getattr(cscn, 'tf_prior_mode', config.tf_prior_mode))
    return {
        'module_name': module_name,
        'barcodes_count': int(expression_df.shape[0]),
        'node_count': int(cscn.data.shape[1]),
        'gene_names': expression_df.columns.astype(str).tolist(),
        'node_names': _node_names_for_run(cscn, expression_df),
        'analysis_mode': config.analysis_mode,
        'nmf_mode': config.nmf_mode,
        'nmf_components': int(config.nmf_components),
        'nmf_max_iter': int(config.nmf_max_iter),
        'use_nmf_applied': bool(cscn.use_nmf_applied),
        'nmf_components_used': int(cscn.nmf_components_used),
        'pc_variant': config.pc_variant,
        'query_engine': config.query_engine,
        'significance_level': float(config.significance_level),
        'max_cond_vars': int(config.max_cond_vars),
        'sigmoid_score': float(config.sigmoid_score),
        'max_bits_cache_entries': int(config.max_bits_cache_entries),
        'node_values_file': node_values_file,
        'use_tf_skeleton_prior': bool(effective_tf_mode == 'atac_prior_cscn'),
        'use_multiome_skeleton_prior': bool(config.use_multiome_skeleton_prior and not cscn.use_nmf_applied),
        'multiome_skeleton_prior_mode': str(config.multiome_skeleton_prior_mode),
        'multiome_skeleton_weight_mode': str(config.multiome_skeleton_weight_mode),
        'multiome_skeleton_alpha': float(config.multiome_skeleton_alpha),
        'multiome_skeleton_min_strength': float(config.multiome_skeleton_min_strength),
        'tf_prior_mode': effective_tf_mode,
        'tf_prior_source': str(config.tf_prior_source),
        'tf_target_prior_dir': None if config.tf_target_prior_dir is None else str(config.tf_target_prior_dir),
        'tf_list_path': None if config.tf_list_path is None else str(config.tf_list_path),
        'tf_top_k': int(config.tf_top_k),
        'tf_skeleton_alpha': float(config.tf_skeleton_alpha),
        'tf_skeleton_min_strength': float(config.tf_skeleton_min_strength),
        'tf_skeleton_max_conditioning_vars': int(config.tf_skeleton_max_conditioning_vars),
        'tf_skeleton_strength_gamma': float(config.tf_skeleton_strength_gamma),
        'atac_ci_mode': str(config.atac_ci_mode),
        'atac_ci_open_threshold': float(config.atac_ci_open_threshold),
        'atac_ci_profile_mode': str(config.atac_ci_profile_mode),
        'atac_ci_profile_mode_used': str(getattr(cscn, 'atac_ci_profile_mode_used', config.atac_ci_profile_mode)),
        'atac_ci_source_h5': None if getattr(cscn, 'atac_ci_source_h5', None) is None else str(getattr(cscn, 'atac_ci_source_h5')),
        'atac_ci_peak_count': int(getattr(cscn, 'atac_ci_peak_count', 0) or 0),
        'atac_ci_missing_peak_count': int(getattr(cscn, 'atac_ci_missing_peak_count', 0) or 0),
        'atac_ci_target_count': int(getattr(cscn, 'atac_ci_target_count', 0) or 0),
        'atac_ci_gate_query_count': int(getattr(cscn, 'atac_ci_gate_query_count', 0) or 0),
        'atac_ci_gate_applied_count': int(getattr(cscn, 'atac_ci_gate_applied_count', 0) or 0),
        'atac_ci_gate_zero_key_count': int(getattr(cscn, 'atac_ci_gate_zero_key_count', 0) or 0),
        'atac_ci_gate_total_cells': int(getattr(cscn, 'atac_ci_gate_total_cells', 0) or 0),
        'atac_ci_joint_query_count': int(getattr(cscn, 'atac_ci_joint_query_count', 0) or 0),
        'atac_ci_joint_gene_applied_count': int(getattr(cscn, 'atac_ci_joint_gene_applied_count', 0) or 0),
        'tf_prior_active_tf_count': int(getattr(cscn, 'tf_prior_active_tf_count', 0) or 0),
        'tf_prior_allowed_edge_count': int(getattr(cscn, 'tf_prior_allowed_edge_count', 0) or 0),
        'tf_local_candidate_edge_count': int(getattr(cscn, 'tf_local_candidate_edge_count', 0) or 0),
        'tf_local_target_count': int(getattr(cscn, 'tf_local_target_count', 0) or 0),
        'tf_direct_supported_pair_count': int(len(getattr(cscn, 'tf_direct_supported_pairs', ()) or ())),
        'tf_direct_edge_count': int(getattr(cscn, 'tf_direct_edge_count', 0) or 0),
        'tf_weighted_prior_pair_count': int(getattr(cscn, 'tf_weighted_prior_pair_count', 0) or 0),
        'tf_weighted_prior_weight_total': float(getattr(cscn, 'tf_weighted_prior_weight_total', 0.0) or 0.0),
        'tf_weighted_candidate_edge_count': int(getattr(cscn, 'tf_weighted_candidate_edge_count', 0) or 0),
        'tf_weighted_oriented_edge_count': int(getattr(cscn, 'tf_weighted_oriented_edge_count', 0) or 0),
        'tf_weighted_satisfied_weight': float(getattr(cscn, 'tf_weighted_satisfied_weight', 0.0) or 0.0),
        'tf_weighted_available_weight': float(getattr(cscn, 'tf_weighted_available_weight', 0.0) or 0.0),
        'tf_weighted_satisfaction_rate': float(getattr(cscn, 'tf_weighted_satisfaction_rate', 0.0) or 0.0),
        'tf_weighted_reversed_pc_edge_count': int(getattr(cscn, 'tf_weighted_reversed_pc_edge_count', 0) or 0),
        'tf_weighted_cycle_avoidance_flip_count': int(getattr(cscn, 'tf_weighted_cycle_avoidance_flip_count', 0) or 0),
        'tf_weighted_forced_cycle_edge_count': int(getattr(cscn, 'tf_weighted_forced_cycle_edge_count', 0) or 0),
        'tf_weighted_prior_conflict_edge_count': int(getattr(cscn, 'tf_weighted_prior_conflict_edge_count', 0) or 0),
        'tf_skeleton_prior_pair_count': int(getattr(cscn, 'tf_skeleton_prior_pair_count', 0) or 0),
        'tf_skeleton_ci_query_count': int(getattr(cscn, 'tf_skeleton_ci_query_count', 0) or 0),
        'tf_skeleton_rescued_ci_count': int(getattr(cscn, 'tf_skeleton_rescued_ci_count', 0) or 0),
        'tf_cell_activity_prior_pair_count': int(getattr(cscn, 'tf_cell_activity_prior_pair_count', 0) or 0),
        'external_prior_allowed_edge_count': int(getattr(cscn, 'external_prior_allowed_edge_count', 0) or 0),
        'combined_prior_allowed_edge_count': int(getattr(cscn, 'combined_prior_allowed_edge_count', 0) or 0),
        'external_skeleton_ci_query_count': int(getattr(cscn, 'external_skeleton_ci_query_count', 0) or 0),
        'external_skeleton_rescued_ci_count': int(getattr(cscn, 'external_skeleton_rescued_ci_count', 0) or 0),
    }


def _default_module_prior_path(csv_path: Path) -> Path:
    return csv_path.with_name(f'{csv_path.stem}_allowed_pairs.csv')


def _default_module_tf_target_prior_path(csv_path: Path) -> Path:
    return csv_path.with_name(f'{csv_path.stem}_tf_target_prior.csv')


def _module_tf_target_prior_path(csv_path: Path, config: RunConfig) -> Path:
    if config.tf_target_prior_dir is not None:
        return Path(config.tf_target_prior_dir) / f'{csv_path.stem}_tf_target_prior.csv'
    return _default_module_tf_target_prior_path(csv_path)


def _default_gene_peak_links_path(csv_path: Path) -> Path:
    return csv_path.with_name('gene_peak_links.csv')


def _default_preprocess_manifest_path(csv_path: Path) -> Path:
    return csv_path.with_name('preprocess_manifest.json')


def _resolve_tf_prior_mode(config: RunConfig) -> str:
    if config.tf_prior_mode != 'none':
        return str(config.tf_prior_mode)
    return 'none'


def _load_allowed_pairs(
    prior_path: Path,
) -> Union[Set[frozenset[str]], Dict[frozenset[str], float]]:
    if not prior_path.exists():
        return set()
    df = pd.read_csv(prior_path)
    required = {'gene_left', 'gene_right'}
    if not required.issubset(df.columns):
        raise ValueError(f'Module prior file {prior_path} must contain columns: {sorted(required)}')
    strength_column = None
    for candidate in ('prior_strength', 'coarse_edge_frequency', 'coarse_edge_count'):
        if candidate in df.columns:
            strength_column = candidate
            break
    if strength_column is None:
        allowed_pairs: Set[frozenset[str]] = set()
        for row in df.itertuples(index=False):
            left = str(getattr(row, 'gene_left')).strip()
            right = str(getattr(row, 'gene_right')).strip()
            if not left or not right or left == right:
                continue
            allowed_pairs.add(frozenset((left, right)))
        return allowed_pairs
    weights: Dict[frozenset[str], float] = {}
    for row in df.itertuples(index=False):
        left = str(getattr(row, 'gene_left')).strip()
        right = str(getattr(row, 'gene_right')).strip()
        if not left or not right or left == right:
            continue
        pair = frozenset((left, right))
        raw_strength = float(getattr(row, strength_column, 0.0) or 0.0)
        if strength_column == 'coarse_edge_count':
            raw_strength = raw_strength / max(float(df['coarse_edge_count'].max()), 1.0)
        strength = float(np.clip(raw_strength, 0.0, 1.0))
        if strength <= 0.0:
            continue
        weights[pair] = max(float(weights.get(pair, 0.0)), strength)
    return weights


def _load_tf_target_prior(prior_path: Path) -> pd.DataFrame:
    resolved = Path(prior_path)
    legacy_path = prior_path.with_name(prior_path.name.replace('_tf_target_prior.csv', '_tf_gene_cis_prior.csv'))
    if not resolved.exists() and legacy_path.exists():
        resolved = legacy_path
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    df = pd.read_csv(resolved)
    required = {'tf_module_column', 'gene_module_column'}
    if not required.issubset(df.columns):
        raise ValueError(f'Module TF prior file {resolved} must contain columns: {sorted(required)}')
    if 'tf_target_score' not in df.columns:
        if 'cis_motif_score' in df.columns:
            df = df.rename(columns={'cis_motif_score': 'tf_target_score'})
        else:
            raise ValueError(f'Module TF prior file {resolved} must contain tf_target_score')
    return df


def _load_gene_peak_links(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {'gene_module_column', 'peak_name'}
    if not required.issubset(df.columns):
        raise ValueError(f'Gene peak links file {path} must contain columns: {sorted(required)}')
    return df


def _load_preprocess_manifest(csv_path: Path) -> Dict[str, object]:
    manifest_path = _default_preprocess_manifest_path(csv_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _resolve_path_from_manifest(manifest_path: Path, value: object) -> Path:
    if value is None:
        raise ValueError(f'{manifest_path} does not define the required path')
    raw_path = Path(str(value))
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(manifest_path.parent / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(raw_path)


def _load_peak_access(
    csv_path: Path,
    barcodes: Sequence[str],
    supporting_peak_names: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    requested_peaks = sorted({str(name).strip() for name in supporting_peak_names if str(name).strip()})
    if not requested_peaks:
        return {}, {
            'source_h5': None,
            'peak_count': 0,
            'missing_peak_count': 0,
        }
    manifest_path = _default_preprocess_manifest_path(csv_path)
    preprocess_manifest = _load_preprocess_manifest(csv_path)
    source_h5 = _resolve_path_from_manifest(manifest_path, preprocess_manifest.get('source_h5'))
    with h5py.File(source_h5, 'r') as handle:
        matrix = handle['matrix']
        feature_names = np.array([value.decode('utf-8') for value in matrix['features']['name'][:]], dtype=object)
        feature_types = np.array([value.decode('utf-8') for value in matrix['features']['feature_type'][:]], dtype=object)
        source_barcodes = np.array([value.decode('utf-8') for value in matrix['barcodes'][:]], dtype=object)
        feature_by_barcode = sparse.csc_matrix(
            (matrix['data'][:], matrix['indices'][:], matrix['indptr'][:]),
            shape=tuple(matrix['shape'][:]),
        )
    barcode_to_idx = {str(barcode): idx for idx, barcode in enumerate(source_barcodes.tolist())}
    missing_barcodes = [str(barcode) for barcode in barcodes if str(barcode) not in barcode_to_idx]
    if missing_barcodes:
        preview = ', '.join(missing_barcodes[:5])
        raise ValueError(f'{source_h5} is missing {len(missing_barcodes)} barcodes required by {csv_path}: {preview}')
    cell_indices = np.asarray([barcode_to_idx[str(barcode)] for barcode in barcodes], dtype=np.int64)
    peak_mask = feature_types == 'Peaks'
    peak_names = feature_names[peak_mask]
    peak_name_to_idx = {str(name): idx for idx, name in enumerate(peak_names.tolist())}
    peak_matrix = feature_by_barcode[peak_mask, :].transpose().tocsr().astype(np.float32)
    peak_access_by_name: Dict[str, np.ndarray] = {}
    missing_peak_count = 0
    for peak_name in requested_peaks:
        peak_idx = peak_name_to_idx.get(peak_name)
        if peak_idx is None:
            missing_peak_count += 1
            continue
        values = np.asarray(peak_matrix[cell_indices, peak_idx].toarray(), dtype=np.float32).ravel()
        positive_values = values[values > 0]
        if positive_values.size == 0:
            peak_access_by_name[peak_name] = np.zeros(values.shape[0], dtype=np.float32)
            continue
        p95_open = float(np.percentile(positive_values, 95))
        if p95_open <= 0:
            peak_access_by_name[peak_name] = np.zeros(values.shape[0], dtype=np.float32)
            continue
        scale = float(np.log1p(p95_open))
        access = np.log1p(values).astype(np.float32, copy=False) / scale
        np.clip(access, 0.0, 1.0, out=access)
        peak_access_by_name[peak_name] = access.astype(np.float32, copy=False)
    return peak_access_by_name, {
        'source_h5': str(source_h5),
        'peak_count': int(len(peak_access_by_name)),
        'missing_peak_count': int(missing_peak_count),
    }


def _module_id_from_csv_path(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.startswith('module_') and stem.endswith('_expression'):
        return stem[len('module_'):-len('_expression')]
    raise ValueError(f'Unable to infer module id from {csv_path}')


def _count_direct_tf_edges(records: Sequence[Tuple[int, object]], direct_pairs: Set[Tuple[str, str]]) -> int:
    if not direct_pairs:
        return 0
    total = 0
    for _, dag in records:
        total += sum(1 for edge in dag.edges() if (str(edge[0]), str(edge[1])) in direct_pairs)
    return int(total)


def _record_discovery_stats(cscn: CSCN, records: Sequence[Tuple[int, object]]) -> None:
    cscn.tf_skeleton_ci_query_count = int(sum(int(graph.graph.get('tf_skeleton_ci_query_count', 0) or 0) for _, graph in records))
    cscn.tf_skeleton_rescued_ci_count = int(sum(int(graph.graph.get('tf_skeleton_rescued_ci_count', 0) or 0) for _, graph in records))
    cscn.external_skeleton_ci_query_count = int(sum(int(graph.graph.get('external_skeleton_ci_query_count', 0) or 0) for _, graph in records))
    cscn.external_skeleton_rescued_ci_count = int(sum(int(graph.graph.get('external_skeleton_rescued_ci_count', 0) or 0) for _, graph in records))
    cscn.atac_ci_gate_query_count = int(sum(int(graph.graph.get('atac_ci_gate_query_count', 0) or 0) for _, graph in records))
    cscn.atac_ci_gate_applied_count = int(sum(int(graph.graph.get('atac_ci_gate_applied_count', 0) or 0) for _, graph in records))
    cscn.atac_ci_gate_zero_key_count = int(sum(int(graph.graph.get('atac_ci_gate_zero_key_count', 0) or 0) for _, graph in records))
    cscn.atac_ci_gate_total_cells = int(sum(int(graph.graph.get('atac_ci_gate_total_cells', 0) or 0) for _, graph in records))
    cscn.atac_ci_joint_query_count = int(sum(int(graph.graph.get('atac_ci_joint_query_count', 0) or 0) for _, graph in records))
    cscn.atac_ci_joint_gene_applied_count = int(sum(int(graph.graph.get('atac_ci_joint_gene_applied_count', 0) or 0) for _, graph in records))


def run_module(
    expression_df: pd.DataFrame,
    config: RunConfig,
    module_name: str = 'module',
    result_dir: Optional[Path] = None,
    allowed_undirected_pairs: Optional[Union[Set[frozenset[str]], Mapping[frozenset[str], float]]] = None,
    tf_target_prior: Optional[pd.DataFrame] = None,
    atac_ci_gene_peak_links: Optional[pd.DataFrame] = None,
    atac_ci_peak_access_by_name: Optional[Dict[str, np.ndarray]] = None,
    atac_ci_metadata: Optional[Dict[str, object]] = None,
) -> RunResult:
    config = config.validate()
    result_dir = Path(result_dir or module_name)
    result_dir.mkdir(parents=True, exist_ok=True)
    use_nmf = should_use_nmf(config, expression_df.shape[1])
    allowed_pair_weights = None
    allowed_pair_set = None
    if isinstance(allowed_undirected_pairs, dict):
        if str(config.multiome_skeleton_weight_mode) == 'binary':
            allowed_pair_set = {
                frozenset(str(node) for node in pair)
                for pair, weight in allowed_undirected_pairs.items()
                if len(pair) == 2 and float(weight) > 0.0
            }
        else:
            allowed_pair_weights = {
                frozenset(str(node) for node in pair): float(weight)
                for pair, weight in allowed_undirected_pairs.items()
                if len(pair) == 2 and float(weight) > 0.0
            }
            allowed_pair_set = set(allowed_pair_weights)
    else:
        allowed_pair_set = allowed_undirected_pairs
    effective_tf_prior_mode = _resolve_tf_prior_mode(config)
    tf_prior_modes = {'atac_prior_cscn'}
    tf_weighted_modes = {'atac_prior_cscn'}
    if effective_tf_prior_mode in tf_prior_modes and use_nmf:
        raise ValueError(f"tf_prior_mode='{effective_tf_prior_mode}' is not supported when NMF is applied")
    cscn = CSCN(
        sigmoid_score=config.sigmoid_score,
        pc_var=config.pc_variant,
        significance_level=config.significance_level,
        max_cond_vars=config.max_cond_vars,
        query_engine=config.query_engine,
        max_bits_cache_entries=config.max_bits_cache_entries,
        tf_prior_mode=effective_tf_prior_mode,
        allowed_undirected_pairs=allowed_pair_set,
        allowed_undirected_pair_weights=allowed_pair_weights,
        external_prior_mode=config.multiome_skeleton_prior_mode if allowed_pair_set is not None else "hard",
        external_prior_alpha=config.multiome_skeleton_alpha,
        external_prior_min_strength=config.multiome_skeleton_min_strength,
        tf_list_path=config.tf_list_path,
        tf_top_k=config.tf_top_k,
        tf_skeleton_alpha=config.tf_skeleton_alpha,
        tf_skeleton_min_strength=config.tf_skeleton_min_strength,
        tf_skeleton_max_conditioning_vars=config.tf_skeleton_max_conditioning_vars,
        tf_skeleton_strength_gamma=config.tf_skeleton_strength_gamma,
        atac_ci_mode=config.atac_ci_mode,
        atac_ci_open_threshold=config.atac_ci_open_threshold,
        atac_ci_profile_mode=config.atac_ci_profile_mode,
        variable_names=expression_df.columns.astype(str).tolist(),
    )
    values = expression_df.to_numpy(dtype=np.float32, copy=False)
    cscn.run_core(values, use_nmf=use_nmf, nmf_components=config.nmf_components, nmf_max_iter=config.nmf_max_iter)
    if effective_tf_prior_mode in tf_prior_modes:
        if tf_target_prior is None:
            raise ValueError(f"tf_target_prior is required for tf_prior_mode='{effective_tf_prior_mode}'")
        cscn.prepare_atac_prior(
            tf_target_prior=tf_target_prior,
            mode=effective_tf_prior_mode,
        )
    if config.atac_ci_mode in {'joint_rna_atac_conditioned_cscn'}:
        if atac_ci_gene_peak_links is None:
            raise ValueError(f"atac_ci_gene_peak_links is required when atac_ci_mode='{config.atac_ci_mode}'")
        if atac_ci_peak_access_by_name is None:
            raise ValueError(f"atac_ci_peak_access_by_name is required when atac_ci_mode='{config.atac_ci_mode}'")
        cscn.prepare_target_atac_ci(
            gene_peak_links=atac_ci_gene_peak_links,
            peak_access_by_name=atac_ci_peak_access_by_name,
            metadata=atac_ci_metadata,
        )
    cell_backend = 'process' if config.parallel_scope == 'cell' else 'serial'
    records = cscn.run_pc_tasks(max_workers=config.workers, backend=cell_backend)
    _record_discovery_stats(cscn, records)
    if effective_tf_prior_mode in tf_weighted_modes:
        records = cscn.apply_tf_weighted_orientation(
            records,
            allow_cycles=True,
            projection_only=False,
        )
    cscn.tf_direct_edge_count = _count_direct_tf_edges(records, cscn.tf_direct_supported_pairs)
    node_values_file = None
    if cscn.use_nmf_applied:
        node_values_file = 'node_values.npy'
        np.save(result_dir / node_values_file, np.asarray(cscn.data, dtype=np.float32))
    manifest = _build_manifest(module_name, expression_df, config, cscn, node_values_file=node_values_file)
    paths = save_compact_results(
        result_dir,
        records,
        node_count=int(cscn.data.shape[1]),
        manifest=manifest,
        node_names=cscn.df.columns.tolist(),
    )
    if config.save_debug_state:
        with paths.debug_state_path.open('wb') as handle:
            pickle.dump(cscn, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return RunResult(
        module_name=module_name,
        result_dir=paths.result_dir,
        results_path=paths.results_path,
        manifest_path=paths.manifest_path,
        cell_count=int(expression_df.shape[0]),
        node_count=int(cscn.data.shape[1]),
        used_nmf=bool(cscn.use_nmf_applied),
        nmf_components_used=int(cscn.nmf_components_used),
        tf_prior_mode=str(cscn.tf_prior_mode),
    )


def run_module_csv(csv_path: Path, result_root: Path, config: RunConfig) -> RunResult:
    df = pd.read_csv(csv_path, index_col=0)
    module_name = csv_path.stem
    allowed_pairs = None
    effective_tf_prior_mode = _resolve_tf_prior_mode(config)
    # The multiome skeleton and TF-target prior encode different evidence:
    # observed RNA/ATAC module adjacency versus directed motif support.  They
    # must remain composable.  Previously selecting ``atac_prior_cscn``
    # silently disabled the multiome skeleton, despite both options being
    # enabled in RunConfig.
    if config.use_multiome_skeleton_prior:
        prior_path = _default_module_prior_path(csv_path)
        if prior_path.exists():
            allowed_pairs = _load_allowed_pairs(prior_path)
    tf_target_prior = None
    atac_ci_gene_peak_links = None
    atac_ci_peak_access_by_name = None
    atac_ci_metadata = None
    if effective_tf_prior_mode == 'atac_prior_cscn':
        tf_target_prior = _load_tf_target_prior(_module_tf_target_prior_path(csv_path, config))
    if config.atac_ci_mode in {'joint_rna_atac_conditioned_cscn'}:
        gene_peak_links = _load_gene_peak_links(_default_gene_peak_links_path(csv_path))
        module_genes = set(df.columns.astype(str).tolist())
        atac_ci_gene_peak_links = gene_peak_links[
            gene_peak_links['gene_module_column'].astype(str).isin(module_genes)
        ].copy()
        atac_ci_peak_access_by_name, atac_ci_metadata = _load_peak_access(
            csv_path,
            df.index.astype(str).tolist(),
            atac_ci_gene_peak_links.get('peak_name', pd.Series(dtype=object)).dropna().astype(str).tolist(),
        )
        atac_ci_metadata = {**dict(atac_ci_metadata or {}), 'atac_ci_profile_mode': str(config.atac_ci_profile_mode)}
    return run_module(
        df,
        config=config,
        module_name=module_name,
        result_dir=result_root / module_name,
        allowed_undirected_pairs=allowed_pairs,
        tf_target_prior=tf_target_prior,
        atac_ci_gene_peak_links=atac_ci_gene_peak_links,
        atac_ci_peak_access_by_name=atac_ci_peak_access_by_name,
        atac_ci_metadata=atac_ci_metadata,
    )


def _run_module_csv_worker(csv_path_str: str, result_root_str: str, config: RunConfig) -> RunResult:
    return run_module_csv(Path(csv_path_str), Path(result_root_str), config)


def run_directory(module_dir: Path, result_root: Path, config: RunConfig) -> RunSummary:
    config = config.validate()
    module_paths = sorted(Path(module_dir).glob('module_*_expression.csv'))
    if not module_paths:
        raise ValueError(f'No module expression files found in {module_dir}')
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    results: List[RunResult] = []
    pending_paths: List[Path] = []
    for path in module_paths:
        module_result_dir = result_root / path.stem
        if module_result_exists(module_result_dir):
            results.append(load_existing_run_result(module_result_dir))
        else:
            pending_paths.append(path)
    if not pending_paths:
        results.sort(key=lambda item: item.module_name)
        return RunSummary(result_root=result_root, modules=tuple(results))
    if config.parallel_scope == 'module' and len(pending_paths) > 1:
        module_config = replace(config, parallel_scope='none')
        module_worker_limit = min(len(pending_paths), 100)
        max_workers = module_worker_limit if config.workers is None else min(int(config.workers), module_worker_limit)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_module_csv_worker, str(path), str(result_root), module_config): path
                for path in pending_paths
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for path in pending_paths:
            results.append(run_module_csv(path, result_root=result_root, config=config))
    results.sort(key=lambda item: item.module_name)
    return RunSummary(result_root=result_root, modules=tuple(results))
