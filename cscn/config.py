from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ANALYSIS_MODES = {"regulatory", "representation"}
NMF_MODES = {"off", "auto", "force"}
QUERY_ENGINES = {"bitmap", "hybrid", "kdt_debug"}
PARALLEL_SCOPES = {"none", "module", "cell"}
INPUT_MODALITIES = {"scrna", "paired_multiome"}
MODULE_BACKENDS = {"wgcna", "multiome_refined_wgcna"}
ATAC_CI_MODES = {"none", "joint_rna_atac_conditioned_cscn"}
ATAC_CI_PROFILE_MODES = {"max", "weighted_sum"}
TF_PRIOR_MODES = {"none", "atac_prior_cscn"}
BETA_MODES = {"blend", "clipped_blend", "expression", "rank_blend", "uniform"}
EDGE_WEIGHT_MODES = {"binary", "tf_prior"}
MULTIOME_SKELETON_PRIOR_MODES = {"hard", "soft"}
MULTIOME_SKELETON_WEIGHT_MODES = {"weighted", "binary"}


@dataclass(frozen=True)
class RunConfig:
    analysis_mode: str = "representation"
    nmf_mode: str = "auto"
    nmf_components: int = 100
    nmf_max_iter: int = 50000
    query_engine: str = "hybrid"
    pc_variant: str = "stable"
    max_cond_vars: int = 20
    significance_level: float = 0.01
    sigmoid_score: float = 0.1
    parallel_scope: str = "module"
    workers: Optional[int] = None
    max_bits_cache_entries: int = 8192
    save_debug_state: bool = False
    use_multiome_skeleton_prior: bool = False
    multiome_skeleton_prior_mode: str = "hard"
    multiome_skeleton_weight_mode: str = "weighted"
    multiome_skeleton_alpha: float = 0.20
    multiome_skeleton_min_strength: float = 0.0
    tf_prior_mode: str = "none"
    tf_target_prior_dir: Optional[Path] = None
    tf_prior_source: str = "module"
    tf_list_path: Optional[Path] = None
    tf_top_k: int = 5
    tf_skeleton_alpha: float = 0.20
    tf_skeleton_min_strength: float = 0.0
    tf_skeleton_max_conditioning_vars: int = -1
    tf_skeleton_strength_gamma: float = 1.0
    atac_ci_mode: str = "none"
    atac_ci_open_threshold: float = 0.0
    atac_ci_profile_mode: str = "max"

    def validate(self) -> "RunConfig":
        if self.analysis_mode not in ANALYSIS_MODES:
            raise ValueError(f"Unsupported analysis_mode: {self.analysis_mode}")
        if self.nmf_mode not in NMF_MODES:
            raise ValueError(f"Unsupported nmf_mode: {self.nmf_mode}")
        if self.query_engine not in QUERY_ENGINES:
            raise ValueError(f"Unsupported query_engine: {self.query_engine}")
        if self.atac_ci_mode not in ATAC_CI_MODES:
            raise ValueError(f"Unsupported atac_ci_mode: {self.atac_ci_mode}")
        if self.atac_ci_profile_mode not in ATAC_CI_PROFILE_MODES:
            raise ValueError(f"Unsupported atac_ci_profile_mode: {self.atac_ci_profile_mode}")
        if self.parallel_scope not in PARALLEL_SCOPES:
            raise ValueError(f"Unsupported parallel_scope: {self.parallel_scope}")
        if self.tf_prior_mode not in TF_PRIOR_MODES:
            raise ValueError(f"Unsupported tf_prior_mode: {self.tf_prior_mode}")
        if self.tf_target_prior_dir is not None and not Path(self.tf_target_prior_dir).exists():
            raise FileNotFoundError(self.tf_target_prior_dir)
        if self.analysis_mode == "regulatory" and self.nmf_mode == "force":
            raise ValueError("regulatory mode does not allow nmf_mode='force'")
        if self.workers is not None and self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.max_bits_cache_entries < -1:
            raise ValueError("max_bits_cache_entries must be >= -1")
        if self.multiome_skeleton_prior_mode not in MULTIOME_SKELETON_PRIOR_MODES:
            raise ValueError(f"Unsupported multiome_skeleton_prior_mode: {self.multiome_skeleton_prior_mode}")
        if self.multiome_skeleton_weight_mode not in MULTIOME_SKELETON_WEIGHT_MODES:
            raise ValueError(f"Unsupported multiome_skeleton_weight_mode: {self.multiome_skeleton_weight_mode}")
        if not 0.0 <= float(self.multiome_skeleton_alpha) <= 1.0:
            raise ValueError("multiome_skeleton_alpha must be within [0, 1]")
        if not 0.0 <= float(self.multiome_skeleton_min_strength) <= 1.0:
            raise ValueError("multiome_skeleton_min_strength must be within [0, 1]")
        if self.nmf_components < 1:
            raise ValueError("nmf_components must be >= 1")
        if self.nmf_max_iter < 1:
            raise ValueError("nmf_max_iter must be >= 1")
        if self.max_cond_vars < 0:
            raise ValueError("max_cond_vars must be >= 0")
        if self.tf_top_k < 1:
            raise ValueError("tf_top_k must be >= 1")
        if not 0.0 <= float(self.tf_skeleton_alpha) <= 1.0:
            raise ValueError("tf_skeleton_alpha must be within [0, 1]")
        if not 0.0 <= float(self.tf_skeleton_min_strength) <= 1.0:
            raise ValueError("tf_skeleton_min_strength must be within [0, 1]")
        if int(self.tf_skeleton_max_conditioning_vars) < -1:
            raise ValueError("tf_skeleton_max_conditioning_vars must be >= -1")
        if float(self.tf_skeleton_strength_gamma) <= 0.0:
            raise ValueError("tf_skeleton_strength_gamma must be > 0")
        if float(self.atac_ci_open_threshold) < 0.0:
            raise ValueError("atac_ci_open_threshold must be >= 0")
        return self


@dataclass(frozen=True)
class PreprocessConfig:
    input_modality: str
    analysis_mode: str = "regulatory"
    module_backend: str = "wgcna"
    input_csv: Optional[Path] = None
    input_h5: Optional[Path] = None
    gtf: Optional[Path] = None
    prefix: str = "cscn_modules"
    top_genes: int = 1000
    top_cells: int = 2000
    top_peaks: int = 5000
    min_gene_cells: int = 10
    min_peak_cells: int = 10
    module_min_size: int = 20
    module_max_size: int = 120
    soft_power: int = 6
    cut_height: float = 0.7
    knn_k: int = 30
    rna_components: int = 30
    atac_components: int = 30
    promoter_window: int = 2000
    distal_window: int = 150000
    max_distal_links: int = 5
    enable_tf_motif_prior: bool = False
    tf_list_path: Optional[Path] = None
    genome_fasta: Optional[Path] = None
    peak_gene_window: int = 250000
    peak_gene_corr_threshold: float = 0.1

    def validate(self) -> "PreprocessConfig":
        if self.input_modality not in INPUT_MODALITIES:
            raise ValueError(f"Unsupported input_modality: {self.input_modality}")
        if self.analysis_mode not in ANALYSIS_MODES:
            raise ValueError(f"Unsupported analysis_mode: {self.analysis_mode}")
        if self.module_backend not in MODULE_BACKENDS:
            raise ValueError(f"Unsupported module_backend: {self.module_backend}")
        if self.input_modality == "scrna" and self.input_csv is None:
            raise ValueError("input_csv is required when input_modality='scrna'")
        if self.input_modality == "paired_multiome" and self.input_h5 is None:
            raise ValueError("input_h5 is required when input_modality='paired_multiome'")
        if self.enable_tf_motif_prior:
            if self.input_modality != "paired_multiome":
                raise ValueError("enable_tf_motif_prior is only supported for input_modality='paired_multiome'")
            if self.genome_fasta is None:
                raise ValueError("genome_fasta is required when enable_tf_motif_prior=True")
            if self.gtf is None:
                raise ValueError("gtf is required when enable_tf_motif_prior=True")
        if self.peak_gene_window < 1:
            raise ValueError("peak_gene_window must be >= 1")
        if self.peak_gene_corr_threshold < -1 or self.peak_gene_corr_threshold > 1:
            raise ValueError("peak_gene_corr_threshold must be within [-1, 1]")
        return self
