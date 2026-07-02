from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    ATAC_CI_MODES,
    ATAC_CI_PROFILE_MODES,
    BETA_MODES,
    EDGE_WEIGHT_MODES,
    MULTIOME_SKELETON_PRIOR_MODES,
    MULTIOME_SKELETON_WEIGHT_MODES,
    PreprocessConfig,
    RunConfig,
)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--analysis-mode", choices=["regulatory", "representation"], default="representation")
    parser.add_argument("--nmf-mode", choices=["off", "auto", "force"], default="auto")
    parser.add_argument("--nmf-components", type=int, default=100)
    parser.add_argument("--nmf-max-iter", type=int, default=50000)
    parser.add_argument("--query-engine", choices=["bitmap", "hybrid", "kdt_debug"], default="hybrid")
    parser.add_argument("--pc-variant", choices=["stable", "orig"], default="stable")
    parser.add_argument("--max-cond-vars", type=int, default=20)
    parser.add_argument("--significance-level", type=float, default=0.01)
    parser.add_argument("--sigmoid-score", type=float, default=0.1)
    parser.add_argument("--parallel-scope", choices=["none", "module", "cell"], default="module")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--max-bits-cache-entries",
        type=int,
        default=8192,
        help="Per-worker bitset cache limit; use -1 for unlimited or 0 to disable bitset caching",
    )
    parser.add_argument("--save-debug-state", action="store_true")
    parser.add_argument("--use-multiome-skeleton-prior", action="store_true")
    parser.add_argument("--multiome-skeleton-prior-mode", choices=sorted(MULTIOME_SKELETON_PRIOR_MODES), default="hard")
    parser.add_argument("--multiome-skeleton-weight-mode", choices=sorted(MULTIOME_SKELETON_WEIGHT_MODES), default="weighted")
    parser.add_argument("--multiome-skeleton-alpha", type=float, default=0.20)
    parser.add_argument("--multiome-skeleton-min-strength", type=float, default=0.0)
    parser.add_argument(
        "--tf-prior-mode",
        choices=["none", "atac_prior_cscn"],
        default="none",
    )
    parser.add_argument("--tf-target-prior-dir", type=Path)
    parser.add_argument("--tf-prior-source", default="module")
    parser.add_argument("--tf-list-path", type=Path)
    parser.add_argument("--tf-top-k", type=int, default=5)
    parser.add_argument("--tf-skeleton-alpha", type=float, default=0.20)
    parser.add_argument("--tf-skeleton-min-strength", type=float, default=0.0)
    parser.add_argument("--tf-skeleton-max-conditioning-vars", type=int, default=-1)
    parser.add_argument("--tf-skeleton-strength-gamma", type=float, default=1.0)
    parser.add_argument("--atac-ci-mode", choices=sorted(ATAC_CI_MODES), default="none")
    parser.add_argument("--atac-ci-open-threshold", type=float, default=0.0)
    parser.add_argument("--atac-ci-profile-mode", choices=sorted(ATAC_CI_PROFILE_MODES), default="max")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSCN paper-code CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Build CSCN-ready modules")
    preprocess_parser.add_argument("--input-modality", required=True, choices=["scrna", "paired_multiome"])
    preprocess_parser.add_argument("--analysis-mode", default="regulatory", choices=["regulatory", "representation"])
    preprocess_parser.add_argument("--module-backend", required=True, choices=["wgcna", "multiome_refined_wgcna"])
    preprocess_parser.add_argument("--input-csv", type=Path)
    preprocess_parser.add_argument("--input-h5", type=Path)
    preprocess_parser.add_argument("--gtf", type=Path)
    preprocess_parser.add_argument("--outdir", required=True, type=Path)
    preprocess_parser.add_argument("--prefix", default="cscn_modules")
    preprocess_parser.add_argument("--top-genes", default=1000, type=int)
    preprocess_parser.add_argument("--top-cells", default=2000, type=int)
    preprocess_parser.add_argument("--top-peaks", default=5000, type=int)
    preprocess_parser.add_argument("--min-gene-cells", default=10, type=int)
    preprocess_parser.add_argument("--min-peak-cells", default=10, type=int)
    preprocess_parser.add_argument("--module-min-size", default=20, type=int)
    preprocess_parser.add_argument("--module-max-size", default=120, type=int)
    preprocess_parser.add_argument("--soft-power", default=6, type=int)
    preprocess_parser.add_argument("--cut-height", default=0.7, type=float)
    preprocess_parser.add_argument("--knn-k", default=30, type=int)
    preprocess_parser.add_argument("--rna-components", default=30, type=int)
    preprocess_parser.add_argument("--atac-components", default=30, type=int)
    preprocess_parser.add_argument("--promoter-window", default=2000, type=int)
    preprocess_parser.add_argument("--distal-window", default=150000, type=int)
    preprocess_parser.add_argument("--max-distal-links", default=5, type=int)
    preprocess_parser.add_argument("--enable-tf-motif-prior", action="store_true")
    preprocess_parser.add_argument("--tf-list-path", type=Path)
    preprocess_parser.add_argument("--genome-fasta", type=Path)
    preprocess_parser.add_argument("--peak-gene-window", default=250000, type=int)
    preprocess_parser.add_argument("--peak-gene-corr-threshold", default=0.1, type=float)

    run_parser = subparsers.add_parser("run", help="Run CSCN on one module CSV or a directory of modules")
    run_inputs = run_parser.add_mutually_exclusive_group(required=True)
    run_inputs.add_argument("--module-csv", type=Path)
    run_inputs.add_argument("--module-dir", type=Path)
    run_parser.add_argument("--result-dir", required=True, type=Path)
    add_run_arguments(run_parser)

    ckm_parser = subparsers.add_parser("ckm", help="Build CKM from existing CSCN results")
    ckm_parser.add_argument("--module-dir", required=True, type=Path)
    ckm_parser.add_argument("--result-dir", required=True, type=Path)
    ckm_parser.add_argument("--alpha", type=float, default=0.05)
    ckm_parser.add_argument("--beta-mode", choices=sorted(BETA_MODES), default="expression")
    ckm_parser.add_argument("--beta-lambda", type=float, default=1.0)
    ckm_parser.add_argument("--edge-weight-mode", choices=sorted(EDGE_WEIGHT_MODES), default="binary")
    ckm_parser.add_argument("--edge-weight-lambda", type=float, default=1.0)
    ckm_parser.add_argument("--tf-top-k", type=int, default=20)
    ckm_parser.add_argument("--out-csv", required=True, type=Path)

    return parser


def command_preprocess(args: argparse.Namespace) -> None:
    config = PreprocessConfig(
        input_modality=args.input_modality,
        analysis_mode=args.analysis_mode,
        module_backend=args.module_backend,
        input_csv=args.input_csv,
        input_h5=args.input_h5,
        gtf=args.gtf,
        prefix=args.prefix,
        top_genes=args.top_genes,
        top_cells=args.top_cells,
        top_peaks=args.top_peaks,
        min_gene_cells=args.min_gene_cells,
        min_peak_cells=args.min_peak_cells,
        module_min_size=args.module_min_size,
        module_max_size=args.module_max_size,
        soft_power=args.soft_power,
        cut_height=args.cut_height,
        knn_k=args.knn_k,
        rna_components=args.rna_components,
        atac_components=args.atac_components,
        promoter_window=args.promoter_window,
        distal_window=args.distal_window,
        max_distal_links=args.max_distal_links,
        enable_tf_motif_prior=bool(args.enable_tf_motif_prior),
        tf_list_path=args.tf_list_path,
        genome_fasta=args.genome_fasta,
        peak_gene_window=args.peak_gene_window,
        peak_gene_corr_threshold=args.peak_gene_corr_threshold,
    )
    from .preprocess import build_modules

    result = build_modules(config, args.outdir)
    print(json.dumps(result.manifest, indent=2))


def command_run(args: argparse.Namespace) -> None:
    config = RunConfig(
        analysis_mode=args.analysis_mode,
        nmf_mode=args.nmf_mode,
        nmf_components=args.nmf_components,
        nmf_max_iter=args.nmf_max_iter,
        query_engine=args.query_engine,
        pc_variant=args.pc_variant,
        max_cond_vars=args.max_cond_vars,
        significance_level=args.significance_level,
        sigmoid_score=args.sigmoid_score,
        parallel_scope=args.parallel_scope,
        workers=args.workers,
        max_bits_cache_entries=args.max_bits_cache_entries,
        save_debug_state=args.save_debug_state,
        use_multiome_skeleton_prior=args.use_multiome_skeleton_prior,
        multiome_skeleton_prior_mode=args.multiome_skeleton_prior_mode,
        multiome_skeleton_weight_mode=args.multiome_skeleton_weight_mode,
        multiome_skeleton_alpha=args.multiome_skeleton_alpha,
        multiome_skeleton_min_strength=args.multiome_skeleton_min_strength,
        tf_prior_mode=args.tf_prior_mode,
        tf_target_prior_dir=args.tf_target_prior_dir,
        tf_prior_source=args.tf_prior_source,
        tf_list_path=args.tf_list_path,
        tf_top_k=args.tf_top_k,
        tf_skeleton_alpha=args.tf_skeleton_alpha,
        tf_skeleton_min_strength=args.tf_skeleton_min_strength,
        tf_skeleton_max_conditioning_vars=args.tf_skeleton_max_conditioning_vars,
        tf_skeleton_strength_gamma=args.tf_skeleton_strength_gamma,
        atac_ci_mode=args.atac_ci_mode,
        atac_ci_open_threshold=args.atac_ci_open_threshold,
        atac_ci_profile_mode=args.atac_ci_profile_mode,
    ).validate()
    from .runner import run_directory, run_module_csv

    if args.module_csv is not None:
        result = run_module_csv(args.module_csv, args.result_dir, config)
        payload = {
            "module_name": result.module_name,
            "result_dir": str(result.result_dir),
            "results_path": str(result.results_path),
            "manifest_path": str(result.manifest_path),
            "cell_count": result.cell_count,
            "node_count": result.node_count,
            "used_nmf": result.used_nmf,
            "nmf_components_used": result.nmf_components_used,
            "tf_prior_mode": result.tf_prior_mode,
            "tf_prior_source": config.tf_prior_source,
            "multiome_skeleton_prior_mode": config.multiome_skeleton_prior_mode,
            "multiome_skeleton_weight_mode": config.multiome_skeleton_weight_mode,
            "multiome_skeleton_alpha": config.multiome_skeleton_alpha,
            "multiome_skeleton_min_strength": config.multiome_skeleton_min_strength,
            "tf_target_prior_dir": None if config.tf_target_prior_dir is None else str(config.tf_target_prior_dir),
            "tf_skeleton_alpha": config.tf_skeleton_alpha,
            "tf_skeleton_min_strength": config.tf_skeleton_min_strength,
            "tf_skeleton_max_conditioning_vars": config.tf_skeleton_max_conditioning_vars,
            "tf_skeleton_strength_gamma": config.tf_skeleton_strength_gamma,
            "atac_ci_mode": config.atac_ci_mode,
            "atac_ci_profile_mode": config.atac_ci_profile_mode,
            "atac_ci_open_threshold": config.atac_ci_open_threshold,
        }
    else:
        summary = run_directory(args.module_dir, args.result_dir, config)
        payload = {
            "result_root": str(summary.result_root),
            "module_count": len(summary.modules),
            "modules": [
                {
                    "module_name": module.module_name,
                    "result_dir": str(module.result_dir),
                    "cell_count": module.cell_count,
                    "node_count": module.node_count,
                    "used_nmf": module.used_nmf,
                    "nmf_components_used": module.nmf_components_used,
                    "tf_prior_mode": module.tf_prior_mode,
                    "tf_prior_source": config.tf_prior_source,
                    "multiome_skeleton_prior_mode": config.multiome_skeleton_prior_mode,
                    "multiome_skeleton_weight_mode": config.multiome_skeleton_weight_mode,
                    "multiome_skeleton_alpha": config.multiome_skeleton_alpha,
                    "multiome_skeleton_min_strength": config.multiome_skeleton_min_strength,
                    "tf_target_prior_dir": None if config.tf_target_prior_dir is None else str(config.tf_target_prior_dir),
                    "tf_skeleton_alpha": config.tf_skeleton_alpha,
                    "tf_skeleton_min_strength": config.tf_skeleton_min_strength,
                    "tf_skeleton_max_conditioning_vars": config.tf_skeleton_max_conditioning_vars,
                    "tf_skeleton_strength_gamma": config.tf_skeleton_strength_gamma,
                    "atac_ci_mode": config.atac_ci_mode,
                    "atac_ci_profile_mode": config.atac_ci_profile_mode,
                    "atac_ci_open_threshold": config.atac_ci_open_threshold,
                }
                for module in summary.modules
            ],
        }
    print(json.dumps(payload, indent=2))


def command_ckm(args: argparse.Namespace) -> None:
    from .ckm import build_ckm

    ckm_df = build_ckm(
        module_dir=args.module_dir,
        result_dir=args.result_dir,
        alpha=args.alpha,
        out_path=args.out_csv,
        beta_mode=args.beta_mode,
        beta_lambda=args.beta_lambda,
        edge_weight_mode=args.edge_weight_mode,
        edge_weight_lambda=args.edge_weight_lambda,
        tf_top_k=args.tf_top_k,
    )
    payload = {
        "module_dir": str(args.module_dir),
        "result_dir": str(args.result_dir),
        "alpha": float(args.alpha),
        "beta_mode": str(args.beta_mode),
        "beta_lambda": float(args.beta_lambda),
        "edge_weight_mode": str(args.edge_weight_mode),
        "edge_weight_lambda": float(args.edge_weight_lambda),
        "tf_top_k": int(args.tf_top_k),
        "shape": [int(ckm_df.shape[0]), int(ckm_df.shape[1])],
        "out_csv": str(args.out_csv),
    }
    print(json.dumps(payload, indent=2))


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "preprocess":
        command_preprocess(args)
    elif args.command == "run":
        command_run(args)
    elif args.command == "ckm":
        command_ckm(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
