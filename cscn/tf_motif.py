from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, TYPE_CHECKING

if TYPE_CHECKING:  # imported lazily at call sites so the core pipeline runs without these extras
    from pyfaidx import Fasta


JASPAR_RELEASE = "2026"
JASPAR_CORE_VERTEBRATES_URL = (
    f"https://jaspar.elixir.no/download/data/{JASPAR_RELEASE}/CORE/"
    f"JASPAR{JASPAR_RELEASE}_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
)
JASPAR_FILENAME = f"JASPAR{JASPAR_RELEASE}_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
DEFAULT_MOTIF_REL_SCORE_THRESHOLD = 0.85


def canonical_gene_token(value: object) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    return token.split(".", 1)[0].upper()


def load_tf_tokens(tf_list_path: Optional[Path]) -> Set[str]:
    if tf_list_path is None:
        return set()
    tokens: Set[str] = set()
    with Path(tf_list_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            token = canonical_gene_token(raw.split("\t", 1)[0])
            if token:
                tokens.add(token)
    return tokens


def map_tf_tokens_to_symbols(tf_tokens: Iterable[str], alias_to_symbol: Mapping[str, str]) -> Set[str]:
    symbols: Set[str] = set()
    for token in tf_tokens:
        canonical = canonical_gene_token(token)
        if not canonical:
            continue
        symbol = alias_to_symbol.get(canonical)
        if symbol:
            symbols.add(canonical_gene_token(symbol))
    return symbols


def ensure_jaspar_cache(cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jaspar_path = cache_dir / JASPAR_FILENAME
    if jaspar_path.exists() and jaspar_path.stat().st_size > 0:
        return jaspar_path
    tmp_path = jaspar_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(JASPAR_CORE_VERTEBRATES_URL, tmp_path)
    except Exception as exc:  # pragma: no cover - network path is hard to test
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Unable to download pinned JASPAR motif file from {JASPAR_CORE_VERTEBRATES_URL}. "
            "Provide network access or populate the cache file manually."
        ) from exc
    tmp_path.replace(jaspar_path)
    return jaspar_path


@dataclass(frozen=True)
class MotifRecord:
    tf_symbol: str
    motif_id: str
    motif_name: str
    pssm: object
    min_score: float
    max_score: float


@dataclass(frozen=True)
class PeakMotifHit:
    tf_symbol: str
    motif_id: str
    motif_name: str
    rel_score: float
    abs_score: float
    position: int
    strand: str


def _motif_name_tokens(name: str) -> List[str]:
    clean = str(name or "").strip()
    if not clean:
        return []
    pieces = []
    for chunk in clean.split("::"):
        token = re.split(r"[\s(,;/]+", chunk.strip(), maxsplit=1)[0]
        token = canonical_gene_token(token)
        if token:
            pieces.append(token)
    return pieces


def load_jaspar_motif_records(
    jaspar_path: Path,
    tf_symbols: Optional[Iterable[str]] = None,
) -> Dict[str, List[MotifRecord]]:
    symbol_lookup = None
    if tf_symbols is not None:
        symbol_lookup = {canonical_gene_token(symbol): canonical_gene_token(symbol) for symbol in tf_symbols if canonical_gene_token(symbol)}
        if not symbol_lookup:
            return {}
    from Bio import motifs  # optional dependency: only needed for the TF-motif prior

    records_by_symbol: Dict[str, List[MotifRecord]] = {}
    with Path(jaspar_path).open("r", encoding="utf-8") as handle:
        for motif in motifs.parse(handle, "jaspar"):
            matched_symbols = []
            for token in _motif_name_tokens(getattr(motif, "name", "")):
                if symbol_lookup is None:
                    if token not in matched_symbols:
                        matched_symbols.append(token)
                else:
                    symbol = symbol_lookup.get(token)
                    if symbol is not None and symbol not in matched_symbols:
                        matched_symbols.append(symbol)
            if not matched_symbols:
                continue
            pssm = motif.counts.normalize(pseudocounts=0.5).log_odds()
            min_score = float(pssm.min)
            max_score = float(pssm.max)
            motif_id = str(getattr(motif, "matrix_id", "") or getattr(motif, "name", ""))
            motif_name = str(getattr(motif, "name", motif_id))
            for symbol in matched_symbols:
                records_by_symbol.setdefault(symbol, [])
                records_by_symbol[symbol].append(
                    MotifRecord(
                        tf_symbol=str(symbol),
                        motif_id=motif_id,
                        motif_name=motif_name,
                        pssm=pssm,
                        min_score=min_score,
                        max_score=max_score,
                    )
                )
    return {symbol: rows for symbol, rows in records_by_symbol.items() if rows}


def _resolve_fasta_chrom(chrom: str, fasta: "Fasta") -> str:
    if chrom in fasta:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in fasta:
        return chrom[3:]
    alt = f"chr{chrom}"
    if not chrom.startswith("chr") and alt in fasta:
        return alt
    raise KeyError(f"Chromosome {chrom!r} not found in genome FASTA")


def load_peak_sequences(
    genome_fasta: Path,
    peak_intervals: Mapping[str, object],
) -> Dict[str, str]:
    from pyfaidx import Fasta  # optional dependency: only needed for the TF-motif prior

    fasta = Fasta(str(genome_fasta), as_raw=True, sequence_always_upper=True)
    sequences: Dict[str, str] = {}
    try:
        for peak_name, interval in peak_intervals.items():
            if interval is None:
                continue
            chrom = _resolve_fasta_chrom(str(interval.chrom), fasta)
            start = max(int(interval.start) - 1, 0)
            end = max(int(interval.end), start + 1)
            sequence = str(fasta[chrom][start:end])
            if sequence:
                sequences[str(peak_name)] = sequence.upper()
    finally:
        fasta.close()
    return sequences


def scan_peak_sequence_for_tf_hits(
    sequence: str,
    motif_records_by_symbol: Mapping[str, Sequence[MotifRecord]],
    threshold_rel: float = DEFAULT_MOTIF_REL_SCORE_THRESHOLD,
) -> Dict[str, PeakMotifHit]:
    sequence = str(sequence or "").upper()
    if not sequence:
        return {}
    hits: Dict[str, PeakMotifHit] = {}
    for tf_symbol, records in motif_records_by_symbol.items():
        best_hit: Optional[PeakMotifHit] = None
        for record in records:
            span = record.max_score - record.min_score
            threshold_abs = record.max_score if span <= 1e-8 else (record.min_score + (float(threshold_rel) * span))
            for position, abs_score in record.pssm.search(sequence, threshold=threshold_abs):
                rel_score = 1.0 if span <= 1e-8 else (float(abs_score) - record.min_score) / span
                candidate = PeakMotifHit(
                    tf_symbol=tf_symbol,
                    motif_id=record.motif_id,
                    motif_name=record.motif_name,
                    rel_score=float(rel_score),
                    abs_score=float(abs_score),
                    position=abs(int(position)),
                    strand="-" if int(position) < 0 else "+",
                )
                if best_hit is None or candidate.rel_score > best_hit.rel_score:
                    best_hit = candidate
        if best_hit is not None:
            hits[tf_symbol] = best_hit
    return hits
