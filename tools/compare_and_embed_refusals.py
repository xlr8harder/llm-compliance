#!/usr/bin/env python3
"""
compare_and_embed_refusals.py

Phase 1: refusal-pattern similarity (Jaccard of refusal sets + BIN_AGREE, KAPPA, ACC3)
Phase 2: embedding-based similarity for top-N peers:
  - Global refusal centroid cosine vs target
  - Overlap refusal centroid cosine vs target (centroids over shared refused QIDs)
  - Pairwise mean cosine over shared refusals
  - Refusal-direction cosine vs target (mean_refuse - mean_comply)
  - NEW: Pairwise mean cosine over shared COMPLIANT responses

Features:
- Terminal-only output (no files)
- Two-space indentation, 48-char model column
- Robust handling of multiple files per model name: merges all peer files by model name
- Works with device: cpu | cuda | mps

Refusal definition: compliance in {"DENIAL", "EVASIVE"} (plus common aliases).
Response text is read from response["choices"][0]["message"]["content"] when available,
falling back to common alternatives; empty strings are allowed (encoder handles them).
"""
from __future__ import annotations

import argparse
import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np

# Optional dependency (only needed for Phase 2)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

from compliance.data import JSONLHandler, ComplianceAnalysis

# -------------------- Label handling --------------------

VALID_3CLASS = {"COMPLETE", "EVASIVE", "DENIAL"}
REFUSAL_BASE = {"DENIAL", "EVASIVE"}
REFUSAL_ALIASES = {"DENY", "DENIED", "REFUSE", "REFUSAL"}  # mapped to REFUSE in binary view

def _norm(s: str | None) -> str:
    return (s or "").strip().upper()

def to_binary(label: str | None) -> str | None:
    up = _norm(label)
    if up == "COMPLETE":
        return "COMPLY"
    if up in REFUSAL_BASE or up in REFUSAL_ALIASES:
        return "REFUSE"
    return None

def is_valid_3class(label: str | None) -> bool:
    return _norm(label) in VALID_3CLASS

# -------------------- Helpers --------------------

def load_rows(path: Path) -> List[ComplianceAnalysis]:
    return JSONLHandler.load_jsonl(path, ComplianceAnalysis) if path.exists() else []

def model_name_from_rows(rows: List[ComplianceAnalysis]) -> str:
    return rows[0].model if rows else "unknown"

def merge_rows_by_model(files: List[Path]) -> Dict[str, List[ComplianceAnalysis]]:
    """
    Merge multiple analysis files by model name (peers only).
    If a question_id appears more than once for a model, keep the last one in file order.
    """
    by_model: Dict[str, Dict[str, ComplianceAnalysis]] = {}
    for p in files:
        rows = load_rows(p)
        if not rows:
            continue
        name = model_name_from_rows(rows)
        bucket = by_model.setdefault(name, {})
        for r in rows:
            bucket[r.question_id] = r  # last write wins
    return {m: list(d.values()) for m, d in by_model.items()}

def index_labels(rows: List[ComplianceAnalysis]) -> Dict[str, str]:
    return {r.question_id: _norm(r.compliance) for r in rows}

def extract_texts_by_label(rows: List[ComplianceAnalysis]) -> Dict[str, Dict[str, str]]:
    """
    Returns {"REFUSE": {qid:text}, "COMPLY": {qid:text}}.
    Includes empty-string text if missing to preserve counts; encoder handles it.
    """
    out = {"REFUSE": {}, "COMPLY": {}}
    for r in rows:
        b = to_binary(r.compliance)
        if b is None:
            continue
        # best-effort text extraction
        text = ""
        try:
            text = r.response["choices"][0]["message"]["content"]
        except Exception:
            try:
                text = r.response.get("text", "") or r.response.get("content", "") or ""
            except Exception:
                text = ""
        out[b][r.question_id] = str(text)
    return out

def jaccard(a_true_qids: Iterable[str], b_true_qids: Iterable[str]) -> float:
    A, B = set(a_true_qids), set(b_true_qids)
    if not A and not B:
        return 1.0
    U = A | B
    return len(A & B) / len(U) if U else float("nan")

def binary_agreement(a: List[str], b: List[str]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else float("nan")

def cohen_kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    labs = ["REFUSE", "COMPLY"]; idx = {lab: i for i, lab in enumerate(labs)}
    conf = [[0, 0], [0, 0]]
    for x, y in zip(a, b):
        if x in idx and y in idx:
            conf[idx[x]][idx[y]] += 1
    total = sum(sum(r) for r in conf)
    if total == 0:
        return float("nan")
    po = (conf[0][0] + conf[1][1]) / total
    a_m = [sum(conf[i]) for i in range(2)]
    b_m = [conf[0][0] + conf[1][0], conf[0][1] + conf[1][1]]
    pe = (a_m[0] * b_m[0] + a_m[1] * b_m[1]) / (total * total)
    return (po - pe) / (1 - pe) if not math.isclose(1 - pe, 0) else float("nan")

def acc3(a: List[str], b: List[str]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else float("nan")

def fmtf(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.4f}"

def print_table(rows: List[List[str]], header: List[str]) -> None:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(header)]
    fmt = "  " + "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(*r))

# -------------------- Phase 1 --------------------

def compare_pair(base: Dict[str, str], other: Dict[str, str]) -> Dict[str, float | int]:
    overlap = set(base) & set(other)
    bin_a: List[str] = []
    bin_b: List[str] = []
    ref_a: List[str] = []
    ref_b: List[str] = []
    tri_a: List[str] = []
    tri_b: List[str] = []

    for qid in overlap:
        a_bin, b_bin = to_binary(base[qid]), to_binary(other[qid])
        if a_bin and b_bin:
            bin_a.append(a_bin); bin_b.append(b_bin)
            if a_bin == "REFUSE": ref_a.append(qid)
            if b_bin == "REFUSE": ref_b.append(qid)
        if is_valid_3class(base[qid]) and is_valid_3class(other[qid]):
            tri_a.append(base[qid]); tri_b.append(other[qid])

    return {
        "overlap": len(overlap),
        "bin_n": len(bin_a),
        "tri_n": len(tri_a),
        "jaccard": jaccard(ref_a, ref_b),
        "bin_agree": binary_agreement(bin_a, bin_b),
        "kappa": cohen_kappa(bin_a, bin_b),
        "acc3": acc3(tri_a, tri_b),
    }

def print_phase1_table(results: List[Tuple[str, Dict[str, float | int]]]) -> None:
    header = ["RANK", "MODEL", "JACCARD", "BIN_AGREE", "KAPPA", "ACC3", "OVERLAP", "BIN_N", "TRI_N"]
    print("  " + "{:<5} {:<48} {:>8} {:>9} {:>8} {:>8} {:>8} {:>6} {:>6}".format(*header))
    for i, (model, m) in enumerate(results, 1):
        def f(x): return "n/a" if (isinstance(x, float) and math.isnan(x)) else (f"{x:.4f}" if isinstance(x, float) else str(x))
        print("  " + f"{i:<5} {model:<48} {f(m['jaccard']):>8} {f(m['bin_agree']):>9} "
                       f"{f(m['kappa']):>8} {f(m['acc3']):>8} {f(m['overlap']):>8} "
                       f"{f(m['bin_n']):>6} {f(m['tri_n']):>6}")

def phase1_rank(target_file: Path, peer_files: List[Path], min_overlap: int
) -> Tuple[str, Dict[str, List[ComplianceAnalysis]], List[Tuple[str, Dict[str, float | int]]]]:
    # Load target (single file)
    tgt_rows = load_rows(target_file)
    if not tgt_rows:
        raise SystemExit(f"No analyses found in target file: {target_file}")
    tgt_name = model_name_from_rows(tgt_rows)
    base_labels = index_labels(tgt_rows)

    # Merge peers by model name
    peer_rows_by_model = merge_rows_by_model(peer_files)

    # Compute Phase 1 metrics vs each peer model
    results: List[Tuple[str, Dict[str, float | int]]] = []
    for model, rows in sorted(peer_rows_by_model.items()):
        other_labels = index_labels(rows)
        metrics = compare_pair(base_labels, other_labels)
        if metrics["overlap"] >= min_overlap:
            results.append((model, metrics))

    results.sort(key=lambda x: (
        -(x[1]["jaccard"] if not math.isnan(x[1]["jaccard"]) else -1.0),
        -(x[1]["bin_agree"] if not math.isnan(x[1]["bin_agree"]) else -1.0),
        -(x[1]["kappa"] if not math.isnan(x[1]["kappa"]) else -1.0),
    ))
    return tgt_name, peer_rows_by_model, results

# -------------------- Phase 2 (embeddings) --------------------

def embedder(model_name: str, device: str) -> SentenceTransformer:
    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers not installed. Try: pip install sentence-transformers")
    return SentenceTransformer(model_name, device=device)

def embed_texts(encoder: SentenceTransformer, texts: List[str], batch: int) -> np.ndarray:
    if not texts:
        dummy = encoder.encode([""], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        return np.zeros((0, len(dummy[0])), dtype=np.float32)
    emb = encoder.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)

def centroid(vecs: np.ndarray) -> np.ndarray:
    if vecs.size == 0: return vecs
    c = vecs.mean(axis=0, dtype=np.float32)
    n = np.linalg.norm(c)
    return c / (n + 1e-12)

def direction(refuse_c: np.ndarray, comply_c: np.ndarray) -> np.ndarray:
    if refuse_c.size == 0 or comply_c.size == 0: return np.zeros((0,), dtype=np.float32)
    d = refuse_c - comply_c
    n = np.linalg.norm(d)
    return d / (n + 1e-12)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0: return float("nan")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def phase2_embed_analysis(
    target_file: Path,
    peer_rows_by_model: Dict[str, List[ComplianceAnalysis]],
    ranked_models: List[str],
    embed_model: str,
    device: str,
    batch_size: int,
    min_shared_refusals: int,
) -> None:
    # Build target maps
    tgt_rows = load_rows(target_file)
    tgt_name = model_name_from_rows(tgt_rows)
    tgt_parts = extract_texts_by_label(tgt_rows)
    tgt_ref_map = tgt_parts["REFUSE"]
    tgt_com_map = tgt_parts["COMPLY"]

    enc = embedder(embed_model, device=device)

    # Global centroids + direction for each model
    globals_map: Dict[str, Dict[str, np.ndarray]] = {}
    counts_map: Dict[str, Tuple[int, int]] = {}
    for model in [tgt_name] + ranked_models:
        rows = tgt_rows if model == tgt_name else peer_rows_by_model.get(model, [])
        parts = extract_texts_by_label(rows)
        ref_emb = embed_texts(enc, list(parts["REFUSE"].values()), batch_size)
        com_emb = embed_texts(enc, list(parts["COMPLY"].values()), batch_size)
        ref_c = centroid(ref_emb)
        com_c = centroid(com_emb)
        globals_map[model] = {"REF": ref_c, "COM": com_c, "DIR": direction(ref_c, com_c)}
        counts_map[model] = (len(parts["REFUSE"]), len(parts["COMPLY"]))

    tgt_ref_c = globals_map[tgt_name]["REF"]
    tgt_dir = globals_map[tgt_name]["DIR"]

    # A) Global refusal centroid cosine vs target
    rows_global: List[List[str]] = []
    for model in ranked_models:
        cval = cosine(tgt_ref_c, globals_map[model]["REF"])
        rcount, _ = counts_map[model]
        rows_global.append([model, fmtf(cval), str(rcount)])
    rows_global.sort(key=lambda r: (-(float(r[1]) if r[1] != "n/a" else -1.0)))
    print("\nGlobal refusal centroid cosine vs target:")
    print_table(rows_global, ["MODEL", "COSINE_GLOBAL_REFUSAL", "REFUSAL_COUNT"])

    # B) Overlap refusal centroid cosine + Pairwise mean cosine over shared refusals
    rows_overlap_ref: List[List[str]] = []
    rows_pairwise_ref: List[List[str]] = []

    # Pre-embed target refusals by QID for reuse
    tgt_ref_vec_by_qid: Dict[str, np.ndarray] = {}
    if tgt_ref_map:
        t_texts = [tgt_ref_map[q] for q in tgt_ref_map.keys()]
        t_vecs = embed_texts(enc, t_texts, batch_size)
        for q, v in zip(tgt_ref_map.keys(), t_vecs):
            tgt_ref_vec_by_qid[q] = v

    for model in ranked_models:
        parts = extract_texts_by_label(peer_rows_by_model.get(model, []))
        shared_qids = list(set(tgt_ref_map) & set(parts["REFUSE"]))
        n_shared = len(shared_qids)

        if n_shared >= min_shared_refusals:
            t_emb = np.vstack([tgt_ref_vec_by_qid[q] for q in shared_qids])
            o_emb = embed_texts(enc, [parts["REFUSE"][q] for q in shared_qids], batch_size)
            # overlap centroid cosine (refusals)
            cval = cosine(centroid(t_emb), centroid(o_emb))
            rows_overlap_ref.append([model, fmtf(cval), str(n_shared)])
            # pairwise mean cosine (refusals)
            sims = (t_emb * o_emb).sum(axis=1)  # normalized embeddings
            rows_pairwise_ref.append([model, f"{float(np.mean(sims)):.4f}", str(n_shared)])
        else:
            rows_overlap_ref.append([model, "n/a", str(n_shared)])
            rows_pairwise_ref.append([model, "n/a", str(n_shared)])

    rows_overlap_ref.sort(key=lambda r: (-(float(r[1]) if r[1] != "n/a" else -1.0), -int(r[2])))
    rows_pairwise_ref.sort(key=lambda r: (-(float(r[1]) if r[1] != "n/a" else -1.0), -int(r[2])))

    print("\nOverlap refusal centroid cosine vs target:")
    print_table(rows_overlap_ref, ["MODEL", "COSINE_OVERLAP_REFUSAL", "SHARED_REFUSALS"])

    print("\nPairwise mean cosine over shared refusals vs target:")
    print_table(rows_pairwise_ref, ["MODEL", "PAIRWISE_MEAN_COS_REFUSALS", "SHARED_REFUSALS"])

    # C) Refusal-direction cosine
    rows_dir: List[List[str]] = []
    for model in ranked_models:
        rcount, ccount = counts_map[model]
        cval = cosine(tgt_dir, globals_map[model]["DIR"])
        rows_dir.append([model, fmtf(cval), str(rcount), str(ccount)])
    rows_dir.sort(key=lambda r: (-(float(r[1]) if r[1] != "n/a" else -1.0)))
    print("\nRefusal-direction cosine vs target:")
    print_table(rows_dir, ["MODEL", "COSINE_REFUSAL_DIR", "REFUSAL_COUNT", "COMPLY_COUNT"])

    # D) NEW: Pairwise mean cosine over shared COMPLIANT responses
    rows_pairwise_comp: List[List[str]] = []
    # Pre-embed target complies by QID for reuse
    tgt_com_vec_by_qid: Dict[str, np.ndarray] = {}
    if tgt_com_map:
        t_texts = [tgt_com_map[q] for q in tgt_com_map.keys()]
        t_vecs = embed_texts(enc, t_texts, batch_size)
        for q, v in zip(tgt_com_map.keys(), t_vecs):
            tgt_com_vec_by_qid[q] = v

    for model in ranked_models:
        parts = extract_texts_by_label(peer_rows_by_model.get(model, []))
        shared_qids = list(set(tgt_com_map) & set(parts["COMPLY"]))
        n_shared = len(shared_qids)
        if n_shared > 0:
            t_emb = np.vstack([tgt_com_vec_by_qid[q] for q in shared_qids])
            o_emb = embed_texts(enc, [parts["COMPLY"][q] for q in shared_qids], batch_size)
            sims = (t_emb * o_emb).sum(axis=1)  # normalized embeddings
            rows_pairwise_comp.append([model, f"{float(np.mean(sims)):.4f}", str(n_shared)])
        else:
            rows_pairwise_comp.append([model, "n/a", "0"])

    rows_pairwise_comp.sort(key=lambda r: (-(float(r[1]) if r[1] != "n/a" else -1.0), -int(r[2])))
    print("\nPairwise mean cosine over shared COMPLIANT responses vs target:")
    print_table(rows_pairwise_comp, ["MODEL", "PAIRWISE_MEAN_COS_COMPLY", "SHARED_COMPLIES"])

# -------------------- CLI & Main --------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1 then Phase 2 on top-N peers.")
    p.add_argument("target", type=Path, help="Target ComplianceAnalysis JSONL (single file).")
    p.add_argument("--others-dir", type=Path, default=Path("analysis"), help="Dir with peer compliance_*.jsonl files.")
    p.add_argument("--others-glob", type=str, default="compliance_*.jsonl", help="Glob within --others-dir for peers.")
    p.add_argument("--others-files", type=Path, nargs="*", help="Explicit peer files (overrides dir/glob).")
    p.add_argument("--min-overlap", type=int, default=20, help="Minimum overlapping QIDs for Phase 1.")
    p.add_argument("--top-n", type=int, default=10, help="Top-N peers (by Phase 1 Jaccard) for Phase 2.")
    p.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder name.")
    p.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    p.add_argument("--min-shared-refusals", type=int, default=20, help="Minimum shared refusals for overlap metrics.")
    return p

def discover_peer_files(args: argparse.Namespace) -> List[Path]:
    if args.others_files:
        paths = [Path(p) for p in args.others_files]
    else:
        paths = [Path(p) for p in glob.glob(str(args.others_dir / args.others_glob))]
    # Exclude the target file itself if it resides next to peers
    return [p for p in paths if p.resolve() != args.target.resolve()]

def main() -> None:
    args = build_arg_parser().parse_args()
    peer_files = discover_peer_files(args)

    # Phase 1
    tgt_name, peer_rows_by_model, results = phase1_rank(args.target, peer_files, args.min_overlap)
    if not results:
        print("No comparable peers met the minimum overlap threshold.")
        return

    print("\nPhase 1 – Refusal-pattern similarity (ranked by Jaccard):")
    print_phase1_table(results)

    # Top-N model names (in order)
    top_models = [name for name, _ in results[:args.top_n]]

    print("\nPhase 2 – Embedding-based similarity (top-N from Phase 1):")
    if SentenceTransformer is None:
        print("  sentence-transformers not installed. Skipping embedding parts.\n"
              "  Install with: pip install sentence-transformers")
        return

    phase2_embed_analysis(
        target_file=args.target,
        peer_rows_by_model=peer_rows_by_model,
        ranked_models=top_models,
        embed_model=args.embed_model,
        device=args.device,
        batch_size=args.batch_size,
        min_shared_refusals=args.min_shared_refusals,
    )

if __name__ == "__main__":
    main()

