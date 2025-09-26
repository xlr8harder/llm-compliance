#!/usr/bin/env python3
"""
compare_refusals.py – rank other models by similarity of refusal patterns.

This is the same as before (Jaccard on refusal sets, etc.) but the output
is formatted into neat columns for readability.
"""
from __future__ import annotations
import argparse, glob, math
from pathlib import Path
from typing import Dict, List, Tuple
from compliance.data import JSONLHandler, ComplianceAnalysis

VALID_3CLASS = {"COMPLETE", "EVASIVE", "DENIAL"}
REFUSAL_SET = {"DENIAL", "EVASIVE"}

def to_binary(label: str) -> str | None:
    up = (label or "").upper()
    if up == "COMPLETE": return "COMPLY"
    if up in REFUSAL_SET: return "REFUSE"
    return None

def is_valid_3class(label: str) -> bool:
    return (label or "").upper() in VALID_3CLASS

def jaccard(a_true: List[str], b_true: List[str]) -> float:
    A, B = set(a_true), set(b_true)
    if not A and not B: return 1.0
    return len(A & B) / len(A | B) if (A | B) else float("nan")

def binary_agreement(a: List[str], b: List[str]) -> float:
    if not a: return float("nan")
    return sum(x==y for x,y in zip(a,b)) / len(a)

def cohen_kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    if n == 0: return float("nan")
    labs = ["REFUSE","COMPLY"]; idx = {lab:i for i,lab in enumerate(labs)}
    conf = [[0,0],[0,0]]
    for x,y in zip(a,b):
        if x in idx and y in idx: conf[idx[x]][idx[y]] += 1
    total = sum(sum(r) for r in conf)
    if total == 0: return float("nan")
    po = (conf[0][0] + conf[1][1]) / total
    a_m = [sum(conf[i]) for i in range(2)]
    b_m = [conf[0][0]+conf[1][0], conf[0][1]+conf[1][1]]
    pe = (a_m[0]*b_m[0] + a_m[1]*b_m[1]) / (total*total)
    return (po - pe) / (1-pe) if not math.isclose(1-pe,0) else float("nan")

def acc3(a: List[str], b: List[str]) -> float:
    if not a: return float("nan")
    return sum(x==y for x,y in zip(a,b)) / len(a)

def load(path: Path) -> List[ComplianceAnalysis]:
    return JSONLHandler.load_jsonl(path, ComplianceAnalysis) if path.exists() else []

def index(rows: List[ComplianceAnalysis]) -> Dict[str,str]:
    return {r.question_id:(r.compliance or "").upper() for r in rows}

def model_name(rows: List[ComplianceAnalysis]) -> str:
    return rows[0].model if rows else "unknown"

def compare(base: Dict[str,str], other: Dict[str,str]) -> Dict[str,float|int]:
    overlap = set(base) & set(other)
    bin_a, bin_b, ref_a, ref_b, tri_a, tri_b = [],[],[],[],[],[]
    for qid in overlap:
        a_bin, b_bin = to_binary(base[qid]), to_binary(other[qid])
        if a_bin and b_bin:
            bin_a.append(a_bin); bin_b.append(b_bin)
            if a_bin=="REFUSE": ref_a.append(qid)
            if b_bin=="REFUSE": ref_b.append(qid)
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

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("target", type=Path)
    p.add_argument("--others-dir", type=Path, default=Path("analysis"))
    p.add_argument("--others-glob", type=str, default="compliance_*.jsonl")
    p.add_argument("--others-files", type=Path, nargs="*")
    p.add_argument("--min-overlap", type=int, default=20)
    p.add_argument("--top-k", type=int, default=50)
    return p

def main():
    args = build_arg_parser().parse_args()
    target_rows = load(args.target)
    if not target_rows: raise SystemExit("No analyses in target.")
    base = index(target_rows)

    if args.others_files: paths = [Path(p) for p in args.others_files]
    else: paths = [Path(p) for p in glob.glob(str(args.others_dir/args.others_glob))]
    paths = [p for p in paths if p.resolve()!=args.target.resolve()]

    results = []
    for path in sorted(paths):
        rows = load(path)
        if not rows: continue
        metrics = compare(base, index(rows))
        if metrics["overlap"] >= args.min_overlap:
            results.append((model_name(rows), metrics))

    results.sort(key=lambda x: (
        -(x[1]["jaccard"] if not math.isnan(x[1]["jaccard"]) else -1),
        -(x[1]["bin_agree"] if not math.isnan(x[1]["bin_agree"]) else -1),
        -(x[1]["kappa"] if not math.isnan(x[1]["kappa"]) else -1),
    ))

    # Pretty table
    header = ["RANK","MODEL","JACCARD","BIN_AGREE","KAPPA","ACC3","OVERLAP","BIN_N","TRI_N"]
    print(" {:<5} {:<48} {:>8} {:>9} {:>8} {:>8} {:>8} {:>6} {:>6}".format(*header))
    for i,(model,m) in enumerate(results[:args.top_k],1):
        def f(x): return "nan" if math.isnan(x) else f"{x:.4f}" if not isinstance(x,int) else str(x)
        print(f" {i:<5} {model:<48} {f(m['jaccard']):>8} {f(m['bin_agree']):>9} "
              f"{f(m['kappa']):>8} {f(m['acc3']):>8} {f(m['overlap']):>8} "
              f"{f(m['bin_n']):>6} {f(m['tri_n']):>6}")

if __name__=="__main__":
    main()

