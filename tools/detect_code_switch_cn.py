#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detect Chinese (CJK Han) code-switching in JSONL model outputs.

Usage:
    python detect_code_switch_cn.py input.jsonl [--csv report.csv]

Behavior:
- Parses the JSONL file line-by-line.
- Recursively inspects ALL string fields in each JSON object.
- Flags any occurrence of:
    * Han characters (CJK Unified Ideographs and extensions)
    * Common CJK punctuation and fullwidth forms (helps catch “，。、（）《》” etc.)
- Prints per-record findings with context snippets.
- Optionally writes a CSV of all hits with: line_no, question_id, field_path, match_text, snippet.

This script is encoding-safe (assumes UTF-8 input) and robust to large files.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Tuple

# Detect Han characters (CJK Unified Ideographs and major extensions) + common CJK punctuation
# Han ranges:
#   U+3400–U+4DBF   CJK Unified Ideographs Extension A
#   U+4E00–U+9FFF   CJK Unified Ideographs
#   U+F900–U+FAFF   CJK Compatibility Ideographs
#   U+20000–U+2A6DF Extension B
#   U+2A700–U+2B73F Extension C
#   U+2B740–U+2B81F Extension D
#   U+2B820–U+2CEAF Extension E
#   U+2CEB0–U+2EBEF Extension F
#   U+30000–U+3134F Extension G
#   U+31350–U+323AF Extension H (partial upper bound to keep pattern compact)
#
# Punctuation ranges:
#   U+3000–U+303F   CJK Symbols and Punctuation
#   U+FF00–U+FFEF   Halfwidth and Fullwidth Forms
#
# Notes:
# - This focuses on Chinese detection by the presence of Han characters or CJK punctuation.
# - Japanese text with Kanji will also trigger; if you need strict “Chinese only,” language ID would be required.
HAN_AND_CJK_PUNCT_PATTERN = re.compile(
    "["  # start character class
    "\u3400-\u4DBF"
    "\u4E00-\u9FFF"
    "\uF900-\uFAFF"
    "\U00020000-\U0002EBEF"
    "\u3000-\u303F"
    "\uFF00-\uFFEF"
    "]"
)

# When showing context around a hit, how many characters to include on each side
CONTEXT_CHARS = 60


def iter_strings(obj: Any, path: str = "") -> Iterable[Tuple[str, str]]:
    """
    Yield (field_path, string_value) for every string found in the JSON object, recursively.
    field_path uses dot/bracket notation, e.g. "response.choices[0].message.content".
    """
    if isinstance(obj, str):
        yield (path or "$", obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            next_path = f"{path}.{k}" if path else k
            yield from iter_strings(v, next_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            next_path = f"{path}[{i}]" if path else f"[{i}]"
            yield from iter_strings(v, next_path)
    # Ignore other types (int, float, bool, None)


def _merge_adjacent_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping/adjacent spans into consolidated ranges.
    """
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # overlap or adjacency
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def find_cjk_spans(text: str) -> List[Tuple[int, int]]:
    """
    Return merged spans (start, end) where Chinese/CJK punctuation occurs in text.
    """
    hits = [(m.start(), m.end()) for m in HAN_AND_CJK_PUNCT_PATTERN.finditer(text)]
    return _merge_adjacent_spans(hits)


def make_snippet(text: str, start: int, end: int, ctx: int = CONTEXT_CHARS) -> str:
    """
    Build a human-readable snippet around the [start:end] span, with ellipses if truncated.
    """
    left = max(0, start - ctx)
    right = min(len(text), end + ctx)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return f"{prefix}{text[left:right]}{suffix}"


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    """
    Safely retrieve nested keys from a dict (no list indexing here).
    """
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def process_file(infile: io.TextIOBase, csv_writer: csv.writer = None) -> int:
    """
    Process a JSONL input stream. Print findings and optionally write CSV rows.

    Returns the number of records with at least one hit.
    """
    total_records = 0
    records_with_hits = 0
    total_hits = 0

    for line_no, raw in enumerate(infile, start=1):
        raw = raw.strip()
        if not raw:
            continue

        total_records += 1

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[Line {line_no}] JSON parse error: {e}", file=sys.stderr)
            continue

        qid = obj.get("question_id")
        model = obj.get("model")
        timestamp = obj.get("timestamp")

        # Collect hits for this record
        record_hits: List[Dict[str, Any]] = []

        for field_path, text in iter_strings(obj):
            spans = find_cjk_spans(text)
            if not spans:
                continue
            for start, end in spans:
                snippet = make_snippet(text, start, end)
                matched = text[start:end]
                record_hits.append({
                    "field_path": field_path,
                    "match_text": matched,
                    "snippet": snippet
                })

        if record_hits:
            records_with_hits += 1
            total_hits += len(record_hits)

            header = f"[Line {line_no}] question_id={qid!r} model={model!r} timestamp={timestamp!r}"
            print(header)
            print("-" * len(header))
            for i, h in enumerate(record_hits, start=1):
                print(f"Hit {i}: field={h['field_path']}")
                print(f"  Match: {h['match_text']}")
                print(f"  Snip : {h['snippet']}")
                print()
                if csv_writer is not None:
                    csv_writer.writerow([
                        line_no,
                        qid if qid is not None else "",
                        model if model is not None else "",
                        timestamp if timestamp is not None else "",
                        h["field_path"],
                        h["match_text"],
                        h["snippet"],
                    ])
            print()

    print("Summary")
    print("-------")
    print(f"Total records processed : {total_records}")
    print(f"Records with CJK hits   : {records_with_hits}")
    print(f"Total hit spans found   : {total_hits}")

    return records_with_hits


def main():
    parser = argparse.ArgumentParser(description="Detect Chinese code-switching in JSONL.")
    parser.add_argument("input", help="Path to JSONL file (UTF-8). Use '-' for stdin.")
    parser.add_argument("--csv", help="Optional path to write a CSV report.")
    args = parser.parse_args()

    # Open input
    if args.input == "-":
        infile = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    else:
        infile = open(args.input, "r", encoding="utf-8")

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "line_no",
            "question_id",
            "model",
            "timestamp",
            "field_path",
            "match_text",
            "snippet",
        ])

    try:
        process_file(infile, csv_writer)
    finally:
        if args.input != "-":
            infile.close()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()
