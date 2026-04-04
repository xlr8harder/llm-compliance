from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


REPORT_DIR = Path("judge_evaluation/reports")
SUMMARY_GLOB = "*__compliance_us_hard_sample_responses.summary.json"
SAMPLE_SUMMARY = Path("judge_evaluation/us_hard_sample_summary.json")
RESULTS_DIR = Path("judge_evaluation/results")
OUT_DIR = Path("judge_evaluation/charts/gpt4o_gold")
AUDIT_SUMMARY = Path("judge_evaluation/reports/gold_audit.summary.json")
AUDIT_REVIEW_V4 = Path("judge_evaluation/reports/gold_audit_review_v4.jsonl")

LABEL_ORDER = ["COMPLETE", "DENIAL", "EVASIVE", "ERROR_JUDGE_FORMAT"]
BUCKET_ORDER = ["COMPLETE", "DENIAL", "EVASIVE", "HARD"]
TRANSITION_ORDER = [
    "COMPLETE->EVASIVE",
    "DENIAL->EVASIVE",
    "EVASIVE->COMPLETE",
    "EVASIVE->DENIAL",
    "COMPLETE->DENIAL",
    "DENIAL->COMPLETE",
    "OTHER",
]

COLORS = {
    "COMPLETE": "#2E8B57",
    "DENIAL": "#C44E52",
    "EVASIVE": "#D4A017",
    "ERROR_JUDGE_FORMAT": "#7A7A7A",
    "gpt4o": "#4C72B0",
    "OTHER": "#999999",
}

DISPLAY_NAMES = {
    "openai_gpt-5.4_no_reasoning": "GPT-5.4\nno reasoning",
    "openai_gpt-5.4_reasoning_medium": "GPT-5.4\nreasoning",
    "openai_gpt-5.4-mini_no_reasoning": "GPT-5.4-mini\nno reasoning",
    "openai_gpt-5.4-mini_reasoning": "GPT-5.4-mini\nreasoning",
    "openai_gpt-5.4-nano_no_reasoning": "GPT-5.4-nano\nno reasoning",
    "openai_gpt-5.4-nano_reasoning": "GPT-5.4-nano\nreasoning",
    "x-ai_grok-4.1-fast_no_reasoning": "Grok-4.1-Fast\nno reasoning",
    "x-ai_grok-4.1-fast_reasoning_medium": "Grok-4.1-Fast\nreasoning",
    "openai_gpt-5-mini": "GPT-5-mini\n(reasoning)",
    "openai_gpt-5-nano": "GPT-5-nano\n(reasoning)",
    "qwen_qwen3.5-9b_together_no_reasoning": "Qwen3.5-9B\nTogether no reasoning",
}


def load_sample_summary() -> dict:
    return json.loads(SAMPLE_SUMMARY.read_text())


def load_manifest_records() -> list[dict]:
    records = []
    with Path("judge_evaluation/us_hard_sample_manifest.jsonl").open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_audit_review_rows() -> list[dict]:
    rows = []
    with AUDIT_REVIEW_V4.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def summary_key(path: Path) -> str:
    suffix = "__compliance_us_hard_sample_responses.summary"
    return path.stem.replace(suffix, "")


def display_name(key: str) -> str:
    return DISPLAY_NAMES.get(key, key.replace("_", "\n"))


def predicted_counts(confusion: dict) -> Counter:
    counts: Counter = Counter()
    for predicted_row in confusion.values():
        for predicted_label, count in predicted_row.items():
            counts[predicted_label] += count
    return counts


def disagreement_breakdown(confusion: dict) -> Counter:
    counts: Counter = Counter()
    for gold_label, predicted_row in confusion.items():
        for predicted_label, count in predicted_row.items():
            if gold_label == predicted_label or count == 0:
                continue
            key = f"{gold_label}->{predicted_label}"
            if key not in TRANSITION_ORDER:
                key = "OTHER"
            counts[key] += count
    return counts


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(REPORT_DIR.glob(SUMMARY_GLOB)):
        key = summary_key(path)
        data = json.loads(path.read_text())
        rows.append(
            {
                "key": key,
                "name": display_name(key),
                "accuracy_pct": data["accuracy_pct"],
                "accuracy_by_bucket": data["accuracy_by_bucket"],
                "confusion": data["confusion"],
                "predicted_counts": predicted_counts(data["confusion"]),
                "transition_counts": disagreement_breakdown(data["confusion"]),
            }
        )
    rows.sort(key=lambda row: row["accuracy_pct"], reverse=True)
    return rows


def load_result_labels(result_key: str) -> dict[tuple[str, str], str]:
    path = RESULTS_DIR / result_key / "compliance_us_hard_sample_responses.jsonl"
    labels = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            labels[(row["question_id"], row["model"])] = row["compliance"]
    return labels


def load_result_labels_by_key(result_key: str) -> dict[str, str]:
    path = RESULTS_DIR / result_key / "compliance_us_hard_sample_responses.jsonl"
    labels = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            labels[f"{row['model']}::{row['question_id']}"] = row["compliance"]
    return labels


def load_baseline_labels() -> dict[tuple[str, str], str]:
    path = Path("judge_evaluation/us_hard_sample_manifest.jsonl")
    labels = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            labels[(row["question_id"], row["model"])] = row["expected_compliance"]
    return labels


def load_baseline_labels_by_key() -> dict[str, str]:
    labels = {}
    with Path("judge_evaluation/us_hard_sample_manifest.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            labels[f"{row['model']}::{row['question_id']}"] = row["expected_compliance"]
    return labels


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overall_agreement(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    names = [row["name"].replace("\n", " ") for row in rows]
    values = [row["accuracy_pct"] for row in rows]
    y = list(range(len(names)))
    bars = ax.barh(y, values, color=COLORS["gpt4o"])
    ax.set_yticks(y, names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Agreement with original GPT-4o gold (%)")
    ax.set_title("Judge Agreement vs Original GPT-4o Gold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            value + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
        )
    save(fig, "overall_agreement.png")


def plot_bucket_accuracy(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    names = [row["name"] for row in rows]
    for ax, bucket in zip(axes.flat, BUCKET_ORDER):
        values = [row["accuracy_by_bucket"][bucket]["accuracy_pct"] for row in rows]
        bars = ax.bar(names, values, color=COLORS.get(bucket, "#4C72B0"))
        ax.set_ylim(0, 100)
        ax.set_title(bucket)
        style_ax(ax)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0, 0].set_ylabel("Bucket accuracy (%)")
    axes[1, 0].set_ylabel("Bucket accuracy (%)")
    fig.suptitle("Agreement by Bucket vs Original GPT-4o Gold", y=1.02)
    save(fig, "bucket_accuracy.png")


def plot_label_distribution(rows: list[dict], sample_summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    baseline_counts = sample_summary["expected_label_counts"].copy()
    baseline_counts["ERROR_JUDGE_FORMAT"] = 0
    ordered = [{"name": "Original GPT-4o\ngold", "counts": baseline_counts}] + [
        {"name": row["name"], "counts": row["predicted_counts"]} for row in rows
    ]
    names = [row["name"].replace("\n", " ") for row in ordered]
    y = list(range(len(ordered)))
    lefts = [0] * len(ordered)
    for label in LABEL_ORDER:
        values = [row["counts"].get(label, 0) for row in ordered]
        ax.barh(y, values, left=lefts, color=COLORS[label], label=label)
        lefts = [l + v for l, v in zip(lefts, values)]
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 400)
    ax.set_xlabel("Rows")
    ax.set_title("Aggregate Label Totals on the 400-row Gold Sample")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    save(fig, "label_distribution.png")


def plot_transition_breakdown(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.4))
    names = [row["name"] for row in rows]
    bottoms = [0] * len(rows)
    transition_colors = {
        "COMPLETE->EVASIVE": "#DD8452",
        "DENIAL->EVASIVE": "#55A868",
        "EVASIVE->COMPLETE": "#4C72B0",
        "EVASIVE->DENIAL": "#8172B2",
        "COMPLETE->DENIAL": "#937860",
        "DENIAL->COMPLETE": "#64B5CD",
        "OTHER": COLORS["OTHER"],
    }
    for transition in TRANSITION_ORDER:
        values = [row["transition_counts"].get(transition, 0) for row in rows]
        ax.bar(
            names,
            values,
            bottom=bottoms,
            label=transition,
            color=transition_colors[transition],
        )
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylabel("Disagreement rows")
    ax.set_title("How Each Judge Disagrees with Original GPT-4o Gold")
    style_ax(ax)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "disagreement_transitions.png")


def plot_pairwise_agreement_heatmap(rows: list[dict]) -> tuple[list[str], list[list[float]]]:
    baseline_key = "original_gpt4o_gold"
    baseline_name = "Original GPT-4o gold\n(baseline)"
    judge_labels = {baseline_key: load_baseline_labels()}
    judge_labels.update({row["key"]: load_result_labels(row["key"]) for row in rows})
    keys = [baseline_key] + [row["key"] for row in rows]
    names = [baseline_name] + [row["name"] for row in rows]
    matrix: list[list[float]] = []
    for key_a in keys:
        row_values = []
        labels_a = judge_labels[key_a]
        for key_b in keys:
            labels_b = judge_labels[key_b]
            shared = sorted(set(labels_a) & set(labels_b))
            matches = sum(1 for item in shared if labels_a[item] == labels_b[item])
            pct = 100.0 * matches / len(shared) if shared else 0.0
            row_values.append(pct)
        matrix.append(row_values)

    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=60, vmax=100)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Pairwise Agreement Between Judges and the GPT-4o Baseline")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement (%)")
    for i, row_values in enumerate(matrix):
        for j, value in enumerate(row_values):
            text_color = "white" if value >= 82 else "#1e1e1e"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=text_color)
    ax.axhline(0.5, color="white", linewidth=2.2)
    ax.axvline(0.5, color="white", linewidth=2.2)
    save(fig, "pairwise_agreement_heatmap.png")
    return names, matrix


def plot_pairwise_agreement_heatmap_revised_gold(
    rows: list[dict], revised_by_key: dict[str, str]
) -> tuple[list[str], list[list[float]]]:
    baseline_key = "revised_audited_gold"
    baseline_name = "Revised audited gold\n(baseline)"
    judge_labels = {baseline_key: {tuple(k.split("::", 1)[::-1]): v for k, v in revised_by_key.items()}}
    judge_labels.update({row["key"]: load_result_labels(row["key"]) for row in rows})
    keys = [baseline_key] + [row["key"] for row in rows]
    names = [baseline_name] + [row["name"] for row in rows]
    matrix: list[list[float]] = []
    for key_a in keys:
        row_values = []
        labels_a = judge_labels[key_a]
        for key_b in keys:
            labels_b = judge_labels[key_b]
            shared = sorted(set(labels_a) & set(labels_b))
            matches = sum(1 for item in shared if labels_a[item] == labels_b[item])
            pct = 100.0 * matches / len(shared) if shared else 0.0
            row_values.append(pct)
        matrix.append(row_values)

    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=60, vmax=100)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Pairwise Agreement Between Judges and the Revised Gold Baseline")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement (%)")
    for i, row_values in enumerate(matrix):
        for j, value in enumerate(row_values):
            text_color = "white" if value >= 82 else "#1e1e1e"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=text_color)
    ax.axhline(0.5, color="white", linewidth=2.2)
    ax.axvline(0.5, color="white", linewidth=2.2)
    save(fig, "pairwise_agreement_heatmap_revised_gold.png")
    return names, matrix


def build_revised_gold(records: list[dict], review_rows: list[dict]) -> dict:
    revised_by_key = {f"{row['model']}::{row['question_id']}": row["expected_compliance"] for row in records}
    status_counts = Counter()
    status_by_bucket = Counter()
    for row in review_rows:
        status = row["audit_status"]
        status_counts[status] += 1
        status_by_bucket[(row["bucket"], status)] += 1
        if status == "likely_gold_wrong":
            revised_by_key[row["key"]] = row["proposed_label"]

    original_counts = Counter()
    revised_counts = Counter()
    transition_counts = Counter()
    for row in records:
        key = f"{row['model']}::{row['question_id']}"
        original = row["expected_compliance"]
        revised = revised_by_key[key]
        original_counts[original] += 1
        revised_counts[revised] += 1
        transition_counts[(original, revised)] += 1

    return {
        "revised_by_key": revised_by_key,
        "status_counts": dict(status_counts),
        "status_by_bucket": {
            f"{bucket}:{status}": count for (bucket, status), count in sorted(status_by_bucket.items())
        },
        "original_counts": dict(original_counts),
        "revised_counts": dict(revised_counts),
        "transition_counts": {
            f"{src}->{dst}": count for (src, dst), count in sorted(transition_counts.items())
        },
    }


def plot_gold_revision_process(review_rows: list[dict]) -> None:
    counts = Counter(row["audit_status"] for row in review_rows)
    labels = ["400 sample rows", "60 consensus\ncandidates", "53 likely\nwrong", "6 kept", "1 ambiguous"]
    values = [400, len(review_rows), counts.get("likely_gold_wrong", 0), counts.get("gold_plausible", 0), counts.get("ambiguous", 0)]
    colors = ["#4C72B0", "#8172B2", "#C44E52", "#55A868", "#DD8452"]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Rows")
    ax.set_title("How the Revised Gold Was Constructed")
    style_ax(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 5, str(value), ha="center", va="bottom", fontsize=10)
    save(fig, "gold_revision_process.png")


def plot_gold_revision_label_totals(revision: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    labels = ["COMPLETE", "DENIAL", "EVASIVE"]
    original = [revision["original_counts"].get(label, 0) for label in labels]
    revised = [revision["revised_counts"].get(label, 0) for label in labels]
    x = list(range(len(labels)))
    width = 0.36

    bars1 = ax.bar([i - width / 2 for i in x], original, width, label="Original GPT-4o gold", color="#4C72B0")
    bars2 = ax.bar([i + width / 2 for i in x], revised, width, label="Revised audited gold", color="#C44E52")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Rows")
    ax.set_title("Original vs Revised Gold Label Totals")
    style_ax(ax)
    ax.legend(frameon=False)
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)
    save(fig, "gold_revision_label_totals.png")


def plot_label_distribution_revised_gold(rows: list[dict], revision: dict) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    baseline_counts = revision["revised_counts"].copy()
    baseline_counts["ERROR_JUDGE_FORMAT"] = 0
    ordered = [{"name": "Revised audited\ngold", "counts": baseline_counts}] + [
        {"name": row["name"], "counts": row["predicted_counts"]} for row in rows
    ]
    names = [row["name"].replace("\n", " ") for row in ordered]
    y = list(range(len(ordered)))
    lefts = [0] * len(ordered)
    for label in LABEL_ORDER:
        values = [row["counts"].get(label, 0) for row in ordered]
        ax.barh(y, values, left=lefts, color=COLORS[label], label=label)
        lefts = [l + v for l, v in zip(lefts, values)]
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 400)
    ax.set_xlabel("Rows")
    ax.set_title("Aggregate Label Totals Against the Revised Gold Set")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    save(fig, "label_distribution_revised_gold.png")


def build_divergence_rows(rows: list[dict], sample_summary: dict, revision: dict) -> list[dict]:
    original_gold = sample_summary["expected_label_counts"]
    revised_gold = revision["revised_counts"]
    label_names = ["COMPLETE", "DENIAL", "EVASIVE"]
    divergence_rows = []
    for row in rows:
        entry = {"name": row["name"], "key": row["key"]}
        for label in label_names:
            pred = row["predicted_counts"].get(label, 0)
            orig_signed = pred - original_gold.get(label, 0)
            rev_signed = pred - revised_gold.get(label, 0)
            entry[f"{label}_signed_original"] = orig_signed
            entry[f"{label}_signed_revised"] = rev_signed
            entry[f"{label}_abs_improvement"] = abs(orig_signed) - abs(rev_signed)
        divergence_rows.append(entry)
    return divergence_rows


def build_row_level_shift_rows(rows: list[dict], revised_by_key: dict[str, str]) -> list[dict]:
    baseline_labels = load_baseline_labels_by_key()
    shift_rows = []
    for row in rows:
        labels = load_result_labels_by_key(row["key"])
        shared = sorted(set(labels) & set(baseline_labels) & set(revised_by_key))
        newly_correct = 0
        newly_wrong = 0
        for key in shared:
            orig_correct = labels[key] == baseline_labels[key]
            revised_correct = labels[key] == revised_by_key[key]
            if not orig_correct and revised_correct:
                newly_correct += 1
            elif orig_correct and not revised_correct:
                newly_wrong += 1
        net_rows = newly_correct - newly_wrong
        pp_lift = round((net_rows / len(shared)) * 100.0, 3) if shared else 0.0
        shift_rows.append(
            {
                "key": row["key"],
                "name": row["name"],
                "newly_correct": newly_correct,
                "newly_wrong": newly_wrong,
                "net_rows": net_rows,
                "pp_lift": pp_lift,
            }
        )
    return shift_rows


def build_aggregate_delta_rows(rows: list[dict], sample_summary: dict) -> list[dict]:
    total = sum(sample_summary["expected_label_counts"].values())
    baseline = {
        label: (sample_summary["expected_label_counts"].get(label, 0) / total) * 100.0
        for label in ["COMPLETE", "DENIAL", "EVASIVE"]
    }
    aggregate_rows = []
    for row in rows:
        predicted = row["predicted_counts"]
        judge_total = sum(predicted.get(label, 0) for label in ["COMPLETE", "DENIAL", "EVASIVE", "ERROR_JUDGE_FORMAT"])
        entry = {"key": row["key"], "name": row["name"]}
        total_abs = 0.0
        for label in ["COMPLETE", "DENIAL", "EVASIVE"]:
            pct = (predicted.get(label, 0) / judge_total) * 100.0 if judge_total else 0.0
            delta = round(pct - baseline[label], 3)
            entry[f"{label}_pp_delta"] = delta
            total_abs += abs(delta)
        entry["total_abs_pp_delta"] = round(total_abs, 3)
        aggregate_rows.append(entry)
    return aggregate_rows


def plot_signed_divergence_dumbbells(divergence_rows: list[dict]) -> None:
    labels = ["COMPLETE", "DENIAL", "EVASIVE"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 8.0), sharey=True)
    names = [row["name"].replace("\n", " ") for row in divergence_rows]
    y = list(range(len(divergence_rows)))
    for ax, label in zip(axes, labels):
        for idx, row in enumerate(divergence_rows):
            x1 = row[f"{label}_signed_original"]
            x2 = row[f"{label}_signed_revised"]
            ax.plot([x1, x2], [idx, idx], color="#B0B0B0", linewidth=2, zorder=1)
            ax.scatter(x1, idx, color="#4C72B0", s=48, zorder=2)
            ax.scatter(x2, idx, color="#C44E52", s=48, zorder=2)
        ax.axvline(0, color="#666666", linewidth=1, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Predicted count minus gold count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Judge")
    fig.suptitle("Signed Label Divergence: Original Gold vs Revised Gold", y=1.02)
    axes[-1].legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=8, label="Original gold"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markersize=8, label="Revised gold"),
        ],
        frameon=False,
        loc="lower right",
    )
    save(fig, "signed_divergence_dumbbells.png")


def plot_row_level_shift_heatmap(shift_rows: list[dict]) -> None:
    names = [row["name"] for row in shift_rows]
    left_labels = ["Newly correct", "Newly wrong", "Net rows"]
    right_labels = ["pp lift"]

    left_matrix = [[row["newly_correct"], row["newly_wrong"], row["net_rows"]] for row in shift_rows]
    right_matrix = [[row["pp_lift"]] for row in shift_rows]

    left_flat = [value for row in left_matrix for value in row]
    left_vmax = max(abs(v) for v in left_flat) if left_flat else 1
    right_flat = [value for row in right_matrix for value in row]
    right_vmax = max(abs(v) for v in right_flat) if right_flat else 1

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8.8, 8.4),
        gridspec_kw={"width_ratios": [3.3, 1.2]},
    )

    im1 = ax1.imshow(left_matrix, cmap="RdYlGn", vmin=-left_vmax, vmax=left_vmax, aspect="auto")
    ax1.set_xticks(range(len(left_labels)), left_labels)
    ax1.set_yticks(range(len(names)), names)
    ax1.set_title("Row-level shift counts")
    for i, row in enumerate(left_matrix):
        for j, value in enumerate(row):
            text_color = "white" if abs(value) >= left_vmax * 0.45 else "#1e1e1e"
            sign = "+" if value > 0 else ""
            ax1.text(j, i, f"{sign}{value}", ha="center", va="center", fontsize=8, color=text_color)

    im2 = ax2.imshow(right_matrix, cmap="RdYlGn", vmin=-right_vmax, vmax=right_vmax, aspect="auto")
    ax2.set_xticks(range(len(right_labels)), right_labels)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([])
    ax2.set_title("Accuracy lift")
    for i, row in enumerate(right_matrix):
        value = row[0]
        text_color = "white" if abs(value) >= right_vmax * 0.45 else "#1e1e1e"
        sign = "+" if value > 0 else ""
        ax2.text(0, i, f"{sign}{value:.1f}", ha="center", va="center", fontsize=8, color=text_color)

    fig.suptitle("Row-level Improvement from Original Gold to Revised Gold", y=1.02)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Rows")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.10)
    cbar2.set_label("Percentage points")
    save(fig, "absolute_divergence_improvement_heatmap.png")


def plot_aggregate_score_shift_heatmap(aggregate_rows: list[dict]) -> None:
    labels = ["COMPLETE", "DENIAL", "EVASIVE", "TOTAL |pp|"]
    names = [row["name"] for row in aggregate_rows]
    matrix = []
    for row in aggregate_rows:
        matrix.append(
            [
                row["COMPLETE_pp_delta"],
                row["DENIAL_pp_delta"],
                row["EVASIVE_pp_delta"],
                row["total_abs_pp_delta"],
            ]
        )

    signed_values = [v for row in matrix for v in row[:3]]
    signed_vmax = max(abs(v) for v in signed_values) if signed_values else 1
    total_vmax = max(row[3] for row in matrix) if matrix else 1

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8.8, 8.4),
        gridspec_kw={"width_ratios": [3.2, 1.1]},
    )

    left_matrix = [row[:3] for row in matrix]
    im1 = ax1.imshow(left_matrix, cmap="RdBu_r", vmin=-signed_vmax, vmax=signed_vmax, aspect="auto")
    ax1.set_xticks(range(3), labels[:3])
    ax1.set_yticks(range(len(names)), names)
    ax1.set_title("Aggregate label share delta vs original GPT-4o gold")
    for i, row in enumerate(left_matrix):
        for j, value in enumerate(row):
            text_color = "white" if abs(value) >= signed_vmax * 0.45 else "#1e1e1e"
            sign = "+" if value > 0 else ""
            ax1.text(j, i, f"{sign}{value:.1f}", ha="center", va="center", fontsize=8, color=text_color)

    right_matrix = [[row[3]] for row in matrix]
    im2 = ax2.imshow(right_matrix, cmap="YlGnBu", vmin=0, vmax=total_vmax, aspect="auto")
    ax2.set_xticks([0], [labels[3]])
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([])
    ax2.set_title("Overall shift")
    for i, row in enumerate(right_matrix):
        value = row[0]
        text_color = "white" if value >= total_vmax * 0.45 else "#1e1e1e"
        ax2.text(0, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=text_color)

    fig.suptitle("Aggregate Score Shifts Relative to the Original GPT-4o Gold Mix", y=1.02)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Percentage-point delta")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.10)
    cbar2.set_label("Total absolute pp shift")
    save(fig, "aggregate_score_shift_heatmap.png")


def plot_gold_revision_transition_heatmap(revision: dict) -> None:
    labels = ["COMPLETE", "DENIAL", "EVASIVE"]
    matrix = []
    for src in labels:
        row = []
        for dst in labels:
            row.append(revision["transition_counts"].get(f"{src}->{dst}", 0))
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(max(r) for r in matrix))
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Revised audited gold")
    ax.set_ylabel("Original GPT-4o gold")
    ax.set_title("Label Changes from Original to Revised Gold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rows")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            text_color = "white" if value >= 40 else "#1e1e1e"
            ax.text(j, i, str(value), ha="center", va="center", fontsize=10, color=text_color)
    save(fig, "gold_revision_transition_heatmap.png")


def plot_accuracy_original_vs_revised_gold(rows: list[dict], revised_by_key: dict[str, str]) -> list[dict]:
    baseline_labels = load_baseline_labels_by_key()
    result_labels = {row["key"]: load_result_labels_by_key(row["key"]) for row in rows}
    revised_scores = []
    for row in rows:
        labels = result_labels[row["key"]]
        shared = sorted(set(labels) & set(revised_by_key))
        correct = sum(1 for item in shared if labels[item] == revised_by_key[item])
        revised_pct = 100.0 * correct / len(shared) if shared else 0.0
        revised_scores.append({"key": row["key"], "name": row["name"], "original_accuracy_pct": row["accuracy_pct"], "revised_accuracy_pct": revised_pct})

    baseline_original = 100.0
    changed = sum(1 for key, label in revised_by_key.items() if label != baseline_labels[key])
    baseline_revised = 100.0 * (len(revised_by_key) - changed) / len(revised_by_key)
    revised_scores.insert(
        0,
        {
            "key": "original_gpt4o_gold",
            "name": "Original GPT-4o\ngold",
            "original_accuracy_pct": baseline_original,
            "revised_accuracy_pct": baseline_revised,
        },
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    y_positions = list(range(len(revised_scores)))
    for idx, row in enumerate(revised_scores):
        ax.plot(
            [row["original_accuracy_pct"], row["revised_accuracy_pct"]],
            [idx, idx],
            color="#B0B0B0",
            linewidth=2,
            zorder=1,
        )
        ax.scatter(row["original_accuracy_pct"], idx, color="#4C72B0", s=55, zorder=2)
        ax.scatter(row["revised_accuracy_pct"], idx, color="#C44E52", s=55, zorder=2)
    ax.set_yticks(y_positions, [row["name"] for row in revised_scores])
    ax.set_xlabel("Agreement (%)")
    ax.set_title("Judge Agreement Under Original vs Revised Gold")
    style_ax(ax)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=8, label="Original gold"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markersize=8, label="Revised gold"),
        ],
        frameon=False,
        loc="lower right",
    )
    save(fig, "judge_accuracy_original_vs_revised_gold.png")
    return revised_scores


def write_summary_json(
    rows: list[dict],
    sample_summary: dict,
    pairwise_names: list[str],
    pairwise_matrix: list[list[float]],
    pairwise_revised_names: list[str],
    pairwise_revised_matrix: list[list[float]],
    gold_revision: dict,
    revised_scores: list[dict],
    divergence_rows: list[dict],
    row_level_shift_rows: list[dict],
    aggregate_delta_rows: list[dict],
) -> None:
    payload = {
        "baseline_expected_label_counts": sample_summary["expected_label_counts"],
        "gold_revision": {
            "status_counts": gold_revision["status_counts"],
            "status_by_bucket": gold_revision["status_by_bucket"],
            "original_counts": gold_revision["original_counts"],
            "revised_counts": gold_revision["revised_counts"],
            "transition_counts": gold_revision["transition_counts"],
        },
        "judge_accuracy_original_vs_revised_gold": revised_scores,
        "label_divergence": divergence_rows,
        "row_level_shift": row_level_shift_rows,
        "aggregate_score_shift": aggregate_delta_rows,
        "pairwise_agreement_pct": {
            name.replace("\n", " "): {
                other_name.replace("\n", " "): round(pairwise_matrix[i][j], 3)
                for j, other_name in enumerate(pairwise_names)
            }
            for i, name in enumerate(pairwise_names)
        },
        "pairwise_agreement_pct_revised_gold": {
            name.replace("\n", " "): {
                other_name.replace("\n", " "): round(pairwise_revised_matrix[i][j], 3)
                for j, other_name in enumerate(pairwise_revised_names)
            }
            for i, name in enumerate(pairwise_revised_names)
        },
        "judges": [
            {
                "key": row["key"],
                "name": row["name"].replace("\n", " "),
                "accuracy_pct": row["accuracy_pct"],
                "accuracy_by_bucket": {
                    bucket: row["accuracy_by_bucket"][bucket]["accuracy_pct"]
                    for bucket in BUCKET_ORDER
                },
                "predicted_counts": dict(row["predicted_counts"]),
                "transition_counts": dict(row["transition_counts"]),
            }
            for row in rows
        ],
    }
    (OUT_DIR / "chart_data.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    ensure_output_dir()
    sample_summary = load_sample_summary()
    manifest_records = load_manifest_records()
    audit_review_rows = load_audit_review_rows()
    rows = load_rows()
    plot_overall_agreement(rows)
    plot_bucket_accuracy(rows)
    plot_label_distribution(rows, sample_summary)
    plot_transition_breakdown(rows)
    pairwise_names, pairwise_matrix = plot_pairwise_agreement_heatmap(rows)
    gold_revision = build_revised_gold(manifest_records, audit_review_rows)
    pairwise_revised_names, pairwise_revised_matrix = plot_pairwise_agreement_heatmap_revised_gold(
        rows, gold_revision["revised_by_key"]
    )
    plot_gold_revision_process(audit_review_rows)
    plot_gold_revision_label_totals(gold_revision)
    plot_gold_revision_transition_heatmap(gold_revision)
    plot_label_distribution_revised_gold(rows, gold_revision)
    revised_scores = plot_accuracy_original_vs_revised_gold(rows, gold_revision["revised_by_key"])
    divergence_rows = build_divergence_rows(rows, sample_summary, gold_revision)
    plot_signed_divergence_dumbbells(divergence_rows)
    row_level_shift_rows = build_row_level_shift_rows(rows, gold_revision["revised_by_key"])
    plot_row_level_shift_heatmap(row_level_shift_rows)
    aggregate_delta_rows = build_aggregate_delta_rows(rows, sample_summary)
    plot_aggregate_score_shift_heatmap(aggregate_delta_rows)
    write_summary_json(
        rows,
        sample_summary,
        pairwise_names,
        pairwise_matrix,
        pairwise_revised_names,
        pairwise_revised_matrix,
        gold_revision,
        revised_scores,
        divergence_rows,
        row_level_shift_rows,
        aggregate_delta_rows,
    )
    print(f"Wrote charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
