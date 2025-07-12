#!/usr/bin/env python3
"""
Lexical Diversity Analyzer for LLM Compliance Analysis

Calculates lexical diversity scores for model responses using Root TTR metric.
Works with ComplianceAnalysis files from the judge script.
"""

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Import from your existing code structure
from compliance.data import JSONLHandler, ModelResponse, ComplianceAnalysis


def setup_nltk():
    """Download required NLTK data."""
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)


def extract_response_text(analysis: ComplianceAnalysis) -> str:
    """Extract the actual text content from a ComplianceAnalysis object."""
    if not analysis.response:
        return ""
    
    try:
        # Navigate the response structure to get the actual text
        content = analysis.response["choices"][0]["message"]["content"]
        return content if content else ""
    except (KeyError, IndexError, TypeError):
        # Handle malformed responses (ERROR cases, etc.)
        return ""


def get_lemmatized_words(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> List[str]:
    """Extract lemmatized words from text, removing stop words and punctuation."""
    if not text.strip():
        return []
    
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token) for token in tokens 
            if token.isalpha() and token not in stop_words]


def calculate_root_ttr(words: List[str]) -> float:
    """Calculate Root Type-Token Ratio (Root TTR) for lexical diversity."""
    if not words:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    return unique_words / math.sqrt(total_words)


def calculate_maas_index(words: List[str]) -> float:
    """Calculate Maas Index for lexical diversity."""
    if not words:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    if total_words <= 1 or unique_words <= 1:
        return 0.0
    
    return (math.log(total_words) - math.log(unique_words)) / (math.log(total_words) ** 2)


def get_cache_key(analysis: ComplianceAnalysis) -> str:
    """Generate a cache key for a compliance analysis based on content and timestamp."""
    text = extract_response_text(analysis)
    # Create hash from model, question_id, text content, and timestamp
    content = f"{analysis.model}:{analysis.question_id}:{text}:{analysis.timestamp}"
    return hashlib.md5(content.encode()).hexdigest()


def load_cache(cache_path: Path) -> Dict[str, Tuple[float, float, int, int]]:
    """Load cached lexical diversity results."""
    if not cache_path.exists():
        return {}
    
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load cache from {cache_path}: {e}")
        return {}


def save_cache(cache: Dict[str, Tuple[float, float, int, int]], cache_path: Path):
    """Save cached lexical diversity results."""
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)


def analyze_lexical_diversity(analyses: List[ComplianceAnalysis], 
                            cache_path: Path = None,
                            complete_only: bool = False) -> List[Tuple[str, str, float, float, int, int]]:
    """
    Analyze lexical diversity for all analyses with caching.
    
    Args:
        analyses: List of ComplianceAnalysis objects
        cache_path: Path to cache file
        complete_only: If True, only analyze responses judged as COMPLETE
    
    Returns:
        List of tuples: (model, question_id, root_ttr, maas_index, unique_words, total_words)
    """
    setup_nltk()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Filter analyses if requested
    if complete_only:
        original_count = len(analyses)
        analyses = [a for a in analyses if a.compliance == "COMPLETE"]
        filtered_count = len(analyses)
        print(f"Filtered to COMPLETE responses: {filtered_count}/{original_count} ({filtered_count/original_count*100:.1f}%)")
    
    # Load cache
    cache = {}
    if cache_path:
        cache = load_cache(cache_path)
        print(f"Loaded cache with {len(cache)} entries")
    
    results = []
    cache_hits = 0
    cache_misses = 0
    
    for analysis in analyses:
        cache_key = get_cache_key(analysis)
        
        if cache_key in cache:
            # Use cached result
            root_ttr, maas_index, unique_words, total_words = cache[cache_key]
            cache_hits += 1
        else:
            # Calculate new result
            text = extract_response_text(analysis)
            words = get_lemmatized_words(text, lemmatizer, stop_words)
            
            root_ttr = calculate_root_ttr(words)
            maas_index = calculate_maas_index(words)
            unique_words = len(set(words))
            total_words = len(words)
            
            # Cache the result
            cache[cache_key] = (root_ttr, maas_index, unique_words, total_words)
            cache_misses += 1
        
        results.append((
            analysis.model,
            analysis.question_id,
            root_ttr,
            maas_index,
            unique_words,
            total_words
        ))
    
    # Save updated cache
    if cache_path:
        save_cache(cache, cache_path)
        print(f"Cache stats: {cache_hits} hits, {cache_misses} misses. Cache now has {len(cache)} entries.")
    
    return results


def truncate_model_name(model_name: str, max_length: int = 45) -> str:
    """Truncate model name if too long, keeping the most important parts."""
    if len(model_name) <= max_length:
        return model_name
    
    # Try to keep the provider and main model name
    if '/' in model_name:
        provider, model = model_name.split('/', 1)
        # If even the model part is too long, truncate it
        available_space = max_length - len(provider) - 1  # -1 for the '/'
        if available_space > 10:  # Only truncate if we have reasonable space
            if len(model) > available_space:
                model = model[:available_space-3] + "..."
            return f"{provider}/{model}"
    
    # Fallback: simple truncation
    return model_name[:max_length-3] + "..."


def print_sorted_results(results: List[Tuple[str, str, float, float, int, int]], 
                        sort_by: str = "root_ttr",
                        top_n: int = None):
    """Print results sorted by the specified metric."""
    if not results:
        print("No results to display.")
        return
    
    # Sort by the specified metric (descending order for diversity scores)
    if sort_by == "root_ttr":
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        metric_name = "Root TTR"
        metric_idx = 2
    elif sort_by == "maas":
        sorted_results = sorted(results, key=lambda x: x[3], reverse=True)
        metric_name = "Maas Index"
        metric_idx = 3
    else:
        raise ValueError(f"Unknown sort metric: {sort_by}")
    
    print(f"\nLexical Diversity Results (sorted by {metric_name}):")
    print("=" * 115)
    print(f"{'Model':<48} {'Question ID':<15} {'Root TTR':<10} {'Maas Index':<12} {'Unique':<8} {'Total':<8}")
    print("-" * 115)
    
    display_results = sorted_results[:top_n] if top_n else sorted_results
    for model, qid, root_ttr, maas_index, unique, total in display_results:
        truncated_model = truncate_model_name(model)
        print(f"{truncated_model:<48} {qid:<15} {root_ttr:<10.4f} {maas_index:<12.4f} {unique:<8} {total:<8}")
    
    if top_n and len(sorted_results) > top_n:
        print(f"\n... showing top {top_n} of {len(sorted_results)} total results")
    
    # Print summary statistics
    print("\n" + "=" * 115)
    summary_title = f"SUMMARY STATISTICS" + (f" (TOP {top_n})" if top_n else "")
    print(summary_title)
    print("=" * 115)
    
    # Group by model
    model_stats = {}
    for model, _, root_ttr, maas_index, unique, total in results:
        if model not in model_stats:
            model_stats[model] = {"root_ttr": [], "maas": [], "unique": [], "total": []}
        model_stats[model]["root_ttr"].append(root_ttr)
        model_stats[model]["maas"].append(maas_index)
        model_stats[model]["unique"].append(unique)
        model_stats[model]["total"].append(total)
    
    print(f"{'Rank':<5} {'Model':<48} {'Root TTR':<9} {'Maas':<6} {'Unique':<7} {'Avg Len':<8} {'Count':<6}")
    print("-" * 115)
    
    # Sort models by Root TTR (descending - higher is better)
    sorted_models = sorted(model_stats.items(), 
                          key=lambda x: sum(x[1]["root_ttr"]) / len(x[1]["root_ttr"]), 
                          reverse=True)
    
    # Apply top_n filter to model summary too
    display_models = sorted_models[:top_n] if top_n else sorted_models
    
    for rank, (model, stats) in enumerate(display_models, 1):
        avg_root_ttr = sum(stats["root_ttr"]) / len(stats["root_ttr"])
        avg_maas = sum(stats["maas"]) / len(stats["maas"])
        avg_unique = sum(stats["unique"]) / len(stats["unique"])
        avg_total = sum(stats["total"]) / len(stats["total"])
        count = len(stats["root_ttr"])
        
        truncated_model = truncate_model_name(model)
        print(f"{rank:<5} {truncated_model:<48} {avg_root_ttr:<9.4f} {avg_maas:<6.4f} {avg_unique:<7.1f} {avg_total:<8.1f} {count:<6}")
    
    if top_n and len(sorted_models) > top_n:
        print(f"\n... showing top {top_n} of {len(sorted_models)} total models")


def main():
    parser = argparse.ArgumentParser(description="Analyze lexical diversity of LLM responses")
    parser.add_argument("analysis_files", nargs="+", type=Path, 
                       help="ComplianceAnalysis JSONL files to analyze")
    parser.add_argument("--sort-by", choices=["root_ttr", "maas"], default="root_ttr",
                       help="Metric to sort results by (default: root_ttr)")
    parser.add_argument("--min-words", type=int, default=10,
                       help="Minimum word count to include in analysis (default: 10)")
    parser.add_argument("--cache-file", type=Path, default=Path(".lexical_diversity_cache.pkl"),
                       help="Cache file for storing computed results (default: .lexical_diversity_cache.pkl)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching (slower but always fresh results)")
    parser.add_argument("--complete-only", action="store_true",
                       help="Only analyze responses judged as COMPLETE (recommended)")
    parser.add_argument("--top", type=int, metavar="N",
                       help="Show only top N models (default: show all)")
    
    args = parser.parse_args()
    
    # Load all analyses
    all_analyses = []
    for path in args.analysis_files:
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping.")
            continue
        
        analyses = JSONLHandler.load_jsonl(path, ComplianceAnalysis)
        print(f"Loaded {len(analyses)} analyses from {path}")
        all_analyses.extend(analyses)
    
    if not all_analyses:
        print("No analyses loaded. Exiting.")
        return
    
    # Show compliance breakdown
    compliance_counts = {}
    for analysis in all_analyses:
        compliance_counts[analysis.compliance] = compliance_counts.get(analysis.compliance, 0) + 1
    
    print(f"\nCompliance breakdown:")
    for compliance, count in sorted(compliance_counts.items()):
        pct = count / len(all_analyses) * 100
        print(f"  {compliance}: {count} ({pct:.1f}%)")
    
    # Analyze lexical diversity
    print(f"\nAnalyzing lexical diversity for {len(all_analyses)} analyses...")
    cache_path = None if args.no_cache else args.cache_file
    results = analyze_lexical_diversity(all_analyses, cache_path, args.complete_only)
    
    # Filter by minimum word count
    filtered_results = [(model, qid, root_ttr, maas, unique, total) 
                       for model, qid, root_ttr, maas, unique, total in results
                       if total >= args.min_words]
    
    if len(filtered_results) < len(results):
        print(f"Filtered out {len(results) - len(filtered_results)} responses with < {args.min_words} words")
    
    # Print results
    print_sorted_results(filtered_results, args.sort_by, args.top)


if __name__ == "__main__":
    main()
