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
import random
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


def load_cache(cache_path: Path) -> Dict[str, Tuple[float, float, int, int, List[str]]]:
    """Load cached lexical diversity results including word lists."""
    if not cache_path.exists():
        return {}
    
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load cache from {cache_path}: {e}")
        return {}


def save_cache(cache: Dict[str, Tuple[float, float, int, int, List[str]]], cache_path: Path):
    """Save cached lexical diversity results including word lists."""
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)


def preprocess_and_cache(all_analyses: List[ComplianceAnalysis], cache_path: Path = None):
    """
    Preprocess all analyses and build cache if needed.
    
    Args:
        all_analyses: All analyses to preprocess
        cache_path: Path to cache file
    """
    if not cache_path:
        return  # No caching requested
    
    # Load existing cache
    cache = load_cache(cache_path)
    print(f"Loaded cache with {len(cache)} entries")
    
    # Find analyses that need processing
    to_process = []
    for analysis in all_analyses:
        cache_key = get_cache_key(analysis)
        if cache_key not in cache:
            to_process.append(analysis)
    
    if not to_process:
        print("All analyses already cached")
        return
    
    print(f"Preprocessing {len(to_process)} uncached analyses...")
    
    # Process uncached analyses
    setup_nltk()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    for analysis in to_process:
        cache_key = get_cache_key(analysis)
        
        # Calculate result
        text = extract_response_text(analysis)
        words = get_lemmatized_words(text, lemmatizer, stop_words)
        
        root_ttr = calculate_root_ttr(words)
        maas_index = calculate_maas_index(words)
        unique_words = len(set(words))
        total_words = len(words)
        
        # Cache the result including word list
        cache[cache_key] = (root_ttr, maas_index, unique_words, total_words, words)
    
    # Save updated cache
    save_cache(cache, cache_path)
    print(f"Cache updated: now has {len(cache)} entries")


def get_cached_word_count(analysis: ComplianceAnalysis, cache: Dict) -> int:
    """
    Get word count for an analysis from cache.
    
    Args:
        analysis: Analysis to get word count for
        cache: Loaded cache dictionary
        
    Returns:
        Word count (0 if not in cache)
    """
    if not cache:
        return 0
        
    cache_key = get_cache_key(analysis)
    
    if cache_key in cache:
        _, _, _, total_words, _ = cache[cache_key]
        return total_words
    
    return 0


def find_balanced_sample(all_analyses: List[ComplianceAnalysis], 
                        sample_size: int, 
                        complete_only: bool = False,
                        min_words: int = 0,
                        cache: Dict = None) -> List[ComplianceAnalysis]:
    """
    Find a balanced sample of questions that all models answered.
    
    Args:
        all_analyses: All analyses
        sample_size: Number of questions to sample
        complete_only: If True, only consider COMPLETE responses
        min_words: Minimum word count filter (applied before sampling)
        cache: Loaded cache dictionary for fast word count lookup
        
    Returns:
        Filtered list of analyses containing only the balanced sample
        
    Raises:
        ValueError: If there aren't enough overlapping questions
    """
    print(f"Finding balanced sample of {sample_size} questions...")
    
    # Filter to complete responses if requested
    if complete_only:
        analyses = [a for a in all_analyses if a.compliance == "COMPLETE"]
    else:
        analyses = all_analyses
    
    # Apply min-words filter using cached values if specified
    if min_words > 0:
        print(f"Applying min-words filter using cached word counts...")
        filtered_analyses = []
        for analysis in analyses:
            word_count = get_cached_word_count(analysis, cache)
            if word_count >= min_words:
                filtered_analyses.append(analysis)
        
        print(f"Min-words filter: kept {len(filtered_analyses)}/{len(analyses)} responses with ≥{min_words} words")
        analyses = filtered_analyses
    
    # Group analyses by model and question_id
    model_questions = {}
    analysis_lookup = {}  # (model, question_id) -> analysis
    
    for analysis in analyses:
        model = analysis.model
        qid = analysis.question_id
        
        if model not in model_questions:
            model_questions[model] = set()
        
        model_questions[model].add(qid)
        analysis_lookup[(model, qid)] = analysis
    
    if not model_questions:
        raise ValueError("No analyses found for balanced sampling")
    
    # Find intersection of all models' question sets
    print(f"Found {len(model_questions)} models")
    all_question_sets = list(model_questions.values())
    common_questions = set.intersection(*all_question_sets)
    
    print(f"Common questions across all models: {len(common_questions)}")
    
    if len(common_questions) < sample_size:
        raise ValueError(
            f"Not enough common questions for balanced sampling. "
            f"Found {len(common_questions)} common questions, need {sample_size}. "
            f"Try a smaller --balanced-sample size."
        )
    
    # Randomly sample from common questions (deterministic order for reproducibility)
    common_questions_list = sorted(list(common_questions))  # Sort for consistent ordering
    sampled_questions = random.sample(common_questions_list, sample_size)
    print(f"Randomly sampled {len(sampled_questions)} questions for balanced comparison")
    
    # Collect analyses for the sampled questions
    balanced_analyses = []
    for model in model_questions.keys():
        for qid in sampled_questions:
            analysis = analysis_lookup.get((model, qid))
            if analysis:
                balanced_analyses.append(analysis)
    
    print(f"Balanced sample contains {len(balanced_analyses)} total analyses ({sample_size} questions × {len(model_questions)} models)")
    return balanced_analyses


def analyze_lexical_diversity(analyses: List[ComplianceAnalysis], 
                            cache_path: Path = None,
                            complete_only: bool = False) -> Tuple[List[Tuple[str, str, float, float, int, int]], Dict[str, List[str]]]:
    """
    Analyze lexical diversity for all analyses with caching.
    
    Args:
        analyses: List of ComplianceAnalysis objects
        cache_path: Path to cache file
        complete_only: If True, only analyze responses judged as COMPLETE
    
    Returns:
        Tuple of:
        - List of tuples: (model, question_id, root_ttr, maas_index, unique_words, total_words)
        - Dict mapping model names to all their words (for global calculations)
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
    model_all_words = {}  # Collect all words for global metrics
    cache_hits = 0
    cache_misses = 0
    
    for analysis in analyses:
        cache_key = get_cache_key(analysis)
        
        if cache_key in cache:
            # Use cached result (includes word list)
            root_ttr, maas_index, unique_words, total_words, words = cache[cache_key]
            cache_hits += 1
        else:
            # Calculate new result
            text = extract_response_text(analysis)
            words = get_lemmatized_words(text, lemmatizer, stop_words)
            
            root_ttr = calculate_root_ttr(words)
            maas_index = calculate_maas_index(words)
            unique_words = len(set(words))
            total_words = len(words)
            
            # Cache the result including word list
            cache[cache_key] = (root_ttr, maas_index, unique_words, total_words, words)
            cache_misses += 1
        
        # Add words to model's global collection
        if analysis.model not in model_all_words:
            model_all_words[analysis.model] = []
        model_all_words[analysis.model].extend(words)
        
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
    
    return results, model_all_words


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
                        model_all_words: Dict[str, List[str]],
                        sort_by: str = "root_ttr",
                        top_n: int = None):
    """Print results sorted by the specified metric."""
    if not results:
        print("No results to display.")
        return
    
    # Sort by the specified metric 
    if sort_by in ["root_ttr", "global_root_ttr"]:
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)  # Higher = better
        metric_name = "Root TTR"
        metric_idx = 2
    elif sort_by in ["maas", "global_maas"]:
        sorted_results = sorted(results, key=lambda x: x[3], reverse=False)  # Lower = better
        metric_name = "Maas Index"
        metric_idx = 3
    else:
        raise ValueError(f"Unknown sort metric: {sort_by}")
    
    print(f"\nLexical Diversity Results (sorted by {metric_name}):")
    print("=" * 140)
    print(f"{'Model':<48} {'Question ID':<15} {'Root TTR':<10} {'Maas Index':<12} {'Unique':<8} {'Total':<8}")
    print("-" * 140)
    
    display_results = sorted_results[:top_n] if top_n else sorted_results
    for model, qid, root_ttr, maas_index, unique, total in display_results:
        truncated_model = truncate_model_name(model)
        print(f"{truncated_model:<48} {qid:<15} {root_ttr:<10.4f} {maas_index:<12.4f} {unique:<8} {total:<8}")
    
    if top_n and len(sorted_results) > top_n:
        print(f"\n... showing top {top_n} of {len(sorted_results)} total results")
    
    # Print summary statistics
    print("\n" + "=" * 140)
    summary_title = f"SUMMARY STATISTICS (sorted by {sort_by})" + (f" (TOP {top_n})" if top_n else "")
    print(summary_title)
    print("=" * 140)
    
    # Group by model and calculate global metrics
    model_stats = {}
    
    for model, _, root_ttr, maas_index, unique, total in results:
        if model not in model_stats:
            model_stats[model] = {"root_ttr": [], "maas": [], "unique": [], "total": []}
        model_stats[model]["root_ttr"].append(root_ttr)
        model_stats[model]["maas"].append(maas_index)
        model_stats[model]["unique"].append(unique)
        model_stats[model]["total"].append(total)
    
    # Calculate global metrics for each model
    print("Calculating global diversity metrics...")
    global_metrics = {}
    for model, all_words in model_all_words.items():
        if all_words:
            global_root_ttr = calculate_root_ttr(all_words)
            global_maas = calculate_maas_index(all_words)
        else:
            global_root_ttr = 0.0
            global_maas = 0.0
        global_metrics[model] = (global_root_ttr, global_maas)
    
    print(f"{'Rank':<5} {'Model':<48} {'Root TTR':>9} {'Global RTT':>10} {'Maas':>6} {'Global Maas':>11} {'Unique':>7} {'Avg Len':>8} {'Global Uniq':>11} {'Global Len':>10} {'Count':>6}")
    print("-" * 140)
    
    # Sort models by the chosen metric
    if sort_by == "root_ttr":
        sorted_models = sorted(model_stats.items(), 
                              key=lambda x: sum(x[1]["root_ttr"]) / len(x[1]["root_ttr"]), 
                              reverse=True)  # Higher = better
    elif sort_by == "maas":
        sorted_models = sorted(model_stats.items(), 
                              key=lambda x: sum(x[1]["maas"]) / len(x[1]["maas"]), 
                              reverse=False)  # Lower = better
    elif sort_by == "global_root_ttr":
        sorted_models = sorted(model_stats.items(), 
                              key=lambda x: global_metrics[x[0]][0], 
                              reverse=True)  # Higher = better
    elif sort_by == "global_maas":
        sorted_models = sorted(model_stats.items(), 
                              key=lambda x: global_metrics[x[0]][1], 
                              reverse=False)  # Lower = better
    else:
        # Default fallback
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
        
        # Get global metrics
        global_root_ttr, global_maas = global_metrics[model]
        
        # Calculate global unique and total word counts
        global_words = model_all_words[model]
        global_unique_count = len(set(global_words))
        global_total_count = len(global_words)
        
        truncated_model = truncate_model_name(model)
        print(f"{rank:<5} {truncated_model:<48} {avg_root_ttr:>9.4f} {global_root_ttr:>10.4f} {avg_maas:>6.4f} {global_maas:>11.4f} {avg_unique:>7.1f} {avg_total:>8.1f} {global_unique_count:>11} {global_total_count:>10} {count:>6}")
    
    if top_n and len(sorted_models) > top_n:
        print(f"\n... showing top {top_n} of {len(sorted_models)} total models")


def main():
    parser = argparse.ArgumentParser(description="Analyze lexical diversity of LLM responses")
    parser.add_argument("analysis_files", nargs="+", type=Path, 
                       help="ComplianceAnalysis JSONL files to analyze")
    parser.add_argument("--sort-by", choices=["root_ttr", "maas", "global_root_ttr", "global_maas"], default="root_ttr",
                       help="Metric to sort results by (default: root_ttr)")
    parser.add_argument("--min-words", type=int, default=10,
                       help="Minimum word count to include in analysis (default: 10)")
    parser.add_argument("--cache-file", type=Path, default=Path(".lexical_diversity_cache.pkl"),
                       help="Cache file for storing computed results (default: .lexical_diversity_cache.pkl)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching (slower but always fresh results)")
    parser.add_argument("--complete-only", action="store_true",
                       help="Only analyze responses judged as COMPLETE (recommended)")
    parser.add_argument("--balanced-sample", type=int, metavar="N",
                       help="Use balanced sampling: only analyze N questions that ALL models answered (eliminates sampling bias)")
    parser.add_argument("--min-samples", type=int, default=0, metavar="N",
                       help="Exclude models with fewer than N responses before balanced sampling (default: 0)")
    parser.add_argument("--seed", type=int, metavar="N",
                       help="Random seed for reproducible balanced sampling (default: random)")
    parser.add_argument("--top", type=int, metavar="N",
                       help="Show only top N models (default: show all)")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Set random seed to {args.seed} for reproducible results")
    
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
    
    # Apply complete-only filter first (before min-samples)
    if args.complete_only:
        original_count = len(all_analyses)
        all_analyses = [a for a in all_analyses if a.compliance == "COMPLETE"]
        print(f"Complete-only filter: kept {len(all_analyses)}/{original_count} COMPLETE responses")
    
    # Apply minimum sample size filter (after complete-only filter)
    if args.min_samples > 0:
        # Count responses per model (already filtered by complete-only if requested)
        model_counts = {}
        for analysis in all_analyses:
            model = analysis.model
            model_counts[model] = model_counts.get(model, 0) + 1
        
        # Filter out models with too few samples
        qualifying_models = {model for model, count in model_counts.items() if count >= args.min_samples}
        original_count = len(all_analyses)
        all_analyses = [a for a in all_analyses if a.model in qualifying_models]
        
        print(f"Min-samples filter: kept {len(qualifying_models)}/{len(model_counts)} models with ≥{args.min_samples} responses")
        print(f"Filtered analyses: {len(all_analyses)}/{original_count} remaining")
    
    # Preprocess and cache all analyses first
    if not args.no_cache:
        preprocess_and_cache(all_analyses, args.cache_file)
    
    # Apply balanced sampling if requested (complete-only already applied above)
    if args.balanced_sample:
        # Load cache once for balanced sampling
        cache = load_cache(args.cache_file) if not args.no_cache else {}
        all_analyses = find_balanced_sample(all_analyses, args.balanced_sample, False, args.min_words, cache)  # complete_only=False since already filtered
        print(f"Using balanced sample: {len(all_analyses)} analyses")
        # Don't apply min_words filter again later since it's already done
        skip_min_words_filter = True
    else:
        skip_min_words_filter = False
    
    # Analyze lexical diversity (complete-only already applied, so pass False)
    print(f"\nAnalyzing lexical diversity for {len(all_analyses)} analyses...")
    cache_path = None if args.no_cache else args.cache_file
    results, model_all_words = analyze_lexical_diversity(all_analyses, cache_path, False)  # complete_only=False since already filtered
    
    # Filter by minimum word count (unless already done in balanced sampling)
    if not skip_min_words_filter:
        filtered_results = [(model, qid, root_ttr, maas, unique, total) 
                           for model, qid, root_ttr, maas, unique, total in results
                           if total >= args.min_words]
        
        if len(filtered_results) < len(results):
            print(f"Filtered out {len(results) - len(filtered_results)} responses with < {args.min_words} words")
    else:
        filtered_results = results
    
    # Print results
    print_sorted_results(filtered_results, model_all_words, args.sort_by, args.top)


if __name__ == "__main__":
    main()
