#!/usr/bin/env python3
import argparse
import sys
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safe_filename(name):
    """Make a filename-friendly version of a string."""
    return re.sub(r'[^a-zA-Z0-9_\-]+', '_', name)

def main():
    parser = argparse.ArgumentParser(description='Generate a report from JSONL files')
    parser.add_argument('jsonl_files', nargs='+', help='JSONL files to process')
    parser.add_argument('-o', '--output', default='report/government_criticism_analysis.png', help='Output filename')

    args = parser.parse_args()

    jsonl_files = args.jsonl_files
    output_file = args.output

    rows = []
    
    # Read each JSONL
    for fpath in jsonl_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                
                rows.append({
                    'model': record.get('model', 'unknown'),
                    'category': record.get('category', 'uncategorized'),
                    'compliance': record.get('compliance', 'UNKNOWN'),
                    'question_id': record.get('question_id', '')
                })
    
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data found. Exiting.")
        return
    
    # Create reports directory
    out_dir = "report"
    os.makedirs(out_dir, exist_ok=True)
    
    # Get unique models and categories
    models = sorted(df['model'].unique())
    categories = sorted(df['category'].unique())
    
    # Calculate percentages for each model-category combination
    stats = []
    for model in models:
        for category in categories:
            model_cat_df = df[(df['model'] == model) & (df['category'] == category)]
            
            # Only count valid responses (COMPLETE, EVASIVE, DENIAL, or ERROR)
            valid_responses = model_cat_df[model_cat_df['compliance'].isin(['COMPLETE', 'EVASIVE', 'DENIAL', 'ERROR'])]
            total_valid = len(valid_responses)
            
            if total_valid > 0:
                complete_pct = (sum(valid_responses['compliance'] == 'COMPLETE') / total_valid) * 100
                evasive_pct = (sum(valid_responses['compliance'] == 'EVASIVE') / total_valid) * 100
                denial_pct = (sum(valid_responses['compliance'] == 'DENIAL') / total_valid) * 100
                error_pct = (sum(valid_responses['compliance'] == 'ERROR') / total_valid) * 100
            else:
                complete_pct = evasive_pct = denial_pct = error_pct = 0
                
            stats.append({
                'model': model,
                'category': category,
                'complete': complete_pct,
                'evasive': evasive_pct,
                'denial': denial_pct,
                'error': error_pct,
                'total_valid': total_valid
            })
    
    # Create visualization
    fig_width = 15
    fig_height = 8  # Increased height to accommodate labels
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Setup the plot
    model_positions = np.arange(len(models))
    bar_width = 0.12  # Reduced width to create spacing between bars
    category_offsets = np.linspace(-(len(categories)-1)*(bar_width*1.5)/2, 
                                 (len(categories)-1)*(bar_width*1.5)/2, 
                                 len(categories))  # Increased spacing between category groups)
    
    # Plot bars for each category and compliance type
    for idx, category in enumerate(categories):
        category_stats = [s for s in stats if s['category'] == category]
        
        # Get data for this category
        complete_data = [s['complete'] for s in category_stats]
        evasive_data = [s['evasive'] for s in category_stats]
        denial_data = [s['denial'] for s in category_stats]
        error_data = [s['error'] for s in category_stats]
        
        # Calculate x positions for this category's bars
        x_positions = model_positions + category_offsets[idx]
        
        # Plot stacked bars
        ax.bar(x_positions, complete_data, bar_width, 
               label=f'{category} - Compliant' if idx == 0 else "", 
               color='#2ecc71', alpha=0.7)
        
        # Add category labels above bars - now completely vertical (90 degrees)
        for i, x_pos in enumerate(x_positions):
            stat = category_stats[i]
            if stat['total_valid'] > 0:  # Only add label if there are valid responses
                ax.text(x_pos, -2, category, 
                       ha='center', va='top', 
                       rotation=90, fontsize=8)  # Changed rotation to 90 and ha to 'center'
                
        # Plot evasive
        ax.bar(x_positions, evasive_data, bar_width, 
               bottom=complete_data,
               label=f'{category} - Evasive' if idx == 0 else "", 
               color='#f1c40f', alpha=0.7)
        
        # Plot denial
        ax.bar(x_positions, denial_data, bar_width,
               bottom=[sum(x) for x in zip(complete_data, evasive_data)],
               label=f'{category} - Denial' if idx == 0 else "", 
               color='#e74c3c', alpha=0.7)
        
        # Plot error (purple, at the top)
        ax.bar(x_positions, error_data, bar_width,
               bottom=[sum(x) for x in zip(complete_data, evasive_data, denial_data)],
               label=f'{category} - Error' if idx == 0 else "", 
               color='#9b59b6', alpha=0.7)
    
    # Customize the plot
    # Add extra space at bottom for labels
    ax.set_ylim(0, 100)
    ax.margins(x=0.02, y=0)  # Add small horizontal margin
    
    # Position model names below category labels
    ax.set_xticks(model_positions)
    ax.set_xticklabels([model.split('/')[-1] for model in models], 
                       rotation=0, ha='center', fontsize=7)
    ax.tick_params(axis='x', pad=80)  # Increased padding between axis and labels
    
    title = "AI Model Responses: Government Criticism Analysis"
    plt.suptitle(title, fontsize=12, y=1.02)
    
    # Add legend with cleaner labels
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#2ecc71', alpha=0.7, label='Compliant'),
        plt.Rectangle((0,0),1,1, facecolor='#f1c40f', alpha=0.7, label='Evasive'),
        plt.Rectangle((0,0),1,1, facecolor='#e74c3c', alpha=0.7, label='Denial'),
        plt.Rectangle((0,0),1,1, facecolor='#9b59b6', alpha=0.7, label='Error')
    ]
    ax.legend(handles=legend_elements, 
             bbox_to_anchor=(1.05, 1), loc='upper left', 
             borderaxespad=0., fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    out_path = output_file
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved chart => {out_path}")
    
    # Print summary statistics
    print("\nSummary Statistics (percentages of valid responses only):")
    print("-" * 50)
    for model in models:
        print(f"\nModel: {model}")
        for category in categories:
            model_cat_stats = next(s for s in stats 
                                 if s['model'] == model and s['category'] == category)
            print(f"\n  Category: {category}")
            print(f"  Valid responses: {model_cat_stats['total_valid']}")
            print(f"  Complete:  {model_cat_stats['complete']:.1f}%")
            print(f"  Evasive:   {model_cat_stats['evasive']:.1f}%")
            print(f"  Denial:    {model_cat_stats['denial']:.1f}%")
            print(f"  Error:     {model_cat_stats['error']:.1f}%")

if __name__ == "__main__":
    main()
