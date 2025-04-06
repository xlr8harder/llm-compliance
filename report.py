#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def main():
    parser = argparse.ArgumentParser(
        description='Grouped-by-model chart for government criticism analysis using Plotly'
    )
    parser.add_argument('jsonl_files', nargs='+', help='JSONL files to process')
    parser.add_argument('-o', '--output', default='report/government_criticism_analysis.png',
                        help='Output filename')
    parser.add_argument('--sort-by', default='name', choices=['name','compliance'],
                        help='Sort models by name or ascending compliance (least at top).')
    parser.add_argument('--highlight-models', default='',
                        help='Comma-separated list of models to highlight (blue + " - New!").')
    args = parser.parse_args()

    # Parse highlight models
    highlight_set = set()
    if args.highlight_models.strip():
        for m in args.highlight_models.split(','):
            highlight_set.add(m.strip())

    # (1) Read data
    rows = []
    for fpath in args.jsonl_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line.strip())
                rows.append({
                    'model': rec.get('model','unknown'),
                    'category': rec.get('category','uncategorized'),
                    'compliance': rec.get('compliance','UNKNOWN'),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data found. Exiting.")
        return

    # (2) Compute compliance by (model, category)
    valid_labels = ['COMPLETE','EVASIVE','DENIAL','ERROR']
    data_list = []
    for (model, category), grp in df.groupby(['model','category']):
        valid = grp[grp['compliance'].isin(valid_labels)]
        total_valid = len(valid)
        if total_valid > 0:
            pct_c = 100.0 * (valid['compliance']=='COMPLETE').sum() / total_valid
            pct_e = 100.0 * (valid['compliance']=='EVASIVE').sum()  / total_valid
            pct_d = 100.0 * (valid['compliance']=='DENIAL').sum()   / total_valid
            pct_r = 100.0 * (valid['compliance']=='ERROR').sum()    / total_valid
        else:
            pct_c = pct_e = pct_d = pct_r = 0.0
        data_list.append({
            'model':    model,
            'category': category,
            'complete': pct_c,
            'evasive':  pct_e,
            'denial':   pct_d,
            'error':    pct_r,
            'total_valid': total_valid
        })
    stats_df = pd.DataFrame(data_list)
    if stats_df.empty:
        print("No valid compliance data found.")
        return

    # (3) Sort models
    categories = sorted(stats_df['category'].unique())
    model_compliance = {}
    for model, mdf in stats_df.groupby('model'):
        w_sum = mdf['total_valid'].sum()
        if w_sum > 0:
            w_compl = (mdf['complete'] * mdf['total_valid']).sum() / w_sum
        else:
            w_compl = 0.0
        model_compliance[model] = w_compl

    all_models = stats_df['model'].unique()
    if args.sort_by == 'name':
        sorted_models = sorted(all_models)
        sort_label = "Name"
    else:
        # Ascending => least compliance at top
        sorted_models = sorted(all_models, key=lambda m: (model_compliance[m], m))
        sort_label = "Compliance (least at top)"

    # (4) Build row list with explicit model identification for each category row
    plot_data = []
    
    # First, explicitly identify which rows belong to which model and give them positions
    current_pos = 0
    for model_idx, model in enumerate(sorted_models):
        # Add model header row
        plot_data.append({
            'type': 'model_header',
            'model': model,
            'position': current_pos,
            'highlighted': model in highlight_set
        })
        current_pos += 1
        
        # Add category rows for this model
        model_stats = stats_df[stats_df['model'] == model]
        for cat in categories:
            cat_stats = model_stats[model_stats['category'] == cat]
            if not cat_stats.empty:
                row = cat_stats.iloc[0]
                plot_data.append({
                    'type': 'category',
                    'model': model,
                    'category': cat,
                    'position': current_pos,
                    'complete': row['complete'],
                    'evasive': row['evasive'],
                    'denial': row['denial'],
                    'error': row['error']
                })
            else:
                # Add empty row if category doesn't exist for this model
                plot_data.append({
                    'type': 'category',
                    'model': model,
                    'category': cat,
                    'position': current_pos,
                    'complete': 0,
                    'evasive': 0,
                    'denial': 0,
                    'error': 0
                })
            current_pos += 1
        
        # Add gap after each model except the last one
        if model_idx < len(sorted_models) - 1:
            current_pos += 0.5
    
    # Create the Plotly chart with the explicit model and category structure
    fig = create_improved_chart(plot_data, categories, sort_label)
    
    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as static image
    fig.write_image(args.output, scale=2)  # scale=2 for higher resolution
    print(f"Saved chart => {args.output}")

def create_improved_chart(plot_data, categories, sort_label):
    """Create a properly structured Plotly chart without excess space."""
    # Create a new figure
    fig = go.Figure()
    
    # Define colors
    colors = {
        'complete': '#2ecc71',  # Green
        'evasive': '#f1c40f',   # Yellow
        'denial': '#e74c3c',    # Red
        'error': '#9b59b6'      # Purple
    }
    
    # Process data rows first to set up traces
    y_ticks = []
    y_labels = []
    legend_added = set()
    
    # Add all data points for category rows
    for row in plot_data:
        position = row['position']
        y_ticks.append(position)
        
        if row['type'] == 'model_header':
            # Format model name with highlight if needed
            if row['highlighted']:
                y_labels.append(f"<b style='color: blue'>{row['model']} - New!</b>")
            else:
                y_labels.append(f"<b>{row['model']}</b>")
            
            # Add invisible bar to model header rows
            fig.add_trace(go.Bar(
                y=[position],
                x=[0.001],
                marker_color='rgba(0,0,0,0)',
                showlegend=False,
                hoverinfo='none',
                orientation='h'
            ))
        else:
            # Add indented category label
            y_labels.append(f"&nbsp;&nbsp;&nbsp;{row['category']}")
            
            # Add data for category rows
            if row['type'] == 'category':
                segments = [
                    ('Compliant', 'complete', 0),
                    ('Evasive', 'evasive', row['complete']),
                    ('Denial', 'denial', row['complete'] + row['evasive']),
                    ('Error', 'error', row['complete'] + row['evasive'] + row['denial'])
                ]
                
                for segment_name, value_key, base_val in segments:
                    show_legend = segment_name not in legend_added
                    if show_legend:
                        legend_added.add(segment_name)
                    
                    fig.add_trace(go.Bar(
                        y=[position],
                        x=[row[value_key]],
                        name=segment_name,
                        marker_color=colors[value_key],
                        orientation='h',
                        base=base_val,
                        showlegend=show_legend,
                        legendgroup=segment_name,
                        hovertemplate=f"{segment_name}: %{{x:.1f}}%<extra></extra>"
                    ))
    
    # Calculate the optimal figure height based on content
    row_count = len(y_ticks)
    
    # Adaptive spacing based on chart size
    # For smaller charts, we need more space proportionally for the legend and labels
    pixels_per_row = 22
    
    # Base height calculation
    base_height = row_count * pixels_per_row
    
    # For small charts, add extra padding to ensure legend has enough room
    min_height = 400  # Minimum chart height
    chart_height = max(min_height, base_height + 200)  # 200px for title, x-axis, and legend
    
    # Determine legend position adaptively
    # For short charts, place legend below the chart with fixed pixel positioning
    if row_count <= 15:  # For smaller charts
        legend_position = dict(
            orientation="h",
            xanchor="center",
            x=0.5,
            y=-0.25,          # Position legend below chart
            yanchor="top",    # Anchor to top of legend box
        )
        # Add more bottom margin for the legend
        margins = {'l': 200, 'r': 100, 't': 80, 'b': 150}
    else:  # For larger charts
        legend_position = dict(
            orientation="h",
            xanchor="center",
            x=0.5,
            y=-0.12,          # Closer to the chart for larger charts
            yanchor="top",    # Anchor to top of legend box
        )
        margins = {'l': 200, 'r': 100, 't': 80, 'b': 120}
    
    # Let Plotly determine the appropriate layout based on content
    fig.update_layout(
        title={
            'text': f"Will models comply with requests to criticize various governments?",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18}
        },
        xaxis_title={
            'text': "Percentage of valid responses (%)",
            'font': {'size': 14},
            'standoff': 20
        },
        xaxis={
            'range': [0, 100],
            'tickfont': {'size': 12},
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'showgrid': True,
        },
        yaxis={
            'tickvals': y_ticks,
            'ticktext': y_labels,
            'autorange': "reversed",
            'tickfont': {'size': 12}
        },
        barmode='stack',
        bargap=0.2,
        bargroupgap=0.1,
        legend=legend_position,
        margin=margins,
        plot_bgcolor='white',
        font={'family': "Arial"},
        height=chart_height,
        width=1000
    )
    
    return fig

if __name__ == "__main__":
    main()

