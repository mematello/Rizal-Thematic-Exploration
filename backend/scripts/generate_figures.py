import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Ensure output directory exists
out_dir = "plots"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def plot_figure_1():
    with open('figure1_score_matrix.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matrix = np.array(data['score_matrix'])
    dp_path = [(p['buod_index'], p['full_text_window_start']) for p in data['dp_path']]
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(matrix, cmap="YlGnBu", cbar_kws={'label': 'Alignment Score'})
    
    # Plot DP path
    x_coords = [p[1] + 0.5 for p in dp_path]
    y_coords = [p[0] + 0.5 for p in dp_path]
    plt.plot(x_coords, y_coords, color='red', linewidth=2, label="DP Global Path", marker='o', markersize=4)
    
    plt.title(f"Figure 1: Global Alignment Score Matrix (Ch. {data['chapter']})")
    plt.xlabel("Full Text Window Start Index")
    plt.ylabel("Buod Sentence Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure1_score_matrix.png'), dpi=300)
    plt.close()
    print("Generated Figure 1")

def plot_figure_2():
    with open('figure2_window_scale_scores.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plt.figure(figsize=(10, 6))
    
    for case in data:
        sizes = list(case['scores_by_window_size'].keys())
        scores = list(case['scores_by_window_size'].values())
        
        # Convert sizes to int for plotting
        sizes_int = [int(s) for s in sizes]
        
        plt.plot(sizes_int, scores, marker='o', linewidth=2, 
                 label=f"ID {case['sentence_id']} ({case['novel']} Ch {case['chapter']})")
        
        # Mark winning size
        winner = case['winning_window_size']
        if str(winner) in sizes:
            idx = sizes.index(str(winner))
            plt.scatter([winner], [scores[idx]], color='red', s=100, zorder=5)

    plt.title("Figure 2: Alignment Score Variability Across Window Sizes")
    plt.xlabel("Window Size (Number of Sentences)")
    plt.ylabel("Maximum Consolidated Score")
    plt.xticks([3, 4, 5, 6])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure2_window_scale.png'), dpi=300)
    plt.close()
    print("Generated Figure 2")

def plot_figure_3():
    with open('figure3_tauhan_gate_scatter.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plt.figure(figsize=(10, 8))
    
    for case in data:
        passed_x, passed_y = [], []
        failed_x, failed_y = [], []
        winner_x, winner_y = [], []
        
        for cand in case['candidates']:
            if cand['is_winner']:
                winner_x.append(cand['lexical_score'])
                winner_y.append(cand['semantic_score'])
            elif cand['tauhan_passed']:
                passed_x.append(cand['lexical_score'])
                passed_y.append(cand['semantic_score'])
            else:
                failed_x.append(cand['lexical_score'])
                failed_y.append(cand['semantic_score'])
                
        plt.scatter(passed_x, passed_y, alpha=0.5, color='blue', marker='o')
        plt.scatter(failed_x, failed_y, alpha=0.5, color='red', marker='x')
        plt.scatter(winner_x, winner_y, s=150, color='gold', edgecolor='black', marker='*', zorder=10)

    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Passed Tauhan Gate'),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='red', markersize=10, label='Failed Tauhan Gate'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=15, label='Winning Selection')
    ]
    
    plt.legend(handles=legend_elements)
    plt.title("Figure 3: Semantic vs Lexical Scores Filtered by Tauhan Gate")
    plt.xlabel("Lexical Score")
    plt.ylabel("Semantic Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure3_tauhan_scatter.png'), dpi=300)
    plt.close()
    print("Generated Figure 3")


def plot_figure_4():
    with open('figure4_greedy_vs_dp.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    greedy = data['greedy_path']
    dp = data['dp_path']
    
    buod_indices = [x['buod_index'] for x in greedy]
    greedy_matches = [x['matched_full_text_index'] for x in greedy]
    dp_matches = [x['matched_full_text_index'] for x in dp]
    
    plt.figure(figsize=(12, 6))
    plt.plot(buod_indices, greedy_matches, 'x--', color='red', label='Greedy (Argmax) Path', alpha=0.7)
    plt.plot(buod_indices, dp_matches, 'o-', color='blue', label='DP (Global) Path', linewidth=2)
    
    plt.title(f"Figure 4: Greedy vs Dynamic Programming Path Sequences (Ch. {data['chapter']})")
    plt.xlabel("Buod Sentence Index (Sequential Progression)")
    plt.ylabel("Matched Full-Text Window Start Index")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure4_greedy_dp.png'), dpi=300)
    plt.close()
    print("Generated Figure 4")

def plot_figure_5():
    with open('figure5_audit_40_sentences.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # Categorize by alignment_status
    order = ['precise', 'high', 'medium', 'rejected']
    colors = {'precise': 'green', 'high': 'blue', 'medium': 'orange', 'rejected': 'red'}
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='alignment_status', order=order, palette=colors)
    plt.title("Figure 5: Alignment Status Distribution (42-Sentence Audit)")
    plt.xlabel("Confidence Category")
    plt.ylabel("Number of Sentences")
    
    # Add count labels on top
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.ylim(0, max(df['alignment_status'].value_counts()) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure5_audit_dist.png'), dpi=300)
    plt.close()
    print("Generated Figure 5")

def plot_figure_6():
    with open('figure6_positional_drift.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 8))
    
    # Ideal line
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.5, label='Ideal 1:1 Mapping')
    
    sns.scatterplot(data=df, x='estimated_position_normalized', y='actual_position_normalized', 
                    hue='alignment_status', hue_order=['precise', 'high', 'medium', 'rejected'],
                    palette={'precise': 'green', 'high': 'blue', 'medium': 'orange', 'rejected': 'red'},
                    s=100, alpha=0.8)
    
    plt.title("Figure 6: Estimated vs. Actual Positional Drift")
    plt.xlabel("Estimated Normalized Position (Buod)")
    plt.ylabel("Actual Normalized Position (Passage Match)")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(title='Confidence Category')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure6_drift.png'), dpi=300)
    plt.close()
    print("Generated Figure 6")

if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    print("All figures successfully exported to the 'plots' folder!")
