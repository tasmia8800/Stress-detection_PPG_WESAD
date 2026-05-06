import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


SUMMARY_FILE = 'result/baseline_results.csv'
OUTPUT_PLOT = 'result/baseline_plots.png'

def plot_results():
    if not os.path.exists(SUMMARY_FILE):
        print(f"Results file {SUMMARY_FILE} not found. Run the pipeline first.")
        return

    df = pd.read_csv(SUMMARY_FILE)
    
    print("Plotting classification performance...")
    df = df.drop_duplicates(subset=['Task', 'Noise', 'Classifier'], keep='last')

    tasks = df['Task'].unique()
    metrics = ['Accuracy', 'F1_Score']
    
    fig, axes = plt.subplots(nrows=len(tasks), ncols=1, figsize=(10, 5 * len(tasks)))
    if len(tasks) == 1:
        axes = [axes]
    
    colors = ['#4c72b0', '#55a868'] 
    bar_width = 0.35

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_df = df[df['Task'] == task]
        classifiers = sorted(task_df['Classifier'].unique())
        x = np.arange(len(classifiers))
        
        for j, metric in enumerate(metrics):
            scores = []
            for clf in classifiers:
                val = task_df[task_df['Classifier'] == clf][metric].values
                scores.append(val[0] if len(val) > 0 else 0)
            
            rects = ax.bar(x + j*bar_width, scores, bar_width, label=metric, color=colors[j], alpha=0.8)
            
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

        ax.set_title(f'{task} Performance', fontsize=14)
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(classifiers)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score (0-1)')
        if i == 0:
            ax.legend()
            
    fig.suptitle('Classification Benchmarks (Accuracy vs F1)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    print(f"Saving plot to {OUTPUT_PLOT}...")
    plt.savefig(OUTPUT_PLOT, bbox_inches='tight', dpi=300)
    print("Done.")

if __name__ == "__main__":
    plot_results()
