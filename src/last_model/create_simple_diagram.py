"""
Basit pipeline diagram oluşturucu - PowerPoint için
Matplotlib backend sorunlarını önlemek için basit yaklaşım
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_simple_diagram(output_path='last_model_pipeline_diagram.png', dpi=300):
    """
    Basit pipeline diagram oluşturur.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 22))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 26)
    ax.axis('off')
    
    # Renkler
    colors = {
        'data': '#E3F2FD',
        'preprocess': '#BBDEFB',
        'feature': '#90CAF9',
        'model': '#64B5F6',
        'eval': '#42A5F5',
        'calibration': '#1E88E5',
        'threshold': '#0D47A1',
        'output': '#1565C0'
    }
    
    # Box style
    box_style = dict(boxstyle="round,pad=0.6", edgecolor='#333', linewidth=2)
    
    steps = [
        (24.5, '[1/11] Data Loading', 'Transaction + Identity Data', colors['data']),
        (23, '[2/11] Preprocessing', 'Type Conversion, Missing Values', colors['preprocess']),
        (21.5, '[3/11] Feature Engineering', 'UID, Rolling Windows, Statistical', colors['feature']),
        (20, '[4/11] Feature Preparation', 'Train/Val Split (80/20, Stratified)', colors['preprocess']),
        (18.5, '[5/11] SMOTE (Optional)', 'Class Imbalance Handling', colors['preprocess'], True),
        (17, '[6/11] Feature Selection', 'Variance, Correlation, Importance', colors['feature']),
        (15.5, '[7/11] Model Training', 'Ensemble: LGB + XGB + CatBoost\n+ Optuna Tuning', colors['model']),
        (14, '[8/11] Cross-Validation (Optional)', '3-Fold CV for Ensemble Weights', colors['eval'], True),
        (12.5, '[9/11] Model Evaluation', 'AUC-ROC, Precision, Recall, F1', colors['eval']),
        (11, '[10/13] Probability Calibration', 'Isotonic Regression', colors['calibration']),
        (9.5, '[10.5/13] Threshold Optimization', 'Multi-Objective\nAUC≥0.94, Recall≥0.90, Precision≥0.40', colors['threshold']),
        (8, '[10.6/13] Percentile Threshold Sweep', '0.5% → 5% (Recall Target ≥0.90)', colors['threshold']),
        (6.5, '[10.7/13] Cost-Based Threshold', 'FN/FP Ratios: 10x, 20x, 50x', colors['threshold']),
        (5, 'Final Threshold Selection', 'Priority: Percentile → Cost-Based → Multi-Objective', colors['threshold']),
        (3.5, '[11/11] Test Set Predictions', 'Submission File Generation', colors['output']),
    ]
    
    # Başlık
    ax.text(5, 25.5, 'Last Model - Fraud Detection Pipeline', 
            ha='center', va='center', fontsize=18, weight='bold', color='#0D47A1')
    ax.text(5, 25, 'Ensemble Model with Calibration & Threshold Optimization', 
            ha='center', va='center', fontsize=12, style='italic', color='#666')
    
    # Adımları çiz
    for i, step in enumerate(steps):
        y_pos = step[0]
        title = step[1]
        desc = step[2]
        color = step[3]
        is_optional = len(step) > 4 and step[4]
        
        # Box
        alpha = 0.6 if is_optional else 0.8
        box = FancyBboxPatch((1, y_pos-0.9), 8, 0.9, **box_style, 
                            facecolor=color, alpha=alpha)
        ax.add_patch(box)
        
        # Text
        ax.text(5, y_pos-0.45, f'{title}\n{desc}', 
                ha='center', va='center', fontsize=10, weight='bold')
        
        # Ok (son adım hariç)
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((5, y_pos-0.9), (5, y_pos-1.4), 
                                   arrowstyle='->', lw=2.5, color='#333')
            ax.add_patch(arrow)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], alpha=0.8, label='Data Processing'),
        mpatches.Patch(facecolor=colors['preprocess'], alpha=0.8, label='Preprocessing'),
        mpatches.Patch(facecolor=colors['feature'], alpha=0.8, label='Feature Engineering'),
        mpatches.Patch(facecolor=colors['model'], alpha=0.8, label='Model Training'),
        mpatches.Patch(facecolor=colors['eval'], alpha=0.8, label='Evaluation'),
        mpatches.Patch(facecolor=colors['calibration'], alpha=0.8, label='Calibration'),
        mpatches.Patch(facecolor=colors['threshold'], alpha=0.8, label='Threshold Optimization'),
        mpatches.Patch(facecolor=colors['output'], alpha=0.8, label='Output')
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
             bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=9, 
             frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.2)
    print(f"✅ Diagram kaydedildi: {output_path}")
    plt.close()
    return output_path

if __name__ == "__main__":
    try:
        output = create_simple_diagram('last_model_pipeline_diagram.png', dpi=300)
        print(f"✅ Pipeline diagram başarıyla oluşturuldu: {output}")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()

