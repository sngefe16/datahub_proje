"""
PowerPoint sunumu için Last Model Pipeline Diagram oluşturucu
Matplotlib ile flowchart oluşturur ve PNG olarak kaydeder
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

def create_pipeline_diagram(output_path='last_model_pipeline_diagram.png', dpi=300):
    """
    Last Model pipeline'ının görsel diagramını oluşturur.
    
    Parameters
    ----------
    output_path : str
        Çıktı dosyası yolu
    dpi : int
        Çözünürlük (PowerPoint için 300 önerilir)
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 25)
    ax.axis('off')
    
    # Renkler
    colors = {
        'data': '#E3F2FD',      # Açık mavi - Veri işleme
        'preprocess': '#BBDEFB', # Mavi - Ön işleme
        'feature': '#90CAF9',    # Orta mavi - Feature engineering
        'model': '#64B5F6',      # Koyu mavi - Model
        'eval': '#42A5F5',       # Daha koyu mavi - Değerlendirme
        'calibration': '#1E88E5', # Çok koyu mavi - Kalibrasyon
        'threshold': '#0D47A1',  # En koyu mavi - Threshold
        'output': '#1565C0'      # Çıktı
    }
    
    # Box style
    box_style = dict(boxstyle="round,pad=0.5", edgecolor='black', linewidth=1.5)
    
    # Adım 1: Data Loading
    y_pos = 24
    box1 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                         facecolor=colors['data'], alpha=0.8)
    ax.add_patch(box1)
    ax.text(5, y_pos-0.4, '[1/11] Data Loading\nTransaction + Identity Data', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Ok
    arrow1 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow1)
    
    # Adım 2: Preprocessing
    y_pos -= 1.5
    box2 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                         facecolor=colors['preprocess'], alpha=0.8)
    ax.add_patch(box2)
    ax.text(5, y_pos-0.4, '[2/11] Preprocessing\nType Conversion, Missing Values', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow2 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow2)
    
    # Adım 3: Feature Engineering
    y_pos -= 1.5
    box3 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                         facecolor=colors['feature'], alpha=0.8)
    ax.add_patch(box3)
    ax.text(5, y_pos-0.4, '[3/11] Feature Engineering\nUID, Rolling Windows, Statistical Features', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow3 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow3)
    
    # Adım 4: Feature Preparation
    y_pos -= 1.5
    box4 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                         facecolor=colors['preprocess'], alpha=0.8)
    ax.add_patch(box4)
    ax.text(5, y_pos-0.4, '[4/11] Feature Preparation\nTrain/Val Split (80/20, Stratified)', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow4 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow4)
    
    # Adım 5: SMOTE (Optional)
    y_pos -= 1.5
    box5 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                         facecolor=colors['preprocess'], alpha=0.6)
    ax.add_patch(box5)
    ax.text(5, y_pos-0.4, '[5/11] SMOTE (Optional)\nClass Imbalance Handling', 
            ha='center', va='center', fontsize=11, weight='bold', style='italic')
    
    arrow5 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow5)
    
    # Adım 6: Feature Selection
    y_pos -= 1.5
    box6 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                         facecolor=colors['feature'], alpha=0.8)
    ax.add_patch(box6)
    ax.text(5, y_pos-0.4, '[6/11] Feature Selection\nVariance, Correlation, Importance', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow6 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow6)
    
    # Adım 7: Model Training
    y_pos -= 1.5
    box7 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                         facecolor=colors['model'], alpha=0.8)
    ax.add_patch(box7)
    ax.text(5, y_pos-0.4, '[7/11] Model Training\nEnsemble (LightGBM + XGBoost + CatBoost)\n+ Optuna Hyperparameter Tuning', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow7 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow7)
    
    # Adım 8: Cross-Validation
    y_pos -= 1.5
    box8 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                         facecolor=colors['eval'], alpha=0.6)
    ax.add_patch(box8)
    ax.text(5, y_pos-0.4, '[8/11] Cross-Validation (Optional)\n3-Fold CV for Ensemble Weights', 
            ha='center', va='center', fontsize=11, weight='bold', style='italic')
    
    arrow8 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow8)
    
    # Adım 9: Model Evaluation
    y_pos -= 1.5
    box9 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                         facecolor=colors['eval'], alpha=0.8)
    ax.add_patch(box9)
    ax.text(5, y_pos-0.4, '[9/11] Model Evaluation\nAUC-ROC, Precision, Recall, F1', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow9 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow9)
    
    # Adım 10: Calibration
    y_pos -= 1.5
    box10 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                          facecolor=colors['calibration'], alpha=0.8)
    ax.add_patch(box10)
    ax.text(5, y_pos-0.4, '[10/13] Probability Calibration\nIsotonic Regression (Ensemble Output)', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow10 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                             arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow10)
    
    # Adım 10.5: Threshold Optimization (Multi-objective)
    y_pos -= 1.5
    box10_5 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                            facecolor=colors['threshold'], alpha=0.8)
    ax.add_patch(box10_5)
    ax.text(5, y_pos-0.4, '[10.5/13] Threshold Optimization\nMulti-Objective (AUC ≥0.94, Recall ≥0.90, Precision ≥0.40)', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow10_5 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                               arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow10_5)
    
    # Adım 10.6: Percentile Threshold Sweep
    y_pos -= 1.5
    box10_6 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                            facecolor=colors['threshold'], alpha=0.7)
    ax.add_patch(box10_6)
    ax.text(5, y_pos-0.4, '[10.6/13] Percentile Threshold Sweep\n0.5% → 5% (Recall Target ≥0.90)', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow10_6 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                               arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow10_6)
    
    # Adım 10.7: Cost-Based Threshold
    y_pos -= 1.5
    box10_7 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                            facecolor=colors['threshold'], alpha=0.7)
    ax.add_patch(box10_7)
    ax.text(5, y_pos-0.4, '[10.7/13] Cost-Based Threshold\nFN/FP Ratios: 10x, 20x, 50x', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow10_7 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                               arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow10_7)
    
    # Final Threshold Selection
    y_pos -= 1.5
    box_final = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                               facecolor=colors['threshold'], alpha=0.9)
    ax.add_patch(box_final)
    ax.text(5, y_pos-0.4, 'Final Threshold Selection\nPriority: Percentile → Cost-Based → Multi-Objective', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    arrow_final = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.2), 
                                 arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow_final)
    
    # Adım 11: Test Predictions
    y_pos -= 1.5
    box11 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, **box_style, 
                          facecolor=colors['output'], alpha=0.8)
    ax.add_patch(box11)
    ax.text(5, y_pos-0.4, '[11/11] Test Set Predictions\nSubmission File Generation', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Başlık
    ax.text(5, 24.5, 'Last Model - Fraud Detection Pipeline', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Alt başlık
    ax.text(5, 24.1, 'Ensemble Model with Calibration & Threshold Optimization', 
            ha='center', va='center', fontsize=12, style='italic')
    
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
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
             ncol=4, fontsize=9, frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ Diagram kaydedildi: {output_path}")
    return fig

if __name__ == "__main__":
    create_pipeline_diagram('last_model_pipeline_diagram.png', dpi=300)
    print("✅ Pipeline diagram oluşturuldu!")

