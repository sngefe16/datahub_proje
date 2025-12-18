"""
Eighth View Model Package
Extends seventh view with calibration and advanced threshold optimization:
1. Calibration (Isotonic) - Ensemble output calibration
2. Percentile Threshold Sweep - %1 → %5 range, recall target ≥ 0.90
3. Cost-Based Threshold - FN/FP cost scenarios (10x, 20x, 50x)
Goal: Achieve Recall > 0.90 without changing the model.
"""

