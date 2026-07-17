#!/usr/bin/env python3
"""Inspect the saved XGBoost model features."""
import pickle
import os

model_path = "models/xgboost_model.pkl"

if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    exit(1)

try:
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    features = model_data.get("feature_names", [])
    print(f"=" * 80)
    print(f"MODEL FEATURE ANALYSIS: {model_path}")
    print(f"=" * 80)
    print(f"Total features: {len(features)}\n")
    
    # Categorize features
    mtf_features = [f for f in features if f.startswith("M5_") or f.startswith("M1_") or f.startswith("M15_")]
    standard_features = [f for f in features if not any(f.startswith(p) for p in ["M5_", "M1_", "M15_"])]
    
    print(f"Feature Breakdown:")
    print(f"  - Standard features: {len(standard_features)}")
    print(f"  - Multi-TF (M5/M1/M15) features: {len(mtf_features)}\n")
    
    if len(features) <= 150:
        print("All features:")
        for i, f in enumerate(features, 1):
            print(f"{i:3d}. {f}")
    else:
        print("First 20 features:")
        for i, f in enumerate(features[:20], 1):
            print(f"{i:3d}. {f}")
        print(f"\n... ({len(features) - 20} more features)\n")
        print("Last 10 features:")
        for i, f in enumerate(features[-10:], len(features) - 9):
            print(f"{i:3d}. {f}")
    
    print(f"\n" + "=" * 80)
    print(f"Model metadata:")
    print(f"  - Train AUC: {model_data.get('train_auc', 'N/A')}")
    print(f"  - Test AUC: {model_data.get('test_auc', 'N/A')}")
    print(f"  - Confidence threshold: {model_data.get('confidence_threshold', 'N/A')}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
