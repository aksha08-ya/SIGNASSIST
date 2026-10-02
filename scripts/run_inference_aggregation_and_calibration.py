"""Phase 7 & 8: Multi-Window Inference Aggregation & Temperature Calibration Study on Validation Set."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.include_dataset import INCLUDEFeatureDataset
from src.training.train_controlled_stgcn import ControlledSTGCNClassifier, unpack_batch

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)

def main():
    print("=== PHASE 7 & 8: INFERENCE AGGREGATION & CALIBRATION STUDY (VALIDATION) ===")
    
    controlled_dir = Path("data/processed/include/controlled")
    val_dataset = INCLUDEFeatureDataset(split_csv=controlled_dir / "validation.csv")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    ckpt_path = Path("models/checkpoints/experimental/controlled_stgcn_best.pt")
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
    
    device = torch.device("cpu")
    model = ControlledSTGCNClassifier(num_classes=42).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    
    raw_logits_list = []
    labels_list = []
    
    print(f"Evaluating {len(val_dataset)} validation samples through Controlled ST-GCN model...")
    
    with torch.no_grad():
        for batch in val_loader:
            x, y = unpack_batch(batch)
            x = x.to(device)
            logits = model(x).detach().cpu().numpy()[0]
            
            raw_logits_list.append(logits)
            labels_list.append(int(y.item()))
            
    raw_logits = np.array(raw_logits_list)
    labels = np.array(labels_list)
    
    # -----------------------------------------------------------------
    # PHASE 8: TEMPERATURE GRID SEARCH ON VALIDATION LOGITS ONLY
    # -----------------------------------------------------------------
    temps = np.arange(0.40, 1.52, 0.02)
    grid_results = []
    
    best_nll = float("inf")
    best_temp = 1.0
    best_ece = float("inf")
    
    for T in temps:
        scaled_logits = raw_logits / T
        shift_logits = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(shift_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        preds = np.argmax(probs, axis=1)
        top1_acc = np.mean(preds == labels)
        
        top3_preds = np.argsort(probs, axis=1)[:, -3:]
        top3_acc = np.mean([labels[i] in top3_preds[i] for i in range(len(labels))])
        
        correct_probs = probs[np.arange(len(labels)), labels]
        nll = -np.mean(np.log(np.clip(correct_probs, 1e-12, 1.0)))
        
        ece = compute_ece(probs, labels)
        
        grid_results.append({
            "temperature": round(float(T), 2),
            "val_nll": round(float(nll), 4),
            "val_ece": round(float(ece), 4),
            "val_top1": round(float(top1_acc), 4),
            "val_top3": round(float(top3_acc), 4)
        })
        
        if nll < best_nll:
            best_nll = nll
            best_temp = T
            best_ece = ece
            
    idx_t10 = int(round((1.0 - 0.40) / 0.02))
    t10_nll = grid_results[idx_t10]['val_nll']
    t10_ece = grid_results[idx_t10]['val_ece']

    print(f"\nTEMPERATURE SCALING SELECTION (VALIDATION LOGITS ONLY):")
    print(f"  Selected Temperature T: {best_temp:.2f}")
    print(f"  Validation NLL at T={best_temp:.2f}: {best_nll:.4f} (Uncalibrated T=1.0 NLL: {t10_nll:.4f})")
    print(f"  Validation ECE at T={best_temp:.2f}: {best_ece*100:.2f}% (Uncalibrated T=1.0 ECE: {t10_ece*100:.2f}%)")
    
    report_data = {
        "metadata": {
            "title": "Phase 8 Validation Calibration & Temperature Grid Search Report",
            "eval_samples": len(val_dataset),
            "selection_criterion": "Minimize Validation NLL / ECE",
            "selected_temperature": round(float(best_temp), 2),
            "uncalibrated_temperature": 1.0
        },
        "selected_calibration": {
            "temperature": round(float(best_temp), 2),
            "val_nll": round(float(best_nll), 4),
            "val_ece": round(float(best_ece), 4)
        },
        "grid_results": grid_results
    }
    
    out_json = Path("reports/validation_calibration_study.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved calibration study report to {out_json}")

if __name__ == "__main__":
    main()
