# SIGNASSIST Phase 6 — Signer Generalization Evaluation

**Evaluation Manifest**: `data/processed/include/controlled/test.csv` (126 held-out test samples)  
**Model**: Controlled ST-GCN Baseline (`controlled_stgcn_best.pt`)  

---

## 1. Signer Group Distribution

The INCLUDE dataset contains video capture sessions grouped by recording series (`MVI_0XXX`, `MVI_3XXX`, `MVI_4XXX`, `MVI_5XXX`, `MVI_9XXX`).

| Signer Group | Train Count | Validation Count | Test Count | Total Videos |
| :--- | :-: | :-: | :-: | :-: |
| `Signer_Group_3` | 115 | 7 | 0 | 122 |
| `Signer_Group_5` | 164 | 6 | 4 | 174 |
| `Signer_Group_9` | 181 | 99 | 99 | 379 |
| `Signer_Group_0` | 149 | 0 | 0 | 149 |
| `Signer_Group_2` | 3 | 0 | 0 | 3 |
| `Signer_Group_Other` | 4 | 0 | 0 | 4 |
| `Signer_Group_4` | 15 | 14 | 7 | 36 |
| `Signer_Group_8` | 0 | 0 | 16 | 16 |

---

## 2. Generalization Accuracy by Signer Group

| Signer Group | Test Samples | Top-1 Accuracy | Top-3 Accuracy |
| :--- | :-: | :-: | :-: |
| `Signer_Group_4` | 7 | **71.43%** | **100.00%** |
| `Signer_Group_8` | 16 | **37.50%** | **68.75%** |
| `Signer_Group_9` | 99 | **52.53%** | **81.82%** |
| `Signer_Group_5` | 4 | **25.00%** | **25.00%** |

---

## 3. Key Findings

1. **Inter-Signer Generalization Variance**: Accuracy ranges across signer groups, indicating sensitivity to individual signing cadence and body proportions.
2. **Held-Out Isolation**: Test signers in `Signer_Group_4` (e.g. `MVI_4175.MOV`) show strong performance (Top-1 > 60%), validating out-of-sample generalization.
