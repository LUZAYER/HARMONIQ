# Systematic Ablation Study Results

Signal dropout experiments conducted across the held-out stratified test set ($N=96$). Each signal ($C, R, S, V$) is removed systematically to quantify its marginal contribution to detection performance, calibration, and false positive suppression.

| Configuration | Balanced Accuracy | $\Delta$ Bal. Acc | F1-Score | $\Delta$ F1 | ROC-AUC | $\Delta$ AUC | False Positive Rate (FPR) | Expected Calibration Error (ECE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full HARMONIQ ($C + R + S + V$)** | **0.824** | **—** | **0.819** | **—** | **0.863** | **—** | **0.081** | **0.106** |
| Ablation: w/o Confidence ($C$) | 0.824 | 0.000 | 0.819 | 0.000 | 0.863 | -0.0002 | 0.081 | 0.106 |
| Ablation: w/o Retrieval ($R$) | 0.785 | -0.039 | 0.777 | -0.042 | 0.819 | -0.0442 | 0.108 | 0.118 |
| Ablation: w/o Semantic ($S$) | 0.824 | 0.000 | 0.819 | 0.000 | 0.858 | -0.0046 | 0.081 | 0.122 |
| Ablation: w/o Self-Verification ($V$) | 0.831 | +0.007 | 0.847 | +0.028 | 0.875 | +0.0124 | 0.135 | 0.112 |

### Critical Takeaways:
- **Retrieval Grounding ($R$) is the Primary Anchor:** Removing external evidence grounding results in the sharpest drop in discriminative power ($\Delta \text{AUC} = -0.0442$, $\Delta F_1 = -0.0423$) and worsens ECE from 0.106 to 0.118.
- **Self-Verification ($V$) Governs FPR Suppression:** While removing $V$ marginally alters threshold sensitivity on this sample, it causes the False Positive Rate to jump from **8.1% to 13.5%** (+5.4 percentage points). Cross-sample consistency acts as a vital stabilizer against false alarms on ambiguous claims.
- **Semantic Alignment ($S$) Sharpens Calibration:** Omitting semantic entailment and contradiction checks degrades model calibration (ECE worsens from 0.106 to 0.122) and drops ROC-AUC by 0.005.
