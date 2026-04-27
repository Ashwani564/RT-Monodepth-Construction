# Figure 3: Training Dynamics of YOLOv11n

This directory contains training curve visualizations for the YOLOv11n model trained on the Construction Site Safety dataset.

## Generated Plots

| File | Description | Dimensions |
|------|-------------|------------|
| `figure3_training_curves.png` | Combined figure with both plots | 1800×675 px |
| `figure3_training_curves.pdf` | PDF version for LaTeX | Vector |
| `loss_curves.png` | Left plot only (Loss vs Epochs) | 900×675 px |
| `map_curves.png` | Right plot only (mAP vs Epochs) | 900×675 px |

## Training Configuration

- **Model**: YOLOv11n (nano variant)
- **Dataset**: Construction Site Safety (10 classes)
- **Epochs**: 300
- **Batch Size**: 16
- **Image Size**: 640×640

## Final Training Metrics (Epoch 300)

| Metric | Value |
|--------|-------|
| Box Loss | 0.4545 |
| Class Loss | 0.2886 |
| DFL Loss | 0.8761 |
| **mAP@50** | **0.8466** |
| **mAP@50-95** | **0.5945** |
| Precision | 0.9254 |
| Recall | 0.7917 |

## Plot Details

### (a) Training Loss vs. Epochs
- **Box Loss** (blue): Bounding box regression loss
- **Class Loss** (red): Classification loss
- **DFL Loss** (green, dashed): Distribution Focal Loss

### (b) mAP vs. Epochs
- **mAP@50** (purple): Mean Average Precision at IoU=0.50
- **mAP@50-95** (orange): Mean Average Precision at IoU=0.50:0.95
- **Vertical line at epoch 150**: Indicates model stability point

## Key Observations

1. **Rapid Initial Learning**: Both loss and mAP metrics improve significantly in the first 50 epochs
2. **Stability Point (~150 epochs)**: Model performance stabilizes, with marginal gains thereafter
3. **Final Performance**: mAP@50 reaches 84.7%, mAP@50-95 reaches 59.5%
4. **No Overfitting**: Loss curves show smooth convergence without signs of overfitting

## How to Use

### Option 1: Direct Image
The `figure3_training_curves.png` is ready to use directly in your paper.

### Option 2: Draw.io Template
1. Open `figure3_template.drawio` in VS Code or draw.io
2. Right-click the placeholder → Edit → Image
3. Import `figure3_training_curves.png`
4. Delete the instruction box
5. Export as needed

### Option 3: LaTeX
Use the PDF version directly:
```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/figure3_training_curves.pdf}
    \caption{Training dynamics of YOLOv11n over 300 epochs...}
    \label{fig:training_curves}
\end{figure}
```

## Caption for Paper

> **Figure 3**: Training dynamics of YOLOv11n over 300 epochs on the Construction Site Safety dataset. (a) Training loss curves showing Box Loss, Class Loss, and DFL Loss converging smoothly. (b) mAP metrics showing model performance stabilizing after approximately 150 epochs. Final performance: mAP@50 = 0.847, mAP@50-95 = 0.595.
