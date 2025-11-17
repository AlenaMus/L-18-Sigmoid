# SigmoidClassification - Results

This folder contains all generated outputs from the sigmoid classification project.

## Dataset Tables

### Original Dataset (100 samples)
- **DataSet.txt** - Basic dataset table with 100 samples showing Index, Bias, X0, X1, y

### Training Progress Tables
- **WeightProgressionTable.txt** - Beta weight progression for 100-sample dataset (50 iterations)
- **DetailedIterationTable.txt** - Detailed iteration table with probabilities and errors for each sample
- **BetaProgressionSummary.txt** - Summary of beta weights evolution (100 samples)

### Large Dataset (6000 samples)
- **BetaProgressionSummary_6000.txt** - Summary of beta weights evolution (6000 samples, 20 iterations)
- **DataSet_6000_WithIterations.txt** - Detailed iteration results organized by iteration sections
- **DataSet_Enhanced.txt** - Formatted table with iteration columns showing β, σ(z), and errors
- **DataSet_Compact_CSV.txt** - CSV format with all iteration data (easy to import into Excel)
- **DataSet_First100_Last50.txt** - CSV table showing first 100 and last 50 samples with all iteration data

## Visualizations

### Dataset Distribution Plots
- **dataset_visualization.png** - Scatter plot of 100-sample dataset showing two separated classes
- **dataset_6000_distribution.png** - Scatter plot of 6000-sample dataset distribution

### Model Predictions & Decision Boundary
- **dataset_with_predictions.png** - Enhanced visualization showing:
  - Blue circles: Class 0 correctly classified
  - Red squares: Class 1 correctly classified
  - Cyan X: Class 0 misclassified (false positives)
  - Orange X: Class 1 misclassified (false negatives)
  - Green line: Decision boundary (where σ(z) = 0.5)
  - Gray contours: Probability levels (0.1, 0.3, 0.5, 0.7, 0.9)

### Training Progress
- **training_progress.png** - Dual-axis plot showing:
  - Blue line: Log-Likelihood vs iterations (increasing)
  - Red line: Average Absolute Error vs iterations (decreasing)

### Model Evolution
- **prediction_evolution.png** - 6-panel visualization showing how predictions evolve:
  - Iterations 0, 5, 10, 15, 20 displayed
  - Yellow X marks: Misclassified samples
  - Green line: Decision boundary evolution
  - Shows accuracy improvement from 42% to 60%

### Sample Range Visualizations
- **sample_ranges_combined.png** - Combined view highlighting first 100 and last 50 samples:
  - First 100 samples: Bold circles/squares (20% accuracy)
  - Last 50 samples: Diamond markers (100% accuracy)
  - Other samples: Light/small markers for context
  - Shows decision boundary and all class distinctions

- **first_100_samples.png** - Detailed view of first 100 samples only:
  - Large bold markers for clear visibility
  - Sample indices labeled (every 10th)
  - Yellow X marks for misclassified (80 samples)
  - Blue circles for Class 0 correct (20 samples)
  - Accuracy: 20.00%

- **last_50_samples.png** - Detailed view of last 50 samples only:
  - Diamond-shaped markers
  - Sample indices labeled (every 5th: 5950, 5955, etc.)
  - Dark red/blue with colored edges
  - All samples correctly classified (Class 1)
  - Accuracy: 100.00%

## Key Results

### Initial Parameters
- β₀ = [0, 0.3, -0.5]
- Learning rate η = 0.3

### Final Results (20 iterations, 6000 samples)
- Final β = [-0.157, 0.763, -0.018]
- Initial Error: 0.510
- Final Error: 0.464
- Error Reduction: ~9%

## Mathematical Implementation

All calculations use pure mathematical formulas:
1. **Sigmoid**: σ(z) = 1 / (1 + e^(-z))
2. **Linear Combination**: z = β₀ + β₁×X0 + β₂×X1
3. **Gradient**: ∇J = (1/n) × X^T × (σ(z) - y)
4. **Weight Update**: β_new = β_old - η × ∇J

No external sigmoid libraries used - only numpy and basic mathematics.
