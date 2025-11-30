# SigmoidClassification Project

Binary classification using sigmoid activation function with pure mathematical implementation.

## Project Overview

This project implements a binary classifier using logistic regression with gradient descent. All calculations are performed using pure mathematics (no external sigmoid libraries) with only numpy for matrix operations.

## Project Structure

```
SigmoidClassification/
├── .env                                  # Configuration parameters
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
│
├── Core Classes (< 160 lines each)
├── dataset_generator.py                 # DatasetGenerator class (122 lines)
├── sigmoid_generator.py                 # SigmoidGenerator class (159 lines)
├── visualizer.py                        # DataVisualizer class (81 lines)
│
├── Step-by-step Scripts
├── step1_generate_dataset.py            # Generate 100-sample dataset & visualize
├── step2_sigmoid_training.py            # Train with detailed iteration tracking
├── step3_full_analysis.py               # Full analysis with 6000 samples
│
├── Utility Scripts
├── create_detailed_dataset_table.py     # Create detailed iteration tables
├── create_enhanced_dataset_table.py     # Create enhanced dataset with columns
├── verify_math.py                       # Verify mathematical correctness
│
└── results/                             # All generated outputs
    ├── README.md                        # Results documentation
    ├── DataSet*.txt                     # Dataset tables (4 files)
    ├── BetaProgressionSummary*.txt      # Beta weight summaries
    ├── DetailedIterationTable.txt       # Detailed iteration data
    ├── WeightProgressionTable.txt       # Weight progression
    └── visualizations/                  # Generated plots
        ├── dataset_visualization.png
        ├── dataset_6000_distribution.png
        └── training_progress.png
```

## Features

### 1. Dataset Generation
- Generates two separated groups using normal distribution
- Each sample: 3D vector (1, X0, X1) where 1 is bias term
- Group 0 (y=0) centered at (0.3, 0.3)
- Group 1 (y=1) centered at (0.7, 0.7)
- All values in [0, 1] range

### 2. Pure Mathematical Implementation
- **Sigmoid**: σ(z) = 1 / (1 + e^(-z))
- **Linear Model**: z = β₀ + β₁×X0 + β₂×X1
- **Gradient Descent**: ∇J = (1/n) × X^T × (σ(z) - y)
- **Weight Update**: β_new = β_old - η × ∇J

### 3. Detailed Tracking
- Beta weights at each iteration
- Sigmoid probabilities for each sample
- Errors (y_true - σ(z)) for each sample
- Log-likelihood and average error metrics

### 4. Visualizations
- Dataset distribution scatter plots
- Training progress graphs (likelihood & error vs iterations)

## Configuration (.env)

```
LEARNING_RATE=0.01
NUM_ITERATIONS=1000
THRESHOLD=0.5
TRAIN_TEST_SPLIT=0.8
RANDOM_SEED=42
LOG_LEVEL=INFO
VERBOSE=True
```

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Step by Step

**Step 1: Generate Dataset**
```bash
python3 step1_generate_dataset.py
```
Output: DataSet.txt, dataset_visualization.png

**Step 2: Train with Iteration Tracking**
```bash
python3 step2_sigmoid_training.py
```
Output: BetaProgressionSummary.txt, DetailedIterationTable.txt

**Step 3: Full Analysis (6000 samples)**
```bash
python3 step3_full_analysis.py
```
Output: BetaProgressionSummary_6000.txt, training_progress.png

**Create Enhanced Tables**
```bash
python3 create_enhanced_dataset_table.py
```
Output: DataSet_Enhanced.txt, DataSet_Compact_CSV.txt

**Verify Mathematics**
```bash
python3 verify_math.py
```
Output: Mathematical step-by-step verification

## Key Results (6000 samples, 20 iterations)

| Metric | Value |
|--------|-------|
| Initial β | [0.000, 0.300, -0.500] |
| Final β | [-0.157, 0.763, -0.018] |
| Initial Error | 0.510 |
| Final Error | 0.464 |
| Error Reduction | ~9% |
| Learning Rate η | 0.3 |

## Algorithm Steps

Each iteration performs:

1. **Compute Probabilities**: Calculate σ(z) for all samples using current β
2. **Calculate Errors**: Compute y_true - σ(z) for each sample
3. **Compute Gradient**: Calculate ∇J using matrix operations
4. **Update Weights**: Apply β_new = β_old - η × ∇J

## Output Files

### Dataset Tables
- **DataSet.txt**: Original 100 samples
- **DataSet_Enhanced.txt**: 6000 samples with iteration columns (formatted)
- **DataSet_Compact_CSV.txt**: 6000 samples with iteration columns (CSV)
- **DataSet_6000_WithIterations.txt**: 6000 samples with iteration sections

### Summary Tables
- **BetaProgressionSummary.txt**: Beta evolution (100 samples)
- **BetaProgressionSummary_6000.txt**: Beta evolution (6000 samples)
- **DetailedIterationTable.txt**: Detailed iteration data with probabilities/errors

### Visualizations
- **dataset_visualization.png**: 100-sample distribution
- **dataset_6000_distribution.png**: 6000-sample distribution
- **training_progress.png**: Likelihood & error vs iterations

## Technical Notes

- All calculations use only numpy and basic mathematics
- No external sigmoid or machine learning libraries
- Class sizes: All under 160 lines
- Supports large datasets (tested with 6000 samples)
- Configurable via .env file

## License

Educational project for AI Development Course - Lesson 18 Homework
