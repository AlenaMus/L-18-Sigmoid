# L-18-Sigmoid

Binary Classification using Sigmoid Activation Function with Pure Mathematical Implementation

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-required-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Project Overview

This project implements a binary classifier using **logistic regression with gradient descent**, built entirely with pure mathematics using only NumPy. No external machine learning libraries are used - all calculations are performed using mathematical formulas from scratch.

### Key Features

- ✅ **Pure Mathematical Implementation** - No sklearn, TensorFlow, or PyTorch
- ✅ **Batch Gradient Descent** - Optimized learning algorithm
- ✅ **Detailed Iteration Tracking** - Complete visibility into training process
- ✅ **Comprehensive Visualizations** - 8 different plots showing model behavior
- ✅ **High Accuracy** - Achieves 99.78% accuracy on 6000-sample dataset
- ✅ **Configurable** - Easy to adjust hyperparameters via `.env` file
- ✅ **Educational** - Includes detailed explanations and guides

## 🎯 Results

| Configuration | Iterations | Learning Rate | Accuracy |
|--------------|-----------|---------------|----------|
| Baseline | 20 | 0.3 | 59.98% |
| Improved | 100 | 0.3 | 98.72% |
| **Optimized** | **500** | **1.0** | **99.78%** ⭐ |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AlenaMus/L-18-Sigmoid.git
cd L-18-Sigmoid

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator

# Generate dataset
generator = DatasetGenerator(n_samples=6000)
data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)

# Train model
model = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=1.0)
model.train(data, labels, n_iterations=500)

# Predict
predictions = model.predict(data)
accuracy = (predictions == labels).mean() * 100
print(f"Accuracy: {accuracy:.2f}%")
```

### Run Complete Analysis

```bash
# Step 1: Generate dataset and visualizations
python3 step1_generate_dataset.py

# Step 2: Train with detailed tracking
python3 step2_sigmoid_training.py

# Step 3: Full analysis with 6000 samples
python3 step3_full_analysis.py

# Custom training with your settings
python3 train_with_custom_iterations.py
```

## 📊 Project Structure

```
L-18-Sigmoid/
├── Core Classes
│   ├── dataset_generator.py      # Dataset generation (122 lines)
│   ├── sigmoid_generator.py      # Sigmoid classifier (168 lines)
│   └── visualizer.py             # Visualization tools (81 lines)
│
├── Step-by-Step Scripts
│   ├── step1_generate_dataset.py
│   ├── step2_sigmoid_training.py
│   └── step3_full_analysis.py
│
├── Analysis Tools
│   ├── improve_model_accuracy.py
│   ├── create_enhanced_visualizations.py
│   └── explain_iterations_vs_samples.py
│
├── Easy Configuration
│   ├── train_with_custom_iterations.py
│   └── .env                      # Hyperparameter settings
│
├── Results (5.0 MB)
│   ├── *.txt                     # 9 dataset tables
│   └── visualizations/           # 8 PNG graphs
│
└── Documentation
    ├── README.md
    ├── HOW_TO_INCREASE_ITERATIONS.md
    └── results/README.md
```

## 🧮 Mathematical Implementation

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
```

### Linear Combination
```
z = β₀ + β₁·X₀ + β₂·X₁
```

### Gradient Descent
```
∇J = (1/n) · X^T · (σ(z) - y)
```

### Weight Update
```
β_new = β_old - η · ∇J
```

All implemented using **only NumPy** - no external ML libraries!

## 📈 Visualizations

The project generates 8 comprehensive visualizations:

1. **Dataset Distribution** (100 samples)
2. **Dataset Distribution** (6000 samples)
3. **Dataset with Predictions** - Shows correct/incorrect classifications
4. **Training Progress** - Likelihood and error vs iterations
5. **Prediction Evolution** - 6-panel showing model learning
6. **Sample Ranges Combined** - First 100 and last 50 samples
7. **First 100 Samples** - Detailed view with 20% accuracy
8. **Last 50 Samples** - Detailed view with 100% accuracy

Plus 2 accuracy comparison graphs showing impact of iterations and learning rate.

## 🎓 Educational Resources

### Guides Included:
- **HOW_TO_INCREASE_ITERATIONS.md** - 5 methods to adjust training
- **explain_iterations_vs_samples.py** - Clarifies iterations vs samples
- **improve_model_accuracy.py** - 4 approaches to boost performance

### Key Concepts Explained:
- Batch vs Stochastic Gradient Descent
- Why 100-500 iterations is optimal for 6000 samples
- Learning rate impact on convergence
- Feature engineering with polynomial features

## 🔧 Configuration

### Via `.env` file:
```env
LEARNING_RATE=0.3
NUM_ITERATIONS=500
THRESHOLD=0.5
TRAIN_TEST_SPLIT=0.8
RANDOM_SEED=42
VERBOSE=True
```

### Via code:
```python
model = SigmoidGenerator(
    beta0=[0, 0.3, -0.5],  # Initial weights
    eta=1.0                 # Learning rate
)
model.train(data, labels, n_iterations=500)
```

## 📊 Sample Results

### Dataset Table (CSV format)
```csv
Idx,Bias,X0,X1,y,It0_β0,It0_β1,It0_β2,It0_σ(z),It0_Error,...
0,1.0,0.349671,0.109219,0,0.000000,0.300000,-0.500000,0.512570,-0.512570,...
```

### Beta Weight Evolution
```
Iteration    β0 (bias)    β1 (X0)      β2 (X1)      Avg Error
0            0.000000     0.300000     -0.500000    0.510181
100          -1.425586    1.981365     1.272130     0.014289
500          -7.823338    8.078539     7.802539     0.002231
```

## 🏆 Performance Improvements

We tested 4 different approaches to improve accuracy:

| Approach | Result | Improvement |
|----------|--------|-------------|
| 1. More Iterations (500) | 99.77% | +39.79% |
| 2. Optimized Learning Rate (η=2.0) | 99.82% | +39.84% |
| 3. Polynomial Features | 99.73% | +39.75% |
| 4. Best Combination | 99.78% | +39.80% |

See `results/AccuracyImprovementSummary.txt` for details.

## 📦 Requirements

- Python 3.8+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- python-dotenv >= 0.19.0

## 🤝 Contributing

This is an educational project for AI Development Course - Lesson 18 Homework.

## 📄 License

MIT License - Feel free to use for learning purposes.

## 👤 Author

**Alena Mus**
- GitHub: [@AlenaMus](https://github.com/AlenaMus)

## 🙏 Acknowledgments

- AI Development Course - Lesson 18
- Pure mathematical implementation without ML libraries
- Built with Claude Code assistance

## 📝 Citation

If you use this code for educational purposes, please cite:

```bibtex
@software{l18_sigmoid,
  author = {Alena Mus},
  title = {L-18-Sigmoid: Binary Classification with Pure Mathematical Implementation},
  year = {2025},
  url = {https://github.com/AlenaMus/L-18-Sigmoid}
}
```

---

**⭐ Star this repository if you find it helpful for learning!**
