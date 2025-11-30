"""
Demonstrate different approaches to improve model accuracy:
1. More training iterations
2. Adjusting learning rate
3. Feature engineering (polynomial features)
4. Better initialization
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator
import os


def test_more_iterations(data, labels):
    """Test with different numbers of iterations."""
    print("=" * 80)
    print("APPROACH 1: MORE TRAINING ITERATIONS")
    print("=" * 80)

    iteration_counts = [10, 20, 50, 100, 200, 500]
    results = []

    for n_iter in iteration_counts:
        model = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=0.3)
        model.train(data, labels, n_iterations=n_iter)
        predictions = model.predict(data)
        accuracy = np.mean(predictions == labels) * 100
        results.append((n_iter, accuracy, model.beta.copy()))
        print(f"Iterations: {n_iter:4d} | Accuracy: {accuracy:6.2f}% | β: {model.beta}")

    return results


def test_learning_rates(data, labels):
    """Test with different learning rates."""
    print("\n" + "=" * 80)
    print("APPROACH 2: ADJUSTING LEARNING RATE")
    print("=" * 80)

    learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    results = []
    n_iterations = 100

    for eta in learning_rates:
        model = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=eta)
        model.train(data, labels, n_iterations=n_iterations)
        predictions = model.predict(data)
        accuracy = np.mean(predictions == labels) * 100
        results.append((eta, accuracy, model.beta.copy()))
        print(f"Learning rate η: {eta:5.2f} | Accuracy: {accuracy:6.2f}% | β: {model.beta}")

    return results


def add_polynomial_features(data, degree=2):
    """Add polynomial features to the dataset."""
    # Original features: [1, X0, X1]
    # Add: X0^2, X1^2, X0*X1
    bias = data[:, 0:1]
    X0 = data[:, 1:2]
    X1 = data[:, 2:3]

    features = [bias, X0, X1]

    if degree >= 2:
        features.extend([X0**2, X1**2, X0*X1])

    return np.hstack(features)


class PolynomialSigmoidGenerator(SigmoidGenerator):
    """Extended sigmoid generator that supports polynomial features."""

    def __init__(self, beta0=None, eta=0.3, n_features=3):
        super().__init__(beta0=None, eta=eta)
        if beta0 is None:
            beta0 = np.zeros(n_features)
        self.beta = np.array(beta0, dtype=float)


def test_polynomial_features(data, labels):
    """Test with polynomial features."""
    print("\n" + "=" * 80)
    print("APPROACH 3: FEATURE ENGINEERING (POLYNOMIAL FEATURES)")
    print("=" * 80)

    # Add polynomial features
    poly_data = add_polynomial_features(data, degree=2)
    print(f"Original features: {data.shape[1]} | Enhanced features: {poly_data.shape[1]}")
    print("Added features: X0², X1², X0×X1")
    print()

    # Train with polynomial features
    beta0_poly = np.zeros(poly_data.shape[1])
    model = PolynomialSigmoidGenerator(beta0=beta0_poly, eta=0.3, n_features=poly_data.shape[1])
    model.train(poly_data, labels, n_iterations=100)

    predictions = model.predict(poly_data)
    accuracy = np.mean(predictions == labels) * 100

    print(f"Polynomial model accuracy: {accuracy:.2f}%")
    print(f"Weights: {model.beta}")

    return accuracy, model


def test_better_initialization(data, labels):
    """Test with different initialization strategies."""
    print("\n" + "=" * 80)
    print("APPROACH 4: BETTER WEIGHT INITIALIZATION")
    print("=" * 80)

    initializations = {
        'Zeros': [0, 0, 0],
        'Original': [0, 0.3, -0.5],
        'Small Random': np.random.randn(3) * 0.1,
        'Xavier': np.random.randn(3) * np.sqrt(2.0 / 3),
        'Ones': [1, 1, 1],
        'Negative': [-1, -1, -1]
    }

    results = []

    for name, beta0 in initializations.items():
        model = SigmoidGenerator(beta0=beta0, eta=0.3)
        model.train(data, labels, n_iterations=100)
        predictions = model.predict(data)
        accuracy = np.mean(predictions == labels) * 100
        results.append((name, accuracy, model.beta.copy()))
        print(f"{name:15s} | Initial β: {str(beta0):40s} | Accuracy: {accuracy:6.2f}%")

    return results


def plot_accuracy_comparison(iteration_results, lr_results, output_path):
    """Create comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Iterations vs Accuracy
    iterations = [r[0] for r in iteration_results]
    accuracies = [r[1] for r in iteration_results]

    ax1.plot(iterations, accuracies, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Iterations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Impact of Training Iterations on Accuracy', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Add annotations
    max_acc_idx = np.argmax(accuracies)
    ax1.annotate(f'Best: {accuracies[max_acc_idx]:.2f}%',
                xy=(iterations[max_acc_idx], accuracies[max_acc_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=12, fontweight='bold')

    # Plot 2: Learning Rate vs Accuracy
    learning_rates = [r[0] for r in lr_results]
    lr_accuracies = [r[1] for r in lr_results]

    ax2.plot(learning_rates, lr_accuracies, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Learning Rate (η)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Impact of Learning Rate on Accuracy', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # Add annotations
    max_lr_idx = np.argmax(lr_accuracies)
    ax2.annotate(f'Best: {lr_accuracies[max_lr_idx]:.2f}%\nη={learning_rates[max_lr_idx]}',
                xy=(learning_rates[max_lr_idx], lr_accuracies[max_lr_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to: {output_path}")


def create_best_model_visualization(data, labels, output_path):
    """Train and visualize the best performing model."""
    print("\n" + "=" * 80)
    print("CREATING BEST MODEL VISUALIZATION")
    print("=" * 80)

    # Best parameters from experiments
    best_eta = 1.0
    best_iterations = 500

    print(f"Training with best parameters: η={best_eta}, iterations={best_iterations}")

    model = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=best_eta)
    model.train(data, labels, n_iterations=best_iterations)

    predictions = model.predict(data)
    accuracy = np.mean(predictions == labels) * 100

    print(f"Best model accuracy: {accuracy:.2f}%")
    print(f"Final weights: {model.beta}")

    # Create visualization
    X0 = data[:, 1]
    X1 = data[:, 2]

    fig, ax = plt.subplots(figsize=(14, 10))

    correct = predictions == labels
    incorrect = predictions != labels

    mask_class0 = labels == 0
    mask_class1 = labels == 1

    # Correctly classified
    ax.scatter(X0[mask_class0 & correct], X1[mask_class0 & correct],
              c='blue', marker='o', s=100, alpha=0.6,
              edgecolors='black', linewidth=1,
              label=f'Class 0 Correct ({np.sum(mask_class0 & correct)})')

    ax.scatter(X0[mask_class1 & correct], X1[mask_class1 & correct],
              c='red', marker='s', s=100, alpha=0.6,
              edgecolors='black', linewidth=1,
              label=f'Class 1 Correct ({np.sum(mask_class1 & correct)})')

    # Misclassified
    ax.scatter(X0[incorrect], X1[incorrect],
              c='yellow', marker='X', s=200, alpha=0.9,
              edgecolors='black', linewidth=2,
              label=f'Misclassified ({np.sum(incorrect)})')

    # Decision boundary
    beta = model.beta
    if beta[2] != 0:
        x0_line = np.linspace(0, 1, 100)
        x1_line = -(beta[0] + beta[1] * x0_line) / beta[2]
        ax.plot(x0_line, x1_line, 'g-', linewidth=3,
               label=f'Decision Boundary')

    ax.set_xlabel('X0', fontsize=14, fontweight='bold')
    ax.set_ylabel('X1', fontsize=14, fontweight='bold')
    ax.set_title(f'Best Model Performance\nη={best_eta}, Iterations={best_iterations}, Accuracy={accuracy:.2f}%',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.text(0.02, 0.98,
           f'Accuracy: {accuracy:.2f}%\n'
           f'Correct: {np.sum(correct)}/{len(labels)}\n'
           f'η: {best_eta}\n'
           f'Iterations: {best_iterations}',
           transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Best model visualization saved to: {output_path}")

    return accuracy, model


def main():
    print("=" * 80)
    print("MODEL ACCURACY IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print()

    # Create output directory
    os.makedirs("results/visualizations", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Generate dataset
    print("Generating 6000-sample dataset...")
    generator = DatasetGenerator(n_samples=6000)
    data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)
    print(f"Dataset: {len(data)} samples")
    print()

    # Test different approaches
    iteration_results = test_more_iterations(data, labels)
    lr_results = test_learning_rates(data, labels)
    poly_accuracy, poly_model = test_polynomial_features(data, labels)
    init_results = test_better_initialization(data, labels)

    # Create comparison plots
    plot_accuracy_comparison(iteration_results, lr_results,
                            "results/visualizations/accuracy_comparison.png")

    # Create best model visualization
    best_accuracy, best_model = create_best_model_visualization(
        data, labels, "results/visualizations/best_model_performance.png")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF IMPROVEMENTS")
    print("=" * 80)
    print(f"Baseline (20 iterations, η=0.3):     ~60%")
    print(f"More iterations (500, η=0.3):        {iteration_results[-1][1]:.2f}%")
    print(f"Optimized learning rate (η=1.0):     {[r[1] for r in lr_results if r[0]==1.0][0]:.2f}%")
    print(f"Polynomial features (100 iter):      {poly_accuracy:.2f}%")
    print(f"Best combination (500 iter, η=1.0):  {best_accuracy:.2f}%")
    print("=" * 80)

    # Save summary
    with open("results/AccuracyImprovementSummary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL ACCURACY IMPROVEMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("APPROACH 1: MORE ITERATIONS\n")
        f.write("-" * 80 + "\n")
        for n_iter, acc, beta in iteration_results:
            f.write(f"Iterations: {n_iter:4d} | Accuracy: {acc:6.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("APPROACH 2: LEARNING RATE OPTIMIZATION\n")
        f.write("-" * 80 + "\n")
        for eta, acc, beta in lr_results:
            f.write(f"Learning rate η: {eta:5.2f} | Accuracy: {acc:6.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("APPROACH 3: POLYNOMIAL FEATURES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {poly_accuracy:.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("BEST PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Learning rate: 1.0\n")
        f.write(f"Iterations: 500\n")
        f.write(f"Final weights: {best_model.beta}\n")
        f.write("=" * 80 + "\n")

    print("\nSummary saved to: results/AccuracyImprovementSummary.txt")
    print("\nFiles created:")
    print("  - results/visualizations/accuracy_comparison.png")
    print("  - results/visualizations/best_model_performance.png")
    print("  - results/AccuracyImprovementSummary.txt")
    print()


if __name__ == "__main__":
    main()
