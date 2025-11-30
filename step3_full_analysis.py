"""
Step 3: Full Analysis with 6000 Samples

Creates:
1. 6000-sample dataset
2. Detailed table with all iterations (betas, sigmoid, errors)
3. Dataset distribution visualization
4. Training progress graph (likelihood and error vs iterations)
"""

import numpy as np
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator
from visualizer import DataVisualizer
import matplotlib.pyplot as plt


def generate_detailed_table(sigmoid_gen, data, labels):
    """Generate table with iteration columns for all samples."""
    lines = []
    lines.append("=" * 200)
    lines.append("DETAILED ITERATION TABLE - ALL SAMPLES")
    lines.append("=" * 200)
    lines.append(f"Total Samples: {len(data)}")
    lines.append(f"Learning Rate η: {sigmoid_gen.eta}")
    lines.append("=" * 200)

    # Header
    header = f"{'Idx':<6} {'Bias':<6} {'X0':<10} {'X1':<10} {'y':<4}"

    # Add iteration columns
    for iter_data in sigmoid_gen.iterations_data:
        iteration = iter_data['iteration']
        beta = iter_data['beta']
        header += f" | It{iteration}_β0={beta[0]:.3f},β1={beta[1]:.3f},β2={beta[2]:.3f}"
        header += f" | σ(z)_It{iteration:<6} | Err_It{iteration:<6}"

    lines.append(header[:200])  # Truncate if too long
    lines.append("-" * 200)

    # Data rows (show first 20 samples)
    for i in range(min(20, len(data))):
        row = f"{i:<6} {data[i,0]:<6.1f} {data[i,1]:<10.6f} {data[i,2]:<10.6f} {int(labels[i]):<4}"

        for iter_data in sigmoid_gen.iterations_data:
            prob = iter_data['probabilities'][i]
            err = iter_data['errors'][i]
            row += f" | {prob:<14.6f} | {err:<12.6f}"

        lines.append(row[:200])

    if len(data) > 20:
        lines.append(f"... ({len(data) - 20} more samples)")

    lines.append("=" * 200)
    return "\n".join(lines)


def compute_log_likelihood(probabilities, y):
    """
    Compute log-likelihood: L = Σ[y*log(p) + (1-y)*log(1-p)]
    """
    epsilon = 1e-10  # Prevent log(0)
    p = np.clip(probabilities, epsilon, 1 - epsilon)
    log_likelihood = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return log_likelihood


def plot_training_progress(sigmoid_gen, data, labels, output_path):
    """Create graph showing likelihood and error vs iterations."""
    iterations = []
    likelihoods = []
    avg_errors = []

    for iter_data in sigmoid_gen.iterations_data:
        iteration = iter_data['iteration']
        probs = iter_data['probabilities']
        errors = iter_data['errors']

        likelihood = compute_log_likelihood(probs, labels)
        avg_error = np.mean(np.abs(errors))

        iterations.append(iteration)
        likelihoods.append(likelihood)
        avg_errors.append(avg_error)

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot likelihood
    color1 = 'blue'
    ax1.set_xlabel('Iteration Number', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Log-Likelihood', fontsize=14, fontweight='bold', color=color1)
    line1 = ax1.plot(iterations, likelihoods, color=color1, marker='o',
                     linewidth=2, markersize=6, label='Log-Likelihood')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for error
    ax2 = ax1.twinx()
    color2 = 'red'
    ax2.set_ylabel('Average Absolute Error', fontsize=14, fontweight='bold', color=color2)
    line2 = ax2.plot(iterations, avg_errors, color=color2, marker='s',
                     linewidth=2, markersize=6, label='Avg Error')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title and legend
    plt.title('Training Progress: Likelihood & Error vs Iterations',
              fontsize=16, fontweight='bold', pad=20)

    # Combine legends
    lines = line1 + line2
    labels_legend = [l.get_label() for l in lines]
    ax1.legend(lines, labels_legend, loc='best', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training progress graph saved to: {output_path}")


def main():
    print("=" * 100)
    print("STEP 3: FULL ANALYSIS WITH 6000 SAMPLES")
    print("=" * 100)
    print()

    # Generate 6000-sample dataset
    print("Generating 6000-sample dataset...")
    generator = DatasetGenerator(n_samples=6000)
    data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)
    print(f"Dataset generated: {len(data)} samples")
    print(f"  - Class 0: {np.sum(labels == 0)} samples")
    print(f"  - Class 1: {np.sum(labels == 1)} samples")
    print()

    # Visualize dataset distribution
    print("Creating dataset distribution visualization...")
    visualizer = DataVisualizer(output_dir="visualizations")
    visualizer.plot_dataset(data, labels,
                           title="6000-Sample Dataset Distribution",
                           filename="dataset_6000_distribution.png")
    print()

    # Train sigmoid generator
    print("Training Sigmoid Generator...")
    beta0 = [0, 0.3, -0.5]
    eta = 0.3
    n_iterations = 20

    sigmoid_gen = SigmoidGenerator(beta0=beta0, eta=eta)
    print(f"Initial β0: {beta0}")
    print(f"Learning rate η: {eta}")
    print(f"Iterations: {n_iterations}")
    print()

    sigmoid_gen.train(data, labels, n_iterations=n_iterations)
    print("Training completed!")
    print()

    # Save summary table
    print("Saving beta progression summary...")
    summary_table = sigmoid_gen.get_summary_table()
    with open("BetaProgressionSummary_6000.txt", 'w') as f:
        f.write(summary_table)
    print("Saved: BetaProgressionSummary_6000.txt")
    print()

    # Create training progress graph
    print("Creating training progress graph...")
    plot_training_progress(sigmoid_gen, data, labels,
                          "visualizations/training_progress.png")
    print()

    # Display summary
    print(summary_table)
    print()

    print("=" * 100)
    print("STEP 3 COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print("Files created:")
    print("  - visualizations/dataset_6000_distribution.png")
    print("  - visualizations/training_progress.png")
    print("  - BetaProgressionSummary_6000.txt")


if __name__ == "__main__":
    main()
