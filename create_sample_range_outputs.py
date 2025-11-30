"""
Create tables and visualizations for specific sample ranges:
- First 100 samples
- Last 50 samples
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator
import os


def create_range_table(data, labels, sigmoid_gen, sample_ranges, filename):
    """
    Create table showing specific sample ranges.

    Args:
        sample_ranges: List of tuples (start, end, label)
    """
    lines = []
    lines.append("=" * 200)
    lines.append("DATASET TABLE - SELECTED SAMPLE RANGES")
    lines.append("=" * 200)
    lines.append(f"Total Dataset: {len(data)} samples")
    lines.append(f"Initial β0: [0, 0.3, -0.5], Learning Rate η: {sigmoid_gen.eta}")
    lines.append("")

    for start, end, range_label in sample_ranges:
        lines.append(f"\n{range_label}: Samples {start} to {end-1}")
        lines.append("=" * 200)

    lines.append("")
    lines.append("=" * 200)

    # CSV Header
    header = "Range,Idx,Bias,X0,X1,y"
    selected_iterations = [0, 5, 10, 15, 20]

    for iter_num in selected_iterations:
        if iter_num < len(sigmoid_gen.iterations_data):
            header += f",It{iter_num}_β0,It{iter_num}_β1,It{iter_num}_β2,It{iter_num}_σ(z),It{iter_num}_Error"

    lines.append(header)
    lines.append("-" * 200)

    # Add data for each range
    for start, end, range_label in sample_ranges:
        for i in range(start, min(end, len(data))):
            row = f"{range_label},{i},{data[i,0]:.1f},{data[i,1]:.6f},{data[i,2]:.6f},{int(labels[i])}"

            for iter_num in selected_iterations:
                if iter_num < len(sigmoid_gen.iterations_data):
                    iter_data = sigmoid_gen.iterations_data[iter_num]
                    beta = iter_data['beta']
                    prob = iter_data['probabilities'][i]
                    err = iter_data['errors'][i]

                    row += f",{beta[0]:.6f},{beta[1]:.6f},{beta[2]:.6f},{prob:.6f},{err:.6f}"

            lines.append(row)

        lines.append("")  # Empty line between ranges

    lines.append("=" * 200)
    lines.append("\nITERATION SUMMARY")
    lines.append("-" * 100)
    lines.append(f"{'Iteration':<12} {'β0 (bias)':<15} {'β1 (X0)':<15} {'β2 (X1)':<15} {'Avg |Error|':<12}")
    lines.append("-" * 100)

    for iter_data in sigmoid_gen.iterations_data:
        iteration = iter_data['iteration']
        beta = iter_data['beta']
        avg_error = np.mean(np.abs(iter_data['errors']))
        lines.append(f"{iteration:<12} {beta[0]:<15.6f} {beta[1]:<15.6f} {beta[2]:<15.6f} {avg_error:<12.6f}")

    lines.append("=" * 100)

    # Save to file
    with open(filename, 'w') as f:
        f.write("\n".join(lines))

    print(f"Sample range table saved to: {filename}")


def plot_sample_ranges(data, labels, sigmoid_gen, output_path):
    """
    Create visualization highlighting first 100 and last 50 samples.
    """
    X0 = data[:, 1]
    X1 = data[:, 2]

    # Get predictions
    predictions = sigmoid_gen.predict(data)

    # Define sample ranges
    first_100 = slice(0, 100)
    last_50 = slice(-50, None)
    middle = np.ones(len(data), dtype=bool)
    middle[:100] = False
    middle[-50:] = False

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot middle samples (lighter, smaller)
    mask_class0_middle = (labels == 0) & middle
    mask_class1_middle = (labels == 1) & middle

    ax.scatter(X0[mask_class0_middle], X1[mask_class0_middle],
              c='lightblue', marker='o', s=30, alpha=0.3,
              edgecolors='none', label='Class 0 - Other samples')
    ax.scatter(X0[mask_class1_middle], X1[mask_class1_middle],
              c='lightcoral', marker='s', s=30, alpha=0.3,
              edgecolors='none', label='Class 1 - Other samples')

    # Plot first 100 samples - BOLD
    mask_class0_first = (labels == 0) & np.arange(len(labels)) < 100
    mask_class1_first = (labels == 1) & np.arange(len(labels)) < 100
    correct_first = predictions == labels
    incorrect_first = predictions != labels

    ax.scatter(X0[mask_class0_first & correct_first], X1[mask_class0_first & correct_first],
              c='blue', marker='o', s=150, alpha=0.8,
              edgecolors='black', linewidth=2,
              label='First 100 - Class 0 Correct')
    ax.scatter(X0[mask_class1_first & correct_first], X1[mask_class1_first & correct_first],
              c='red', marker='s', s=150, alpha=0.8,
              edgecolors='black', linewidth=2,
              label='First 100 - Class 1 Correct')

    first_100_mask = np.arange(len(labels)) < 100
    ax.scatter(X0[first_100_mask & incorrect_first], X1[first_100_mask & incorrect_first],
              c='yellow', marker='X', s=250, alpha=0.95,
              edgecolors='black', linewidth=2,
              label='First 100 - Misclassified')

    # Plot last 50 samples - DIFFERENT SHAPE
    last_50_indices = np.arange(len(labels)) >= len(labels) - 50
    mask_class0_last = (labels == 0) & last_50_indices
    mask_class1_last = (labels == 1) & last_50_indices
    correct_last = predictions == labels
    incorrect_last = predictions != labels

    ax.scatter(X0[mask_class0_last & correct_last], X1[mask_class0_last & correct_last],
              c='darkblue', marker='D', s=150, alpha=0.9,
              edgecolors='cyan', linewidth=2,
              label='Last 50 - Class 0 Correct')
    ax.scatter(X0[mask_class1_last & correct_last], X1[mask_class1_last & correct_last],
              c='darkred', marker='D', s=150, alpha=0.9,
              edgecolors='orange', linewidth=2,
              label='Last 50 - Class 1 Correct')

    ax.scatter(X0[last_50_indices & incorrect_last], X1[last_50_indices & incorrect_last],
              c='magenta', marker='P', s=250, alpha=0.95,
              edgecolors='black', linewidth=2,
              label='Last 50 - Misclassified')

    # Decision boundary
    beta = sigmoid_gen.beta
    if beta[2] != 0:
        x0_line = np.linspace(0, 1, 100)
        x1_line = -(beta[0] + beta[1] * x0_line) / beta[2]
        ax.plot(x0_line, x1_line, 'g-', linewidth=3,
               label=f'Decision Boundary')

    # Formatting
    ax.set_xlabel('X0', fontsize=14, fontweight='bold')
    ax.set_ylabel('X1', fontsize=14, fontweight='bold')
    ax.set_title('Dataset: First 100 Samples (Bold) & Last 50 Samples (Diamonds)',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Statistics
    first_100_acc = np.mean(predictions[:100] == labels[:100]) * 100
    last_50_acc = np.mean(predictions[-50:] == labels[-50:]) * 100
    overall_acc = np.mean(predictions == labels) * 100

    ax.text(0.02, 0.98,
           f'Overall Accuracy: {overall_acc:.2f}%\n'
           f'First 100: {first_100_acc:.2f}%\n'
           f'Last 50: {last_50_acc:.2f}%',
           transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Sample ranges visualization saved to: {output_path}")
    print(f"  First 100 accuracy: {first_100_acc:.2f}%")
    print(f"  Last 50 accuracy: {last_50_acc:.2f}%")


def plot_separate_ranges(data, labels, sigmoid_gen, output_dir):
    """Create separate plots for first 100 and last 50."""
    predictions = sigmoid_gen.predict(data)
    X0 = data[:, 1]
    X1 = data[:, 2]

    # First 100 samples plot
    fig, ax = plt.subplots(figsize=(12, 10))

    first_100_data = data[:100]
    first_100_labels = labels[:100]
    first_100_preds = predictions[:100]
    X0_first = X0[:100]
    X1_first = X1[:100]

    correct = first_100_preds == first_100_labels
    incorrect = first_100_preds != first_100_labels

    # Class 0
    mask_0 = first_100_labels == 0
    ax.scatter(X0_first[mask_0 & correct], X1_first[mask_0 & correct],
              c='blue', marker='o', s=200, alpha=0.7,
              edgecolors='black', linewidth=2,
              label=f'Class 0 Correct ({np.sum(mask_0 & correct)})')

    # Class 1
    mask_1 = first_100_labels == 1
    ax.scatter(X0_first[mask_1 & correct], X1_first[mask_1 & correct],
              c='red', marker='s', s=200, alpha=0.7,
              edgecolors='black', linewidth=2,
              label=f'Class 1 Correct ({np.sum(mask_1 & correct)})')

    # Misclassified
    ax.scatter(X0_first[incorrect], X1_first[incorrect],
              c='yellow', marker='X', s=300, alpha=0.95,
              edgecolors='black', linewidth=2,
              label=f'Misclassified ({np.sum(incorrect)})')

    # Decision boundary
    beta = sigmoid_gen.beta
    if beta[2] != 0:
        x0_line = np.linspace(0, 1, 100)
        x1_line = -(beta[0] + beta[1] * x0_line) / beta[2]
        ax.plot(x0_line, x1_line, 'g-', linewidth=3, label='Decision Boundary')

    # Add sample indices as text
    for i in range(100):
        if i % 10 == 0:  # Show every 10th index
            ax.text(X0_first[i], X1_first[i], str(i), fontsize=8, alpha=0.6)

    accuracy = np.mean(correct) * 100
    ax.set_title(f'First 100 Samples - Accuracy: {accuracy:.2f}%',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('X0', fontsize=14, fontweight='bold')
    ax.set_ylabel('X1', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/first_100_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"First 100 samples plot saved")

    # Last 50 samples plot
    fig, ax = plt.subplots(figsize=(12, 10))

    last_50_data = data[-50:]
    last_50_labels = labels[-50:]
    last_50_preds = predictions[-50:]
    X0_last = X0[-50:]
    X1_last = X1[-50:]

    correct = last_50_preds == last_50_labels
    incorrect = last_50_preds != last_50_labels

    # Class 0
    mask_0 = last_50_labels == 0
    ax.scatter(X0_last[mask_0 & correct], X1_last[mask_0 & correct],
              c='darkblue', marker='D', s=200, alpha=0.8,
              edgecolors='cyan', linewidth=2,
              label=f'Class 0 Correct ({np.sum(mask_0 & correct)})')

    # Class 1
    mask_1 = last_50_labels == 1
    ax.scatter(X0_last[mask_1 & correct], X1_last[mask_1 & correct],
              c='darkred', marker='D', s=200, alpha=0.8,
              edgecolors='orange', linewidth=2,
              label=f'Class 1 Correct ({np.sum(mask_1 & correct)})')

    # Misclassified
    ax.scatter(X0_last[incorrect], X1_last[incorrect],
              c='magenta', marker='P', s=300, alpha=0.95,
              edgecolors='black', linewidth=2,
              label=f'Misclassified ({np.sum(incorrect)})')

    # Decision boundary
    if beta[2] != 0:
        x0_line = np.linspace(0, 1, 100)
        x1_line = -(beta[0] + beta[1] * x0_line) / beta[2]
        ax.plot(x0_line, x1_line, 'g-', linewidth=3, label='Decision Boundary')

    # Add sample indices
    start_idx = len(data) - 50
    for i in range(50):
        if i % 5 == 0:  # Show every 5th index
            ax.text(X0_last[i], X1_last[i], str(start_idx + i), fontsize=8, alpha=0.6)

    accuracy = np.mean(correct) * 100
    ax.set_title(f'Last 50 Samples (Indices {len(data)-50} to {len(data)-1}) - Accuracy: {accuracy:.2f}%',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('X0', fontsize=14, fontweight='bold')
    ax.set_ylabel('X1', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/last_50_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Last 50 samples plot saved")


def main():
    print("=" * 100)
    print("CREATING SAMPLE RANGE OUTPUTS (FIRST 100 & LAST 50)")
    print("=" * 100)
    print()

    # Create output directory
    os.makedirs("results/visualizations", exist_ok=True)

    # Generate dataset
    print("Generating 6000-sample dataset...")
    generator = DatasetGenerator(n_samples=6000)
    data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)
    print(f"Dataset: {len(data)} samples")
    print()

    # Train model
    print("Training sigmoid generator (20 iterations)...")
    sigmoid_gen = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=0.3)
    sigmoid_gen.train(data, labels, n_iterations=20)
    print("Training completed!")
    print()

    # Create range table
    print("Creating sample range table...")
    sample_ranges = [
        (0, 100, "FIRST_100"),
        (len(data) - 50, len(data), "LAST_50")
    ]
    create_range_table(data, labels, sigmoid_gen, sample_ranges,
                      "results/DataSet_First100_Last50.txt")
    print()

    # Create combined visualization
    print("Creating combined sample ranges visualization...")
    plot_sample_ranges(data, labels, sigmoid_gen,
                      "results/visualizations/sample_ranges_combined.png")
    print()

    # Create separate visualizations
    print("Creating separate range visualizations...")
    plot_separate_ranges(data, labels, sigmoid_gen, "results/visualizations")
    print()

    print("=" * 100)
    print("SAMPLE RANGE OUTPUTS COMPLETED!")
    print("=" * 100)
    print("Files created:")
    print("  - results/DataSet_First100_Last50.txt")
    print("  - results/visualizations/sample_ranges_combined.png")
    print("  - results/visualizations/first_100_samples.png")
    print("  - results/visualizations/last_50_samples.png")
    print()


if __name__ == "__main__":
    main()
