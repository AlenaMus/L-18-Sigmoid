"""
Create enhanced dataset table with iteration data as columns.
Each row shows: Idx, Bias, X0, X1, y, and for each iteration: β values, σ(z), error
"""

import numpy as np
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator


def create_enhanced_dataset_table(data, labels, sigmoid_gen, max_samples=100):
    """Create dataset table with iteration columns."""
    lines = []

    # Header
    lines.append("=" * 250)
    lines.append("ENHANCED DATASET TABLE WITH ITERATION RESULTS")
    lines.append("=" * 250)
    lines.append(f"Total Samples: {len(data)}")
    lines.append(f"Initial β0: [0, 0.3, -0.5], Learning Rate η: {sigmoid_gen.eta}")
    lines.append(f"Showing first {min(max_samples, len(data))} samples")
    lines.append("=" * 250)
    lines.append("")

    # Build column headers
    header_line1 = f"{'Idx':<5} {'Bias':<5} {'X0':<10} {'X1':<10} {'y':<3}"
    header_line2 = f"{'':5} {'':5} {'':10} {'':10} {'':3}"

    # Select iterations to display (e.g., every 5 iterations)
    selected_iterations = [0, 5, 10, 15, 20]

    for iter_num in selected_iterations:
        if iter_num < len(sigmoid_gen.iterations_data):
            iter_data = sigmoid_gen.iterations_data[iter_num]
            beta = iter_data['beta']
            header_line1 += f" | Iteration {iter_num:2d}"
            header_line1 += " " * 35
            header_line2 += f" | β=[{beta[0]:6.3f},{beta[1]:6.3f},{beta[2]:6.3f}]"
            header_line2 += f" | σ(z)      | Error     "

    lines.append(header_line1)
    lines.append(header_line2)
    lines.append("-" * 250)

    # Data rows
    n_samples = min(max_samples, len(data))
    for i in range(n_samples):
        row = f"{i:<5} {data[i,0]:<5.1f} {data[i,1]:<10.6f} {data[i,2]:<10.6f} {int(labels[i]):<3}"

        for iter_num in selected_iterations:
            if iter_num < len(sigmoid_gen.iterations_data):
                iter_data = sigmoid_gen.iterations_data[iter_num]
                prob = iter_data['probabilities'][i]
                err = iter_data['errors'][i]
                beta = iter_data['beta']

                # Add empty space for beta values (shown in header)
                row += f" | {' '*30}"
                row += f" | {prob:9.6f} | {err:9.6f}"

        lines.append(row)

    if len(data) > max_samples:
        lines.append(f"... ({len(data) - max_samples} more samples)")

    lines.append("=" * 250)
    lines.append("")

    # Add summary statistics
    lines.append("ITERATION SUMMARY:")
    lines.append("-" * 100)
    lines.append(f"{'Iteration':<12} {'β0 (bias)':<15} {'β1 (X0)':<15} {'β2 (X1)':<15} {'Avg |Error|':<12}")
    lines.append("-" * 100)

    for iter_data in sigmoid_gen.iterations_data:
        iteration = iter_data['iteration']
        beta = iter_data['beta']
        avg_error = np.mean(np.abs(iter_data['errors']))
        lines.append(f"{iteration:<12} {beta[0]:<15.6f} {beta[1]:<15.6f} {beta[2]:<15.6f} {avg_error:<12.6f}")

    lines.append("=" * 100)

    return "\n".join(lines)


def create_compact_csv_style_table(data, labels, sigmoid_gen, max_samples=100):
    """Create more compact CSV-style table."""
    lines = []

    lines.append("COMPACT DATASET TABLE - CSV FORMAT")
    lines.append("=" * 200)
    lines.append(f"Total Samples: {len(data)}, η: {sigmoid_gen.eta}, Initial β0: [0, 0.3, -0.5]")
    lines.append("=" * 200)
    lines.append("")

    # CSV Header
    header = "Idx,Bias,X0,X1,y"
    selected_iterations = [0, 5, 10, 15, 20]

    for iter_num in selected_iterations:
        if iter_num < len(sigmoid_gen.iterations_data):
            header += f",It{iter_num}_β0,It{iter_num}_β1,It{iter_num}_β2,It{iter_num}_σ(z),It{iter_num}_Error"

    lines.append(header)

    # CSV Data
    n_samples = min(max_samples, len(data))
    for i in range(n_samples):
        row = f"{i},{data[i,0]:.1f},{data[i,1]:.6f},{data[i,2]:.6f},{int(labels[i])}"

        for iter_num in selected_iterations:
            if iter_num < len(sigmoid_gen.iterations_data):
                iter_data = sigmoid_gen.iterations_data[iter_num]
                beta = iter_data['beta']
                prob = iter_data['probabilities'][i]
                err = iter_data['errors'][i]

                row += f",{beta[0]:.6f},{beta[1]:.6f},{beta[2]:.6f},{prob:.6f},{err:.6f}"

        lines.append(row)

    if len(data) > max_samples:
        lines.append(f"# ... ({len(data) - max_samples} more samples)")

    lines.append("")
    lines.append("=" * 200)

    return "\n".join(lines)


def main():
    print("=" * 100)
    print("CREATING ENHANCED DATASET TABLE WITH ITERATION COLUMNS")
    print("=" * 100)
    print()

    # Generate dataset
    print("Generating dataset...")
    generator = DatasetGenerator(n_samples=6000)
    data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)
    print(f"Dataset: {len(data)} samples")
    print()

    # Train sigmoid generator
    print("Training sigmoid generator...")
    sigmoid_gen = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=0.3)
    sigmoid_gen.train(data, labels, n_iterations=20)
    print("Training completed!")
    print()

    # Create enhanced table
    print("Creating enhanced table format...")
    enhanced_table = create_enhanced_dataset_table(data, labels, sigmoid_gen, max_samples=50)

    with open("DataSet_Enhanced.txt", 'w') as f:
        f.write(enhanced_table)

    print("Saved: DataSet_Enhanced.txt")
    print()

    # Create compact CSV-style table
    print("Creating compact CSV-style table...")
    csv_table = create_compact_csv_style_table(data, labels, sigmoid_gen, max_samples=100)

    with open("DataSet_Compact_CSV.txt", 'w') as f:
        f.write(csv_table)

    print("Saved: DataSet_Compact_CSV.txt")
    print()

    # Display preview
    print("Preview of enhanced table:")
    print("-" * 100)
    print(enhanced_table[:2000])
    print("...")
    print()

    print("=" * 100)
    print("COMPLETED!")
    print("=" * 100)
    print("Files created:")
    print("  - DataSet_Enhanced.txt (formatted table with iteration columns)")
    print("  - DataSet_Compact_CSV.txt (CSV format with all iteration data)")
    print()


if __name__ == "__main__":
    main()
