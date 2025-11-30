"""
Create detailed dataset table with iteration data.
Shows: Idx, Bias, X0, X1, y, and for each iteration: β values, σ(z), error
"""

import numpy as np
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator


def create_compact_table(sigmoid_gen, data, labels, max_rows=50):
    """Create table with selected iterations to keep it readable."""
    lines = []
    lines.append("=" * 180)
    lines.append("DATASET WITH ITERATION RESULTS - 6000 SAMPLES")
    lines.append("=" * 180)
    lines.append(f"Total Samples: {len(data)}, Learning Rate η: {sigmoid_gen.eta}")
    lines.append("")
    lines.append("Table shows first {0} samples with selected iterations (0, 5, 10, 15, 20)".format(max_rows))
    lines.append("For each iteration: Beta weights | Sigmoid σ(z) | Error")
    lines.append("=" * 180)
    lines.append("")

    # Base data header
    lines.append(f"{'Idx':<5} {'Bias':<5} {'X0':<10} {'X1':<10} {'y':<3}")
    lines.append("-" * 40)

    # Show first max_rows samples
    for i in range(min(max_rows, len(data))):
        line = f"{i:<5} {data[i,0]:<5.1f} {data[i,1]:<10.6f} {data[i,2]:<10.6f} {int(labels[i]):<3}"
        lines.append(line)

    lines.append("")
    lines.append("=" * 180)
    lines.append("ITERATION DETAILS FOR SELECTED ITERATIONS")
    lines.append("=" * 180)

    # Show detailed iterations (0, 5, 10, 15, 20)
    selected_iterations = [0, 5, 10, 15, 20]

    for iter_idx in selected_iterations:
        if iter_idx < len(sigmoid_gen.iterations_data):
            iter_data = sigmoid_gen.iterations_data[iter_idx]
            iteration = iter_data['iteration']
            beta = iter_data['beta']
            probs = iter_data['probabilities']
            errors = iter_data['errors']

            lines.append("")
            lines.append(f"ITERATION {iteration}:")
            lines.append(f"  Beta weights: β0={beta[0]:.6f}, β1={beta[1]:.6f}, β2={beta[2]:.6f}")
            lines.append(f"  {'Idx':<6} {'y':<4} {'σ(z)':<12} {'Error':<12} | Sample details")
            lines.append("  " + "-" * 80)

            # Show first 30 samples for this iteration
            for i in range(min(30, len(probs))):
                sample_line = f"  {i:<6} {int(labels[i]):<4} {probs[i]:<12.6f} {errors[i]:<12.6f}"
                sample_line += f" | X0={data[i,1]:.4f}, X1={data[i,2]:.4f}"
                lines.append(sample_line)

            if len(probs) > 30:
                lines.append(f"  ... ({len(probs) - 30} more samples)")

            avg_abs_error = np.mean(np.abs(errors))
            lines.append(f"  Average Absolute Error: {avg_abs_error:.6f}")

    lines.append("")
    lines.append("=" * 180)
    lines.append("END OF TABLE")
    lines.append("=" * 180)

    return "\n".join(lines)


def main():
    print("Creating detailed dataset table...")
    print()

    # Generate dataset
    generator = DatasetGenerator(n_samples=6000)
    data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)

    # Train sigmoid generator
    sigmoid_gen = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=0.3)
    sigmoid_gen.train(data, labels, n_iterations=20)

    # Create table
    table = create_compact_table(sigmoid_gen, data, labels, max_rows=50)

    # Save to file
    with open("DataSet_6000_WithIterations.txt", 'w') as f:
        f.write(table)

    print("Detailed dataset table saved to: DataSet_6000_WithIterations.txt")
    print()
    print("Table preview:")
    print("-" * 100)
    print(table[:3000])
    print("...")
    print()


if __name__ == "__main__":
    main()
