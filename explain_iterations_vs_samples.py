"""
Demonstration: Samples vs Iterations
Shows the difference between Batch, Stochastic, and Mini-Batch gradient descent
"""

import numpy as np
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator


def demonstrate_batch_vs_stochastic():
    """Show how many samples are processed in each method."""

    print("=" * 80)
    print("UNDERSTANDING: SAMPLES vs ITERATIONS")
    print("=" * 80)
    print()

    # Generate small dataset for demo
    generator = DatasetGenerator(n_samples=6000)
    data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)

    print(f"Dataset size: {len(data)} samples")
    print()

    # Current method: Batch Gradient Descent
    print("=" * 80)
    print("METHOD 1: BATCH GRADIENT DESCENT (What we're using)")
    print("=" * 80)
    print()
    print("Each iteration:")
    print("  1. Process ALL 6000 samples")
    print("  2. Compute average gradient from all samples")
    print("  3. Update weights ONCE")
    print()

    iterations = 100
    print(f"With {iterations} iterations:")
    print(f"  - Weight updates: {iterations}")
    print(f"  - Samples processed per iteration: {len(data)}")
    print(f"  - Total sample views: {iterations * len(data):,}")
    print(f"  - Unique samples seen: {len(data)} (same samples repeated)")
    print()

    model_batch = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=0.3)
    model_batch.train(data, labels, n_iterations=iterations)
    accuracy_batch = np.mean(model_batch.predict(data) == labels) * 100
    print(f"Result: {accuracy_batch:.2f}% accuracy in {iterations} iterations")
    print()

    # Alternative: Stochastic Gradient Descent
    print("=" * 80)
    print("METHOD 2: STOCHASTIC GRADIENT DESCENT (Alternative)")
    print("=" * 80)
    print()
    print("Each iteration:")
    print("  1. Process ONE sample")
    print("  2. Compute gradient from that single sample")
    print("  3. Update weights ONCE")
    print()

    # For 1 epoch (see each sample once)
    iterations_sgd = len(data)
    print(f"For 1 epoch (see each sample once):")
    print(f"  - Weight updates: {iterations_sgd:,}")
    print(f"  - Samples processed per iteration: 1")
    print(f"  - Total iterations needed: {iterations_sgd:,}")
    print()

    # For multiple epochs
    epochs = 10
    iterations_sgd_multi = len(data) * epochs
    print(f"For {epochs} epochs (typical):")
    print(f"  - Weight updates: {iterations_sgd_multi:,}")
    print(f"  - Total iterations needed: {iterations_sgd_multi:,}")
    print()

    # Mini-batch
    print("=" * 80)
    print("METHOD 3: MINI-BATCH GRADIENT DESCENT (Middle Ground)")
    print("=" * 80)
    print()
    print("Each iteration:")
    print("  1. Process BATCH of samples (e.g., 32)")
    print("  2. Compute gradient from batch")
    print("  3. Update weights ONCE")
    print()

    batch_size = 32
    iterations_per_epoch = len(data) // batch_size
    print(f"With batch size {batch_size}:")
    print(f"  - Iterations per epoch: {iterations_per_epoch}")
    print(f"  - Weight updates per epoch: {iterations_per_epoch}")
    print()
    print(f"For {epochs} epochs:")
    print(f"  - Total iterations: {iterations_per_epoch * epochs:,}")
    print(f"  - Total weight updates: {iterations_per_epoch * epochs:,}")
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Method':<20} {'Iterations':<15} {'Updates':<15} {'Time/Iteration'}")
    print("-" * 80)
    print(f"{'Batch GD':<20} {100:<15} {100:<15} {'Slow (all data)'}")
    print(f"{'Stochastic GD':<20} {60000:<15} {60000:<15} {'Fast (1 sample)'}")
    print(f"{'Mini-Batch GD':<20} {iterations_per_epoch * 10:<15} {iterations_per_epoch * 10:<15} {'Medium (32 samples)'}")
    print()

    # Why batch works well
    print("=" * 80)
    print("WHY BATCH GRADIENT DESCENT NEEDS FEWER ITERATIONS:")
    print("=" * 80)
    print()
    print("✓ Each iteration uses ALL information from the dataset")
    print("✓ Gradient is accurate (averaged over 6000 samples)")
    print("✓ Each step is in the right direction")
    print("✓ Converges reliably in 100-500 iterations")
    print()
    print("Stochastic GD needs more iterations because:")
    print("✗ Each iteration uses only 1 sample (noisy gradient)")
    print("✗ Steps are less accurate (lots of zigzagging)")
    print("✗ Needs many passes through data to converge")
    print()

    # Answer the question
    print("=" * 80)
    print("ANSWER TO YOUR QUESTION:")
    print("=" * 80)
    print()
    print("Q: With 6000 samples, should we run 6000 iterations?")
    print()
    print("A: NO! Because:")
    print()
    print("   Our method (Batch GD) uses ALL 6000 samples in EACH iteration.")
    print()
    print("   - 1 iteration = process 6000 samples + 1 weight update")
    print("   - 100 iterations = process 6000 samples 100 times + 100 weight updates")
    print()
    print("   We achieve 98-99% accuracy in just 100 iterations because each")
    print("   iteration is very informative (uses all data).")
    print()
    print("   If we used Stochastic GD (1 sample per iteration), then YES,")
    print("   we'd need ~6000 iterations per epoch × 10 epochs = 60,000 iterations!")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_batch_vs_stochastic()
