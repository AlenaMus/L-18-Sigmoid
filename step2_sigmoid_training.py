"""
Step 2: Sigmoid Generator Training with Detailed Tracking

Shows step-by-step iteration process:
1. Compute probabilities using sigmoid σ(z) = 1/(1+e^(-z))
2. Calculate errors between y_true and σ(z)
3. Compute gradient using mathematics
4. Update beta weights using gradient descent
"""

import numpy as np
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator


def main():
    print("=" * 100)
    print("STEP 2: SIGMOID GENERATOR WITH DETAILED ITERATION TRACKING")
    print("=" * 100)
    print()

    # Load dataset
    print("Loading dataset...")
    generator = DatasetGenerator(n_samples=100)
    data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)
    print(f"Dataset loaded: {len(data)} samples")
    print()

    # Initialize sigmoid generator
    print("Initializing Sigmoid Generator...")
    beta0 = [0, 0.3, -0.5]
    eta = 0.3

    sigmoid_gen = SigmoidGenerator(beta0=beta0, eta=eta)
    print(f"Initial weights β0: {beta0}")
    print(f"Learning rate η: {eta}")
    print()

    print("Training Process:")
    print("  Step 1: Compute probability σ(z) = 1/(1+e^(-z)) where z = β0 + β1*X0 + β2*X1")
    print("  Step 2: Calculate error = y_true - σ(z)")
    print("  Step 3: Compute gradient ∇J = (1/n) * X^T * (σ(z) - y)")
    print("  Step 4: Update weights β_new = β_old - η * ∇J")
    print()

    # Train the model
    n_iterations = 10

    print(f"Training for {n_iterations} iterations...")
    print()
    sigmoid_gen.train(data, labels, n_iterations=n_iterations)
    print("Training completed!")
    print()

    # Save detailed table
    print("Generating detailed iteration table...")
    detailed_table = sigmoid_gen.get_detailed_table(max_samples=5)

    with open("DetailedIterationTable.txt", 'w') as f:
        f.write(detailed_table)

    print("Detailed table saved to: DetailedIterationTable.txt")
    print()

    # Save summary table
    print("Generating summary table...")
    summary_table = sigmoid_gen.get_summary_table()

    with open("BetaProgressionSummary.txt", 'w') as f:
        f.write(summary_table)

    print("Summary table saved to: BetaProgressionSummary.txt")
    print()

    # Display summary
    print(summary_table)
    print()

    # Display sample from detailed table
    print("Sample from detailed iteration table (first 5 samples per iteration):")
    print("-" * 100)
    print(detailed_table[:2000])
    print("...")
    print()

    print("=" * 100)
    print("STEP 2 COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print("Files created:")
    print("  - DetailedIterationTable.txt (full iteration details with σ(z) and errors)")
    print("  - BetaProgressionSummary.txt (beta weights summary)")


if __name__ == "__main__":
    main()
