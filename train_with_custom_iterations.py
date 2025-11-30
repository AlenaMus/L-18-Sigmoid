"""
Simple script to train the model with custom number of iterations.
Easy to modify and experiment with different settings.
"""

import numpy as np
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator
import os


def train_and_evaluate(n_iterations=100, learning_rate=0.3):
    """
    Train model with specified iterations and learning rate.

    Args:
        n_iterations: Number of training iterations (default: 100)
        learning_rate: Learning rate η (default: 0.3)
    """
    print("=" * 80)
    print(f"TRAINING WITH {n_iterations} ITERATIONS (η={learning_rate})")
    print("=" * 80)
    print()

    # Generate dataset
    print("Generating dataset...")
    generator = DatasetGenerator(n_samples=6000)
    data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)
    print(f"Dataset: {len(data)} samples")
    print()

    # Train model
    print(f"Training for {n_iterations} iterations with η={learning_rate}...")
    model = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=learning_rate)
    model.train(data, labels, n_iterations=n_iterations)
    print("Training completed!")
    print()

    # Evaluate
    predictions = model.predict(data)
    accuracy = np.mean(predictions == labels) * 100
    correct = np.sum(predictions == labels)
    total = len(labels)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Iterations:    {n_iterations}")
    print(f"Learning rate: {learning_rate}")
    print(f"Accuracy:      {accuracy:.2f}%")
    print(f"Correct:       {correct}/{total}")
    print(f"Misclassified: {total - correct}")
    print(f"Final weights: β = {model.beta}")
    print("=" * 80)
    print()

    return accuracy, model


def main():
    # ===========================================
    # EASY CONFIGURATION - CHANGE THESE VALUES
    # ===========================================

    # Option 1: Try different iteration counts
    n_iterations = 500  # 👈 CHANGE THIS NUMBER!

    # Option 2: Try different learning rates
    learning_rate = 1.0  # 👈 CHANGE THIS NUMBER!

    # ===========================================

    # Train the model
    accuracy, model = train_and_evaluate(
        n_iterations=n_iterations,
        learning_rate=learning_rate
    )

    # Save results
    os.makedirs("results", exist_ok=True)

    result_file = f"results/Training_Iter{n_iterations}_LR{learning_rate}.txt"
    with open(result_file, 'w') as f:
        f.write(f"Training Results\n")
        f.write(f"================\n\n")
        f.write(f"Iterations: {n_iterations}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Final Weights: {model.beta}\n")

    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    # You can also run with different values directly:

    # Example 1: 100 iterations (fast)
    # train_and_evaluate(n_iterations=100, learning_rate=0.3)

    # Example 2: 500 iterations (best balance)
    # train_and_evaluate(n_iterations=500, learning_rate=1.0)

    # Example 3: 1000 iterations (maximum accuracy)
    # train_and_evaluate(n_iterations=1000, learning_rate=1.0)

    main()
