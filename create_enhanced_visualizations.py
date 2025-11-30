"""
Create enhanced visualizations showing:
1. Original dataset with trained model predictions
2. Decision boundary
3. Correctly vs incorrectly classified samples
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator
import os


def plot_dataset_with_predictions(data, labels, sigmoid_gen, output_path):
    """
    Plot dataset with model predictions using different colors/shapes.
    """
    X0 = data[:, 1]
    X1 = data[:, 2]

    # Get predictions from trained model
    predictions = sigmoid_gen.predict(data)
    probabilities = sigmoid_gen.predict_probability(data)

    # Classify samples
    correct_class0 = (labels == 0) & (predictions == 0)
    correct_class1 = (labels == 1) & (predictions == 1)
    incorrect_class0 = (labels == 0) & (predictions == 1)  # False positive
    incorrect_class1 = (labels == 1) & (predictions == 0)  # False negative

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot correctly classified class 0 (blue circles)
    ax.scatter(X0[correct_class0], X1[correct_class0],
              c='blue', marker='o', s=100, alpha=0.6,
              edgecolors='black', linewidth=1,
              label=f'Class 0 - Correct ({np.sum(correct_class0)})')

    # Plot correctly classified class 1 (red squares)
    ax.scatter(X0[correct_class1], X1[correct_class1],
              c='red', marker='s', s=100, alpha=0.6,
              edgecolors='black', linewidth=1,
              label=f'Class 1 - Correct ({np.sum(correct_class1)})')

    # Plot incorrectly classified class 0 (blue X)
    if np.sum(incorrect_class0) > 0:
        ax.scatter(X0[incorrect_class0], X1[incorrect_class0],
                  c='cyan', marker='X', s=200, alpha=0.9,
                  edgecolors='blue', linewidth=2,
                  label=f'Class 0 - Misclassified ({np.sum(incorrect_class0)})')

    # Plot incorrectly classified class 1 (red X)
    if np.sum(incorrect_class1) > 0:
        ax.scatter(X0[incorrect_class1], X1[incorrect_class1],
                  c='orange', marker='X', s=200, alpha=0.9,
                  edgecolors='red', linewidth=2,
                  label=f'Class 1 - Misclassified ({np.sum(incorrect_class1)})')

    # Add decision boundary
    # Decision boundary: σ(z) = 0.5 => z = 0
    # z = β0 + β1*X0 + β2*X1 = 0
    # X1 = -(β0 + β1*X0) / β2
    beta = sigmoid_gen.beta
    if beta[2] != 0:
        x0_line = np.linspace(0, 1, 100)
        x1_line = -(beta[0] + beta[1] * x0_line) / beta[2]
        ax.plot(x0_line, x1_line, 'g-', linewidth=3,
               label=f'Decision Boundary\n(β={beta[0]:.2f}, {beta[1]:.2f}, {beta[2]:.2f})')

    # Add probability contours
    x0_mesh = np.linspace(-0.05, 1.05, 200)
    x1_mesh = np.linspace(-0.05, 1.05, 200)
    X0_mesh, X1_mesh = np.meshgrid(x0_mesh, x1_mesh)

    # Create data points for mesh
    mesh_data = np.column_stack([
        np.ones(X0_mesh.ravel().shape),
        X0_mesh.ravel(),
        X1_mesh.ravel()
    ])

    # Calculate probabilities
    Z = sigmoid_gen.predict_probability(mesh_data).reshape(X0_mesh.shape)

    # Add contours
    contour = ax.contour(X0_mesh, X1_mesh, Z, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                        colors='gray', alpha=0.3, linestyles='dashed', linewidths=1)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

    # Formatting
    ax.set_xlabel('X0', fontsize=14, fontweight='bold')
    ax.set_ylabel('X1', fontsize=14, fontweight='bold')
    ax.set_title('Dataset with Trained Model Predictions & Decision Boundary',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Add accuracy text
    accuracy = np.mean(predictions == labels) * 100
    ax.text(0.02, 0.98,
           f'Accuracy: {accuracy:.2f}%\n'
           f'Correct: {np.sum(predictions == labels)}\n'
           f'Incorrect: {np.sum(predictions != labels)}\n'
           f'Total: {len(labels)}',
           transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Enhanced visualization saved to: {output_path}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct predictions: {np.sum(predictions == labels)}/{len(labels)}")


def plot_prediction_evolution(data, labels, output_path):
    """
    Show how predictions evolve across iterations.
    """
    # Train model and track predictions at different iterations
    iterations_to_track = [0, 5, 10, 15, 20]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    X0 = data[:, 1]
    X1 = data[:, 2]

    for idx, n_iter in enumerate(iterations_to_track):
        ax = axes[idx]

        # Train model up to this iteration
        sigmoid_gen = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=0.3)
        if n_iter > 0:
            sigmoid_gen.train(data, labels, n_iterations=n_iter)

        # Get predictions
        predictions = sigmoid_gen.predict(data)

        # Plot
        mask_class0 = labels == 0
        mask_class1 = labels == 1

        # Correctly classified
        correct = predictions == labels
        incorrect = predictions != labels

        ax.scatter(X0[mask_class0 & correct], X1[mask_class0 & correct],
                  c='blue', marker='o', s=50, alpha=0.5, edgecolors='black', linewidth=0.5)
        ax.scatter(X0[mask_class1 & correct], X1[mask_class1 & correct],
                  c='red', marker='s', s=50, alpha=0.5, edgecolors='black', linewidth=0.5)

        # Incorrectly classified
        ax.scatter(X0[incorrect], X1[incorrect],
                  c='yellow', marker='X', s=100, alpha=0.9, edgecolors='black', linewidth=1)

        # Decision boundary
        beta = sigmoid_gen.beta
        if beta[2] != 0:
            x0_line = np.linspace(0, 1, 100)
            x1_line = -(beta[0] + beta[1] * x0_line) / beta[2]
            ax.plot(x0_line, x1_line, 'g-', linewidth=2)

        accuracy = np.mean(predictions == labels) * 100
        ax.set_title(f'Iteration {n_iter}\nAccuracy: {accuracy:.1f}%',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Remove extra subplot
    axes[-1].axis('off')

    plt.suptitle('Model Evolution Across Iterations', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Prediction evolution saved to: {output_path}")


def main():
    print("=" * 100)
    print("CREATING ENHANCED VISUALIZATIONS WITH MODEL PREDICTIONS")
    print("=" * 100)
    print()

    # Create output directory
    os.makedirs("results/visualizations", exist_ok=True)

    # Generate and train on 6000 samples
    print("Generating 6000-sample dataset...")
    generator = DatasetGenerator(n_samples=6000)
    data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)
    print(f"Dataset: {len(data)} samples")
    print()

    print("Training sigmoid generator (20 iterations)...")
    sigmoid_gen = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=0.3)
    sigmoid_gen.train(data, labels, n_iterations=20)
    print("Training completed!")
    print()

    # Create enhanced visualization
    print("Creating dataset with predictions visualization...")
    plot_dataset_with_predictions(
        data, labels, sigmoid_gen,
        "results/visualizations/dataset_with_predictions.png"
    )
    print()

    # Create prediction evolution visualization
    print("Creating prediction evolution visualization...")
    plot_prediction_evolution(
        data, labels,
        "results/visualizations/prediction_evolution.png"
    )
    print()

    print("=" * 100)
    print("VISUALIZATIONS COMPLETED!")
    print("=" * 100)
    print("Files created:")
    print("  - results/visualizations/dataset_with_predictions.png")
    print("  - results/visualizations/prediction_evolution.png")
    print()


if __name__ == "__main__":
    main()
