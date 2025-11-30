import matplotlib.pyplot as plt
import numpy as np
import os


class DataVisualizer:
    """
    Visualizes binary classification dataset.
    """

    def __init__(self, output_dir="visualizations"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_dataset(self, data, labels, title="Dataset Visualization",
                     filename="dataset_plot.png"):
        """
        Create scatter plot of the dataset with different colors per class.

        Args:
            data: Array of shape (n_samples, 3) with columns [bias, X0, X1]
            labels: Array of shape (n_samples,) with values 0 or 1
            title: Plot title
            filename: Output filename
        """
        # Extract X0 and X1 (skip bias column)
        X0 = data[:, 1]
        X1 = data[:, 2]

        # Separate by class
        mask_class0 = labels == 0
        mask_class1 = labels == 1

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot class 0 in blue
        plt.scatter(X0[mask_class0], X1[mask_class0],
                   c='blue', marker='o', s=100, alpha=0.6,
                   edgecolors='black', linewidth=1,
                   label='Class 0 (y=0)')

        # Plot class 1 in red
        plt.scatter(X0[mask_class1], X1[mask_class1],
                   c='red', marker='s', s=100, alpha=0.6,
                   edgecolors='black', linewidth=1,
                   label='Class 1 (y=1)')

        # Formatting
        plt.xlabel('X0', fontsize=14, fontweight='bold')
        plt.ylabel('X1', fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)

        # Add sample counts to legend
        n_class0 = np.sum(mask_class0)
        n_class1 = np.sum(mask_class1)
        plt.text(0.02, 0.98, f'Total samples: {len(labels)}\n'
                             f'Class 0: {n_class0}\nClass 1: {n_class1}',
                transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save plot
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {filepath}")

        return filepath
