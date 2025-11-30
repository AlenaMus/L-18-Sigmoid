import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()


class DatasetGenerator:
    """
    Generates binary classification dataset with normal distribution.
    Each sample is a 3D vector: (1, X0, X1) where 1 is bias term.
    """

    def __init__(self, n_samples=100, random_seed=None):
        """
        Initialize dataset generator.

        Args:
            n_samples: Total number of samples (split equally between classes)
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_seed = random_seed or int(os.getenv('RANDOM_SEED', 42))
        np.random.seed(self.random_seed)

        self.data = None
        self.labels = None

    def generate(self, mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1):
        """
        Generate two separated groups using normal distribution.

        Args:
            mean0: Mean for group 0 (X0, X1)
            mean1: Mean for group 1 (X0, X1)
            std: Standard deviation for both groups

        Returns:
            data: Array of shape (n_samples, 3) with columns [bias, X0, X1]
            labels: Array of shape (n_samples,) with values 0 or 1
        """
        samples_per_class = self.n_samples // 2

        # Generate group 0 (y=0)
        X0_group0 = np.random.normal(mean0[0], std, samples_per_class)
        X1_group0 = np.random.normal(mean0[1], std, samples_per_class)

        # Clip to [0, 1] range
        X0_group0 = np.clip(X0_group0, 0, 1)
        X1_group0 = np.clip(X1_group0, 0, 1)

        # Generate group 1 (y=1)
        X0_group1 = np.random.normal(mean1[0], std, samples_per_class)
        X1_group1 = np.random.normal(mean1[1], std, samples_per_class)

        # Clip to [0, 1] range
        X0_group1 = np.clip(X0_group1, 0, 1)
        X1_group1 = np.clip(X1_group1, 0, 1)

        # Combine groups
        X0 = np.concatenate([X0_group0, X0_group1])
        X1 = np.concatenate([X1_group0, X1_group1])

        # Add bias term (column of ones)
        bias = np.ones(self.n_samples)

        # Create data array: [bias, X0, X1]
        self.data = np.column_stack([bias, X0, X1])

        # Create labels
        self.labels = np.concatenate([
            np.zeros(samples_per_class),
            np.ones(samples_per_class)
        ])

        return self.data, self.labels

    def get_table_string(self):
        """
        Generate formatted table string for the dataset.

        Returns:
            table_str: Formatted string representation of the dataset
        """
        if self.data is None or self.labels is None:
            raise ValueError("Dataset not generated yet. Call generate() first.")

        lines = []
        lines.append("=" * 60)
        lines.append("DATASET TABLE")
        lines.append("=" * 60)
        lines.append(f"{'Index':<8} {'Bias':<10} {'X0':<12} {'X1':<12} {'y':<5}")
        lines.append("-" * 60)

        for i in range(len(self.data)):
            bias, x0, x1 = self.data[i]
            y = int(self.labels[i])
            lines.append(f"{i:<8} {bias:<10.1f} {x0:<12.6f} {x1:<12.6f} {y:<5}")

        lines.append("=" * 60)
        lines.append(f"Total samples: {len(self.data)}")
        lines.append(f"Class 0 samples: {np.sum(self.labels == 0)}")
        lines.append(f"Class 1 samples: {np.sum(self.labels == 1)}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_table(self, filepath="DataSet.txt"):
        """
        Save dataset table to text file.

        Args:
            filepath: Path to save the table
        """
        table_str = self.get_table_string()
        with open(filepath, 'w') as f:
            f.write(table_str)
        print(f"Dataset table saved to: {filepath}")

    def get_data(self):
        """Return data and labels."""
        return self.data, self.labels
