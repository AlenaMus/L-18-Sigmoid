"""
Step 1: Dataset Generation and Visualization

This script generates a binary classification dataset with two separated groups
and visualizes the data with tables and graphs.
"""

from dataset_generator import DatasetGenerator
from visualizer import DataVisualizer


def main():
    print("=" * 60)
    print("STEP 1: DATASET GENERATION AND VISUALIZATION")
    print("=" * 60)
    print()

    # Parameters
    n_samples = 100
    mean_group0 = (0.3, 0.3)  # Mean for class 0
    mean_group1 = (0.7, 0.7)  # Mean for class 1
    std = 0.1  # Standard deviation

    # Generate dataset
    print("Generating dataset...")
    generator = DatasetGenerator(n_samples=n_samples)
    data, labels = generator.generate(mean0=mean_group0, mean1=mean_group1, std=std)

    print(f"Generated {n_samples} samples:")
    print(f"  - Class 0: {(labels == 0).sum()} samples")
    print(f"  - Class 1: {(labels == 1).sum()} samples")
    print(f"  - Feature range: [0, 1]")
    print()

    # Save table
    print("Saving dataset table...")
    generator.save_table("DataSet.txt")
    print()

    # Visualize dataset
    print("Creating visualization...")
    visualizer = DataVisualizer(output_dir="visualizations")
    visualizer.plot_dataset(data, labels,
                           title="Binary Classification Dataset",
                           filename="dataset_visualization.png")
    print()

    print("=" * 60)
    print("STEP 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Generated files:")
    print("  - DataSet.txt (dataset table)")
    print("  - visualizations/dataset_visualization.png (graph)")
    print()


if __name__ == "__main__":
    main()
