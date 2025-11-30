import numpy as np


class SigmoidGenerator:
    """Sigmoid-based binary classifier with detailed iteration tracking."""

    def __init__(self, beta0=None, eta=0.3):
        """Initialize with β0=[0,0.3,-0.5] and η=0.3."""
        if beta0 is None:
            beta0 = [0, 0.3, -0.5]
        self.beta = np.array(beta0, dtype=float)
        self.eta = eta

        # Detailed tracking
        self.iterations_data = []

    def sigmoid_math(self, z):
        """
        Pure mathematical sigmoid: σ(z) = 1 / (1 + e^(-z))
        Using only numpy exp and basic arithmetic.
        """
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1.0 / (1.0 + np.exp(-z))

    def compute_probabilities(self, X, beta):
        """
        Step 1: Find probability using sigmoid mathematical function.

        For each sample: z = β0*1 + β1*X0 + β2*X1
        Then: σ(z) = 1 / (1 + e^(-z))
        """
        # X shape: (n_samples, 3) - columns are [1, X0, X1]
        # beta shape: (3,)
        # z = X @ beta
        z = np.dot(X, beta)
        probabilities = self.sigmoid_math(z)
        return probabilities

    def compute_errors(self, probabilities, y):
        """Calculate error: error = y - σ(z)"""
        return y - probabilities

    def compute_gradient_math(self, X, probabilities, y):
        """
        Step 2: Gradient descent using pure mathematics.

        Gradient formula: ∇J = (1/n) * X^T * (σ(z) - y)
        Where σ(z) are the probabilities.
        """
        n_samples = X.shape[0]
        errors = probabilities - y  # σ(z) - y
        gradient = (1.0 / n_samples) * np.dot(X.T, errors)
        return gradient

    def update_beta(self, beta, gradient, eta):
        """
        Step 3: Update betas for next iteration.

        Update rule: β_new = β_old - η * ∇J
        """
        beta_new = beta - eta * gradient
        return beta_new

    def train_iteration(self, X, y, iteration_num):
        """Perform one complete training iteration with tracking."""
        # Step 1: Compute probabilities
        probabilities = self.compute_probabilities(X, self.beta)

        # Calculate errors
        errors = self.compute_errors(probabilities, y)

        # Store iteration data
        iteration_data = {
            'iteration': iteration_num,
            'beta': self.beta.copy(),
            'probabilities': probabilities.copy(),
            'errors': errors.copy(),
            'y_true': y.copy()
        }
        self.iterations_data.append(iteration_data)

        # Step 2: Compute gradient using mathematics
        gradient = self.compute_gradient_math(X, probabilities, y)

        # Step 3: Update beta for next iteration
        self.beta = self.update_beta(self.beta, gradient, self.eta)

        return self.beta

    def train(self, X, y, n_iterations=10):
        """Train for n iterations, tracking all steps."""
        # Store initial state (iteration 0)
        probabilities = self.compute_probabilities(X, self.beta)
        errors = self.compute_errors(probabilities, y)
        self.iterations_data.append({
            'iteration': 0,
            'beta': self.beta.copy(),
            'probabilities': probabilities.copy(),
            'errors': errors.copy(),
            'y_true': y.copy()
        })

        # Run iterations
        for i in range(1, n_iterations + 1):
            self.train_iteration(X, y, i)

        return self.beta

    def get_detailed_table(self, max_samples=10):
        """Generate detailed table with probabilities and errors."""
        lines = []
        lines.append("=" * 100)
        lines.append("SIGMOID GENERATOR - DETAILED ITERATION TABLE")
        lines.append("=" * 100)
        lines.append(f"Learning rate η: {self.eta}")
        lines.append("=" * 100)

        for iter_data in self.iterations_data:
            iteration = iter_data['iteration']
            beta = iter_data['beta']
            probs = iter_data['probabilities']
            errors = iter_data['errors']
            y_true = iter_data['y_true']

            lines.append(f"\nITERATION {iteration}:")
            lines.append(f"  Beta: β0={beta[0]:.6f}, β1={beta[1]:.6f}, β2={beta[2]:.6f}")
            lines.append(f"  {'Sample':<8} {'y_true':<8} {'σ(z)':<12} {'Error':<12}")
            lines.append("  " + "-" * 50)

            n_show = min(max_samples, len(probs))
            for i in range(n_show):
                lines.append(f"  {i:<8} {int(y_true[i]):<8} {probs[i]:<12.6f} {errors[i]:<12.6f}")

            if len(probs) > max_samples:
                lines.append(f"  ... ({len(probs) - max_samples} more samples)")

            avg_error = np.mean(np.abs(errors))
            lines.append(f"  Average Absolute Error: {avg_error:.6f}")

        lines.append("\n" + "=" * 100)
        return "\n".join(lines)

    def get_summary_table(self):
        """Generate summary table showing beta evolution."""
        lines = []
        lines.append("=" * 80)
        lines.append("BETA WEIGHTS PROGRESSION SUMMARY")
        lines.append("=" * 80)
        lines.append(f"{'Iteration':<12} {'β0 (bias)':<18} {'β1 (X0)':<18} {'β2 (X1)':<18} {'Avg |Error|':<12}")
        lines.append("-" * 80)

        for iter_data in self.iterations_data:
            iteration = iter_data['iteration']
            beta = iter_data['beta']
            avg_error = np.mean(np.abs(iter_data['errors']))
            lines.append(f"{iteration:<12} {beta[0]:<18.6f} {beta[1]:<18.6f} {beta[2]:<18.6f} {avg_error:<12.6f}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def predict_probability(self, X):
        """Predict probabilities using current beta weights."""
        return self.compute_probabilities(X, self.beta)

    def predict(self, X, threshold=0.5):
        """Predict class labels using current beta weights."""
        probabilities = self.predict_probability(X)
        return (probabilities >= threshold).astype(int)
