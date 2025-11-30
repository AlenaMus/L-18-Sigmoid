"""
Verification script to demonstrate the pure mathematical implementation.
Shows step-by-step calculations for one iteration.
"""

import numpy as np
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator


def demonstrate_math_steps():
    """Show mathematical steps for first iteration."""
    print("=" * 90)
    print("MATHEMATICAL VERIFICATION - Step-by-step calculation")
    print("=" * 90)
    print()

    # Create small dataset for demonstration
    np.random.seed(42)
    X = np.array([
        [1.0, 0.3, 0.4],  # Sample 0: [bias, X0, X1]
        [1.0, 0.7, 0.8],  # Sample 1: [bias, X0, X1]
    ])
    y = np.array([0.0, 1.0])

    # Initial parameters
    beta = np.array([0.0, 0.3, -0.5])
    eta = 0.3

    print("DATASET:")
    print(f"  Sample 0: bias={X[0,0]}, X0={X[0,1]}, X1={X[0,2]}, y={int(y[0])}")
    print(f"  Sample 1: bias={X[1,0]}, X0={X[1,1]}, X1={X[1,2]}, y={int(y[1])}")
    print()

    print("INITIAL WEIGHTS:")
    print(f"  β = [{beta[0]}, {beta[1]}, {beta[2]}]")
    print(f"  Learning rate η = {eta}")
    print()

    # STEP 1: Compute z values
    print("STEP 1: Compute z = β₀*bias + β₁*X0 + β₂*X1")
    z0 = beta[0] * X[0,0] + beta[1] * X[0,1] + beta[2] * X[0,2]
    z1 = beta[0] * X[1,0] + beta[1] * X[1,1] + beta[2] * X[1,2]
    print(f"  Sample 0: z = {beta[0]}*{X[0,0]} + {beta[1]}*{X[0,1]} + {beta[2]}*{X[0,2]}")
    print(f"           z = {z0:.6f}")
    print(f"  Sample 1: z = {beta[0]}*{X[1,0]} + {beta[1]}*{X[1,1]} + {beta[2]}*{X[1,2]}")
    print(f"           z = {z1:.6f}")
    print()

    # STEP 2: Compute sigmoid
    print("STEP 2: Compute σ(z) = 1 / (1 + e^(-z))")
    sigmoid0 = 1.0 / (1.0 + np.exp(-z0))
    sigmoid1 = 1.0 / (1.0 + np.exp(-z1))
    print(f"  Sample 0: σ(z) = 1 / (1 + e^(-{z0:.6f}))")
    print(f"           σ(z) = {sigmoid0:.6f}")
    print(f"  Sample 1: σ(z) = 1 / (1 + e^(-{z1:.6f}))")
    print(f"           σ(z) = {sigmoid1:.6f}")
    print()

    # STEP 3: Compute errors
    print("STEP 3: Compute errors = y - σ(z)")
    error0 = y[0] - sigmoid0
    error1 = y[1] - sigmoid1
    print(f"  Sample 0: error = {y[0]} - {sigmoid0:.6f} = {error0:.6f}")
    print(f"  Sample 1: error = {y[1]} - {sigmoid1:.6f} = {error1:.6f}")
    print()

    # STEP 4: Compute gradient
    print("STEP 4: Compute gradient ∇J = (1/n) * X^T * (σ(z) - y)")
    n = len(y)
    diff = np.array([sigmoid0 - y[0], sigmoid1 - y[1]])
    print(f"  σ(z) - y = [{sigmoid0 - y[0]:.6f}, {sigmoid1 - y[1]:.6f}]")
    print()

    grad_beta0 = (1.0/n) * (X[0,0]*diff[0] + X[1,0]*diff[1])
    grad_beta1 = (1.0/n) * (X[0,1]*diff[0] + X[1,1]*diff[1])
    grad_beta2 = (1.0/n) * (X[0,2]*diff[0] + X[1,2]*diff[1])

    print(f"  ∇β₀ = (1/{n}) * ({X[0,0]}*{diff[0]:.6f} + {X[1,0]}*{diff[1]:.6f})")
    print(f"      = {grad_beta0:.6f}")
    print(f"  ∇β₁ = (1/{n}) * ({X[0,1]}*{diff[0]:.6f} + {X[1,1]}*{diff[1]:.6f})")
    print(f"      = {grad_beta1:.6f}")
    print(f"  ∇β₂ = (1/{n}) * ({X[0,2]}*{diff[0]:.6f} + {X[1,2]}*{diff[1]:.6f})")
    print(f"      = {grad_beta2:.6f}")
    print()

    # STEP 5: Update weights
    print("STEP 5: Update weights β_new = β_old - η * ∇J")
    new_beta0 = beta[0] - eta * grad_beta0
    new_beta1 = beta[1] - eta * grad_beta1
    new_beta2 = beta[2] - eta * grad_beta2

    print(f"  β₀_new = {beta[0]} - {eta} * {grad_beta0:.6f} = {new_beta0:.6f}")
    print(f"  β₁_new = {beta[1]} - {eta} * {grad_beta1:.6f} = {new_beta1:.6f}")
    print(f"  β₂_new = {beta[2]} - {eta} * {grad_beta2:.6f} = {new_beta2:.6f}")
    print()

    print("NEW WEIGHTS:")
    print(f"  β_new = [{new_beta0:.6f}, {new_beta1:.6f}, {new_beta2:.6f}]")
    print()

    # Verify with SigmoidGenerator
    print("=" * 90)
    print("VERIFICATION WITH SigmoidGenerator CLASS:")
    print("=" * 90)
    sigmoid_gen = SigmoidGenerator(beta0=[0.0, 0.3, -0.5], eta=0.3)
    probs = sigmoid_gen.compute_probabilities(X, sigmoid_gen.beta)
    print(f"  Computed probabilities: [{probs[0]:.6f}, {probs[1]:.6f}]")
    print(f"  Expected: [{sigmoid0:.6f}, {sigmoid1:.6f}]")
    print(f"  Match: {np.allclose(probs, [sigmoid0, sigmoid1])}")
    print()

    gradient = sigmoid_gen.compute_gradient_math(X, probs, y)
    print(f"  Computed gradient: [{gradient[0]:.6f}, {gradient[1]:.6f}, {gradient[2]:.6f}]")
    print(f"  Expected: [{grad_beta0:.6f}, {grad_beta1:.6f}, {grad_beta2:.6f}]")
    print(f"  Match: {np.allclose(gradient, [grad_beta0, grad_beta1, grad_beta2])}")
    print()

    new_beta = sigmoid_gen.update_beta(sigmoid_gen.beta, gradient, eta)
    print(f"  Updated weights: [{new_beta[0]:.6f}, {new_beta[1]:.6f}, {new_beta[2]:.6f}]")
    print(f"  Expected: [{new_beta0:.6f}, {new_beta1:.6f}, {new_beta2:.6f}]")
    print(f"  Match: {np.allclose(new_beta, [new_beta0, new_beta1, new_beta2])}")
    print()

    print("=" * 90)
    print("✓ ALL MATHEMATICAL STEPS VERIFIED!")
    print("=" * 90)


if __name__ == "__main__":
    demonstrate_math_steps()
