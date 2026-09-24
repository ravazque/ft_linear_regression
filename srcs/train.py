"""Train the linear model with gradient descent and save θ0/θ1 for predict.py.

Raw mileage goes up to 240 000 km, which makes the gradient descent update
steps blow up, so the descent runs on standardized mileage ((km - mean) / std).
The thetas are converted back afterwards: predict.py applies the hypothesis on
raw mileage and never needs to know about the scaling.
"""

import argparse
import math
import sys
from pathlib import Path

from utils import (
    DATA_PATH,
    THETAS_PATH,
    estimate_price,
    load_dataset,
    mean,
    mean_squared_error,
    save_thetas,
    std,
)


def gradient_descent(mileages, prices, learning_rate, iterations):
    """Batch gradient descent: both thetas updated simultaneously each step."""
    m = len(mileages)
    theta0, theta1 = 0.0, 0.0
    cost = mean_squared_error(mileages, prices, theta0, theta1)
    log_every = max(1, iterations // 10)

    for i in range(1, iterations + 1):
        errors = [estimate_price(x, theta0, theta1) - y for x, y in zip(mileages, prices)]
        tmp_theta0 = learning_rate * sum(errors) / m
        tmp_theta1 = learning_rate * sum(e * x for e, x in zip(errors, mileages)) / m
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        # Gradient descent must lower the cost every step; going up means the
        # learning rate overshoots the minimum and the thetas are diverging.
        previous, cost = cost, mean_squared_error(mileages, prices, theta0, theta1)
        if not math.isfinite(cost) or (cost > previous and not math.isclose(cost, previous)):
            raise ValueError(f"cost went up at iteration {i}: the learning rate is too high")
        if i == 1 or i % log_every == 0:
            print(f"iteration {i:>6}   cost {cost:.2f}")

    return theta0, theta1


def train(data_path, learning_rate, iterations):
    mileages, prices = load_dataset(data_path)
    mu, sigma = mean(mileages), std(mileages)
    if sigma == 0:
        raise ValueError("all mileages are identical: nothing to fit")

    scaled = [(x - mu) / sigma for x in mileages]
    theta0_s, theta1_s = gradient_descent(scaled, prices, learning_rate, iterations)

    # θ0_s + θ1_s·(x − μ)/σ  ==  (θ0_s − θ1_s·μ/σ) + (θ1_s/σ)·x
    theta1 = theta1_s / sigma
    theta0 = theta0_s - theta1 * mu
    return theta0, theta1


def main():
    parser = argparse.ArgumentParser(description="Fit price = θ0 + θ1·km with gradient descent.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV with a km,price header")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    if args.learning_rate <= 0 or args.iterations <= 0:
        sys.exit("learning rate and iterations must be positive")

    try:
        theta0, theta1 = train(args.data, args.learning_rate, args.iterations)
        save_thetas(theta0, theta1, THETAS_PATH)
    except (OSError, ValueError) as error:
        sys.exit(f"train: {error}")

    print(f"\ntheta0 = {theta0:.6f}")
    print(f"theta1 = {theta1:.6f}")
    print(f"saved to {THETAS_PATH}")


if __name__ == "__main__":
    main()
