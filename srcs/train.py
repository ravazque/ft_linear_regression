# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ravazque <ravazque@student.42madrid.com    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/08/17 18:31:05 by ravazque          #+#    #+#              #
#    Updated: 2026/09/24 12:26:53 by ravazque         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""Fits the line with gradient descent and saves θ0, θ1 for predict."""

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
    m = len(mileages)
    theta0, theta1 = 0.0, 0.0
    log_every = max(1, iterations // 10)

    for i in range(1, iterations + 1):
        errors = [estimate_price(x, theta0, theta1) - y for x, y in zip(mileages, prices)]
        tmp_theta0 = learning_rate * sum(errors) / m
        tmp_theta1 = learning_rate * sum(e * x for e, x in zip(errors, mileages)) / m
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        cost = mean_squared_error(mileages, prices, theta0, theta1)
        if not math.isfinite(cost):
            raise ValueError(f"cost overflowed at iteration {i}: the values are too large")
        if i == 1 or i % log_every == 0:
            print(f"iteration {i:>6}   cost {cost:.2f}")

    return theta0, theta1


def train(data_path, learning_rate, iterations):
    mileages, prices = load_dataset(data_path)
    mu, sigma = mean(mileages), std(mileages)
    if sigma == 0:
        raise ValueError("all mileages are identical: nothing to fit")

    # raw km is too big for the descent, so it runs on standardized km
    scaled = [(x - mu) / sigma for x in mileages]
    theta0_s, theta1_s = gradient_descent(scaled, prices, learning_rate, iterations)

    # undo the scaling so the thetas work on raw km
    theta1 = theta1_s / sigma
    theta0 = theta0_s - theta1 * mu
    return theta0, theta1, max(mileages)


def main():
    parser = argparse.ArgumentParser(description="Fit price = θ0 + θ1·km with gradient descent.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV with a km,price header")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    # on standardized km each step scales the error by |1 - learning rate|
    if not 0 < args.learning_rate < 2:
        sys.exit("train: the learning rate must be above 0 and below 2")
    if args.iterations <= 0:
        sys.exit("train: iterations must be positive")

    try:
        theta0, theta1, km_max = train(args.data, args.learning_rate, args.iterations)
        save_thetas(theta0, theta1, km_max, THETAS_PATH)
    except (OSError, ValueError) as error:
        sys.exit(f"train: {error}")
    except KeyboardInterrupt:
        sys.exit("\ntrain: interrupted, nothing saved")

    print(f"\ntheta0 = {theta0:.6f}")
    print(f"theta1 = {theta1:.6f}")
    print(f"saved to {THETAS_PATH}")


if __name__ == "__main__":
    main()
