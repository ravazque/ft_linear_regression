# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    utils.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ravazque <ravazque@student.42madrid.com    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/08/17 17:42:13 by ravazque          #+#    #+#              #
#    Updated: 2026/08/21 12:08:37 by ravazque         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""Helpers shared by train and predict.

Plain Python only: the linear hypothesis, dataset/theta I/O and the small
statistics needed to standardize the mileage before gradient descent.
"""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "information" / "data.csv"
THETAS_PATH = ROOT / "information" / "thetas.json"


def estimate_price(mileage, theta0, theta1):
    """Linear hypothesis: estimatePrice(mileage) = θ0 + θ1 · mileage."""
    return theta0 + theta1 * mileage


def load_dataset(path=DATA_PATH):
    """Read a `km,price` CSV and return both columns as lists of floats."""
    mileages, prices = [], []
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["km", "price"]:
            raise ValueError(f"{path}: expected header 'km,price'")
        for row in reader:
            try:
                mileages.append(float(row["km"]))
                prices.append(float(row["price"]))
            except (TypeError, ValueError):
                raise ValueError(f"{path}: line {reader.line_num}: expected two numbers") from None
    if len(mileages) < 2:
        raise ValueError(f"{path}: need at least two rows to fit a line")
    return mileages, prices


def load_thetas(path=THETAS_PATH):
    """Return (θ0, θ1). A missing file means the model is untrained: both are 0."""
    path = Path(path)
    if not path.exists():
        return 0.0, 0.0
    try:
        with open(path) as file:
            data = json.load(file)
        theta0, theta1 = float(data["theta0"]), float(data["theta1"])
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        raise ValueError(f'{path}: expected {{"theta0": number, "theta1": number}}') from None
    if not (math.isfinite(theta0) and math.isfinite(theta1)):
        raise ValueError(f"{path}: thetas must be finite numbers")
    return theta0, theta1


def save_thetas(theta0, theta1, path=THETAS_PATH):
    with open(path, "w") as file:
        json.dump({"theta0": theta0, "theta1": theta1}, file, indent=2)
        file.write("\n")


def mean(values):
    return sum(values) / len(values)


def std(values):
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def mean_squared_error(mileages, prices, theta0, theta1):
    errors = (estimate_price(x, theta0, theta1) - y for x, y in zip(mileages, prices))
    return sum(e * e for e in errors) / len(prices)
