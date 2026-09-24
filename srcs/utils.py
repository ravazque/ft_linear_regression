# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    utils.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ravazque <ravazque@student.42madrid.com    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/08/17 17:42:13 by ravazque          #+#    #+#              #
#    Updated: 2026/09/23 21:37:14 by ravazque         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""Hypothesis, file loading/saving and small stats shared by train and predict."""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "information" / "data.csv"
THETAS_PATH = ROOT / "information" / "thetas.json"


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def load_dataset(path=DATA_PATH):
    mileages, prices = [], []
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["km", "price"]:
            raise ValueError(f"{path}: expected header 'km,price'")
        for row in reader:
            try:
                km, price = float(row["km"]), float(row["price"])
            except (TypeError, ValueError):
                km = price = math.nan
            if None in row or not (math.isfinite(km) and math.isfinite(price)):
                raise ValueError(f"{path}: line {reader.line_num}: expected two numbers")
            mileages.append(km)
            prices.append(price)
    if len(mileages) < 2:
        raise ValueError(f"{path}: need at least two rows to fit a line")
    return mileages, prices


def load_thetas(path=THETAS_PATH):
    """No file means the model is not trained yet: (0, 0, None)."""
    path = Path(path)
    if not path.exists():
        return 0.0, 0.0, None
    try:
        with open(path) as file:
            data = json.load(file)
        theta0, theta1 = float(data["theta0"]), float(data["theta1"])
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        raise ValueError(f'{path}: expected {{"theta0": number, "theta1": number}}') from None
    if not (math.isfinite(theta0) and math.isfinite(theta1)):
        raise ValueError(f"{path}: thetas must be finite numbers")
    try:
        km_max = None if data.get("km_max") is None else float(data["km_max"])
    except (TypeError, ValueError):
        km_max = math.nan
    if km_max is not None and not (math.isfinite(km_max) and km_max >= 0):
        raise ValueError(f"{path}: km_max must be a non-negative number")
    return theta0, theta1, km_max


def save_thetas(theta0, theta1, km_max, path=THETAS_PATH):
    with open(path, "w") as file:
        json.dump({"theta0": theta0, "theta1": theta1, "km_max": km_max}, file, indent=2)
        file.write("\n")


def mean(values):
    return sum(values) / len(values)


def std(values):
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def mean_squared_error(mileages, prices, theta0, theta1):
    errors = (estimate_price(x, theta0, theta1) - y for x, y in zip(mileages, prices))
    return sum(e * e for e in errors) / len(prices)
