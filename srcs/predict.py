# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    predict.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ravazque <ravazque@student.42madrid.com    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/08/18 11:07:28 by ravazque          #+#    #+#              #
#    Updated: 2026/08/19 16:45:09 by ravazque         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""Prompt for a mileage and print the price estimated by the trained model."""

import math
import sys

from utils import THETAS_PATH, estimate_price, load_thetas


def read_mileage():
    try:
        raw = input("Mileage (km): ")
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)

    try:
        mileage = float(raw.strip())
    except ValueError:
        sys.exit(f"predict: '{raw}' is not a number")
    if not math.isfinite(mileage) or mileage < 0:
        sys.exit("predict: mileage must be a non-negative number")
    return mileage


def main():
    try:
        theta0, theta1 = load_thetas(THETAS_PATH)
    except (OSError, ValueError) as error:
        sys.exit(f"predict: {error}")

    if theta0 == 0.0 and theta1 == 0.0:
        print("Model not trained yet (θ0 = θ1 = 0): run train.py first.")

    mileage = read_mileage()
    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price: {price:.2f}")
    if price < 0:
        print("(negative estimate: this mileage is beyond what the fitted line can model)")


if __name__ == "__main__":
    main()
