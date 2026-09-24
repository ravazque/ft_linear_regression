# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    predict.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ravazque <ravazque@student.42madrid.com    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/08/18 11:07:28 by ravazque          #+#    #+#              #
#    Updated: 2026/09/23 23:12:48 by ravazque         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""Asks for a mileage and prints the estimated price."""

import math
import sys

from utils import THETAS_PATH, estimate_price, load_thetas

# False hides the extrapolated price, the out-of-range warning stays
EXTRAPOLATE = True


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
        sys.exit("predict: mileage must be a finite, non-negative number")
    return mileage


def extrapolated_price(mileage, theta0, theta1, km_max):
    """Past km_max: starts with the line's value and slope, decays and never hits 0."""
    edge = estimate_price(km_max, theta0, theta1)
    if not theta1 < 0 < edge:
        return None
    return edge * math.exp(theta1 * (mileage - km_max) / edge)


def main():
    try:
        theta0, theta1, km_max = load_thetas(THETAS_PATH)
    except (OSError, ValueError) as error:
        sys.exit(f"predict: {error}")

    if theta0 == 0.0 and theta1 == 0.0:
        print("Model not trained yet (θ0 = θ1 = 0): run train.py first.")

    mileage = read_mileage()
    price = round(estimate_price(mileage, theta0, theta1), 2) + 0.0  # + 0.0 avoids printing -0.00
    print(f"Estimated price: {price:.2f}")

    if km_max is not None and mileage > km_max:
        if price < 0:
            print(f"(unrealistic: no training data past {km_max:.0f} km, the line has dropped below 0)")
        else:
            print(f"(unreliable: no training data past {km_max:.0f} km, the line is only extrapolated)")
        extrapolated = extrapolated_price(mileage, theta0, theta1, km_max) if EXTRAPOLATE else None
        if extrapolated is not None:
            print(f"Extrapolated price: {extrapolated:.2f}")
    elif price < 0:
        print("(negative estimate: this mileage is beyond what the fitted line can model)")


if __name__ == "__main__":
    main()
