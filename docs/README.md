*This project has been created as part of the 42 curriculum by ravazque.*

---

## Description

ft_linear_regression is an introduction to machine learning: a **single-variable linear regression** trained with **gradient descent**, implemented from scratch, that predicts the price of a car from its mileage.

The project consists of two programs:

- **train** reads the dataset (`data.csv`) and fits the hypothesis `estimatePrice(km) = θ0 + θ1 · km` by iteratively updating both parameters with simultaneous gradient descent steps, then stores the resulting thetas.
- **predict** asks for a mileage and returns the estimated price using the trained parameters (before any training, both thetas are zero and the prediction is 0).

No library is allowed to perform the regression itself — the cost function, the derivatives and the update loop are written by hand. Feature scaling is applied to the mileage so the descent converges, and the bonus part plots the data with the fitted line and measures the precision of the model.
