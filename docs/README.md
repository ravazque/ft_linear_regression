# ft_linear_regression

## Description

ft_linear_regression is an introduction to machine learning: a **single-variable linear regression** trained with **gradient descent**, written from scratch in plain Python, that predicts the price of a car from its mileage.

The project is two programs:

- **train** reads the dataset (`information/data.csv`), fits the hypothesis `estimatePrice(km) = θ0 + θ1 · km` by repeatedly applying the gradient descent update rule with a simultaneous update of both parameters, and stores the result in `information/thetas.json`.
- **predict** prompts for a mileage and prints the estimated price. Until the model is trained both thetas are 0, so it answers 0. Past the largest mileage in the dataset, where the line can go negative, it warns that the estimate has no data behind it and also prints an extrapolated price: an exponential decay that starts with the line's value and slope and never drops below 0. `EXTRAPOLATE = False` at the top of `predict.py` keeps the warning and drops the extrapolated price.

No library performs the regression: the hypothesis, the update loop and the cost are hand-written and use only the Python standard library. Because raw mileage reaches 240 000 km, `train` standardizes it (`(km − mean) / std`) before the descent and converts the thetas back afterwards, so `predict` applies the hypothesis directly on raw kilometres. On standardized mileage each step scales the remaining error by `|1 − learningRate|`, so `train` only accepts a learning rate strictly between 0 and 2 and refuses anything else before training.

## Instructions

Requires Python 3, nothing else to install.

```sh
python3 srcs/train.py                 # fit the model and write information/thetas.json
python3 srcs/predict.py               # prompts "Mileage (km): " and prints the estimate
```

`train.py` accepts `--learning-rate` (default `0.1`, between 0 and 2), `--iterations` (default `1000`) and `--data path.csv` to use another `km,price` dataset. Both programs locate `information/` relative to their own file and work from any directory.

Expected result on the provided dataset: `θ0 = 8499.599650`, `θ1 = −0.021449`.

## Resources

- Andrew Ng, *Machine Learning* (Coursera) — linear regression with one variable, gradient descent, feature scaling.
- [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) and [Feature scaling](https://en.wikipedia.org/wiki/Feature_scaling) on Wikipedia.
