# ft_linear_regression

## 📖 About

"ft_linear_regression" is a machine learning project written in **Python**: a
car price estimator. Two programs share one model — `train` reads a dataset of
mileages and prices and fits a straight line through it, `predict` asks for a
mileage and answers with the price that line gives back.

The interesting part is not the line, it is how it is found. The model is the
hypothesis `estimatePrice(km) = θ0 + θ1 · km`, and its two parameters are
learned with **gradient descent**: starting from `θ0 = θ1 = 0`, the program
measures how far the line is from every point of the dataset, moves both
parameters a small step in the direction that reduces that error, and repeats
that step a fixed number of times, long after the error has stopped shrinking.
No library does this for it — the sums, the
update rule and the cost are written by hand with nothing beyond the Python
standard library.

Raw mileage goes up to 240 000 km, and on numbers that large the update steps
overshoot and the descent blows up. `train` therefore works on a
**standardized** mileage (`(km − mean) / std`), where the descent converges in
about a hundred iterations, and converts the parameters back to raw kilometres
before saving them, so `predict` applies the hypothesis exactly as written.

## 🎯 Objectives

- Reading a `km,price` dataset and rejecting every malformed row
- Implementing the linear hypothesis `θ0 + θ1 · mileage`
- Implementing the gradient descent update rule with a **simultaneous** update
  of both parameters
- Standardizing the feature so the descent converges, and undoing the scaling
  so the saved parameters work on raw mileage
- Refusing learning rates that cannot converge instead of saving garbage
  parameters
- Persisting the model between programs and behaving sensibly before training

## 📋 Function Overview

<details>
<summary><strong>ft_linear_regression</strong></summary>

<br>

| Module | Feature | Description |
|--------|---------|-------------|
| **utils** | Hypothesis | `estimate_price` — `θ0 + θ1 · mileage`, the one formula both programs share |
| **utils** | Dataset | `load_dataset` reads a CSV whose header is exactly `km,price`; a short, long or non-numeric row (`nan` and `inf` included) is reported with its line number, fewer than two rows is an error |
| **utils** | Parameters | `load_thetas` / `save_thetas` keep `θ0`, `θ1` and `km_max` (the largest training mileage) in a small JSON file; a missing file means `(0, 0)`, a malformed one is an error |
| **utils** | Statistics | `mean`, population `std` and `mean_squared_error`, written by hand |
| **train** | Standardization | Mileage becomes `(km − μ) / σ` before the descent; a dataset where every mileage is the same is refused |
| **train** | Gradient descent | The update rule: the error of every point is computed once per iteration with the current parameters, then `θ0` and `θ1` are both moved |
| **train** | Learning rate bound | Only rates strictly between 0 and 2 are accepted: on standardized mileage that is exactly where the descent converges. A cost that overflows also aborts the run; the saved parameters are never touched on failure |
| **train** | De-standardization | `θ1 = θ1′ / σ`, `θ0 = θ0′ − θ1 · μ`, so the saved line is expressed in raw kilometres |
| **train** | Options | `--learning-rate` (0.1, between 0 and 2), `--iterations` (1000), `--data` for another dataset |
| **predict** | Prompt | `Mileage (km):` — any finite non-negative number is accepted; text, negatives, `nan`, `inf` and values too large for a float are rejected |
| **predict** | Untrained model | With `(0, 0)` parameters it says so and answers 0 |
| **predict** | Out-of-range warning | Past `km_max` the estimate is flagged as unreliable, or as unrealistic once the line has dropped below zero: there is no training data there |
| **predict** | Extrapolation | Past `km_max` a second estimate follows the line's price and slope at `km_max` and then decays exponentially, so it never drops below zero; `EXTRAPOLATE = False` at the top of `predict.py` turns it off |
| **both** | Error path | Every I/O or data error ends as `program: message` on stderr with exit status 1 |

<br>

</details>

<details>
<summary><strong>Usage Example & Testing</strong></summary>

### Train, then predict

```bash
$ python3 srcs/train.py
iteration      1   cost 33911113.85
iteration    100   cost 445645.27
iteration    200   cost 445645.25
…
iteration   1000   cost 445645.25

theta0 = 8499.599650
theta1 = -0.021449
saved to …/information/thetas.json

$ python3 srcs/predict.py
Mileage (km): 50000
Estimated price: 7427.15
```

The cost is the mean squared error over the 24 cars of the dataset. A flat
line at zero scores 41.8 million, the first step already brings it to 33.9
million, and it settles after about a hundred iterations: every extra
kilometre costs 2.1 cents, and a car with no mileage is worth 8 499.60.

### Beyond the dataset

```bash
$ echo 300000 | python3 srcs/predict.py
Mileage (km): Estimated price: 2064.91
(unreliable: no training data past 240000 km, the line is only extrapolated)
Extrapolated price: 2283.17
$ echo 400000 | python3 srcs/predict.py
Mileage (km): Estimated price: -79.99
(unrealistic: no training data past 240000 km, the line has dropped below 0)
Extrapolated price: 1204.00
```

The linear estimate is always printed as the line gives it, negative or not.
Past the largest mileage of the dataset it is flagged, and a second,
extrapolated estimate is added; up to that mileage the output is unchanged.
Setting `EXTRAPOLATE = False` at the top of `predict.py` drops the
extrapolated line and keeps the warning.

### Before training

```bash
$ rm information/thetas.json
$ python3 srcs/predict.py
Model not trained yet (θ0 = θ1 = 0): run train.py first.
Mileage (km): 50000
Estimated price: 0.00
```

### The learning rate matters

```bash
$ python3 srcs/train.py --learning-rate 2
train: the learning rate must be above 0 and below 2
$ python3 srcs/train.py --learning-rate 0.001 --iterations 100 | grep theta1
theta1 = -0.002042            # ten times too small: the run stopped early
```

From 2 on the steps overshoot the minimum by as much as they started away
from it (exactly 2) or more (above 2), so the error never shrinks; such a rate
is refused before training and the previous parameters are kept. Too small a
rate converges, but not within the iteration budget.

### Error handling

```bash
$ printf 'km,price\n1000,5\n2000\n' > bad.csv
$ python3 srcs/train.py --data bad.csv
train: bad.csv: line 3: expected two numbers
$ echo '[1, 2]' > information/thetas.json
$ python3 srcs/predict.py
predict: …/information/thetas.json: expected {"theta0": number, "theta1": number}
$ python3 srcs/predict.py
Mileage (km): -5
predict: mileage must be a finite, non-negative number
```

### Full sweep

```bash
python3 srcs/train.py | tail -3                                         # θ0 8499.599650, θ1 -0.021449
echo 120000 | python3 srcs/predict.py                                   # 5925.72
for lr in 0.01 0.1 1 1.9; do python3 srcs/train.py --learning-rate $lr | tail -3 | head -1; done
python3 srcs/train.py --learning-rate 2.5 ; echo "exit $?"               # refused, exit 1
```

<br>

</details>

## 🚀 Installation & Structure

<details>
<summary><strong>📥 Setup & Usage</strong></summary>

<br>

### Host requirements

| Tool | Needed for | Notes |
|------|------------|-------|
| `python3` ≥ 3.8 | Everything | Only the standard library: `csv`, `json`, `math`, `argparse`, `pathlib` |

There is no virtual environment and nothing to build: the programs run
directly from `srcs/`.

### Running

```bash
python3 srcs/train.py                       # fit the line, write information/thetas.json
python3 srcs/predict.py                     # ask for a mileage, print the estimate
```

Both programs locate `information/` relative to their own file, so they work
from any working directory. `train.py` accepts `--data path.csv` to use
another dataset with the same `km,price` header, plus `--learning-rate` and
`--iterations`.

### Dataset format

```
km,price          # header, exactly these two names
240000,3650       # one car per line: mileage in km, price
139800,3800
…
```

Values are read as floats. A row with a missing, extra or non-numeric field
(`nan` and `inf` included) stops the program with its line number; the file needs at least two rows, and not all
mileages may be equal.

<br>

</details>

<details>
<summary><strong>📁 Project Structure</strong></summary>

<br>

```
ft_linear_regression/
│
├── README.md                             # Main project documentation
├── .gitignore                            # Files and directories ignored by Git
│
├── docs/
│   └── README.md                         # Condensed project documentation
│
├── information/
│   ├── data.csv                          # Training dataset: 24 cars, km and price
│   └── thetas.json                       # Written by train.py, ignored by Git
│
└── srcs/
    ├── utils.py                          # Hypothesis, dataset and parameter I/O, mean/std, MSE
    ├── train.py                          # Standardize → gradient descent → de-standardize → save
    └── predict.py                        # Prompt for a mileage, apply the hypothesis
```

`utils.py` is the only module the others import: it owns the hypothesis and
the file formats, so `train` and `predict` cannot drift apart on what a
parameter file or a dataset looks like.

<br>

</details>

<details>
<summary><strong>🧱 Algorithm Overview</strong></summary>

<br>

### Pipeline

```
data.csv ─> standardize km ─> gradient descent ─> de-standardize ─> thetas.json ─> predict
```

### Hypothesis and cost

```
estimatePrice(km) = θ0 + θ1 · km

cost(θ0, θ1) = (1/m) · Σ (estimatePrice(km[i]) − price[i])²        m = 24 cars
```

The cost is a bowl-shaped surface over `(θ0, θ1)` with a single minimum; the
line at the bottom of the bowl is the one closest to every point at once.

### Gradient descent — one step

```
errors[i] = estimatePrice(km[i]) − price[i]          with the current θ0, θ1

tmpθ0 = learningRate · (1/m) · Σ errors[i]
tmpθ1 = learningRate · (1/m) · Σ errors[i] · km[i]

θ0 = θ0 − tmpθ0          both updated from the same errors:
θ1 = θ1 − tmpθ1          neither one sees the other's new value
```

The two sums are the partial derivatives of the cost: they point uphill, so
the parameters move the other way, scaled by the learning rate. The errors are
computed once per iteration and only then are both parameters changed, which
is what a simultaneous update means.

### Standardization — why the descent converges

```
raw km:            θ1 step  ∝  Σ errors · km        km ≈ 10⁵  ->  step ≈ 10⁵ × θ0 step
standardized km:   x′ = (km − μ) / σ                mean 0, std 1
```

With raw mileage the two parameters live on scales five orders of magnitude
apart: a learning rate small enough for `θ1` leaves `θ0` crawling, one large
enough for `θ0` sends `θ1` to infinity. On standardized mileage the second
derivatives of the cost form the identity matrix — `mean(x′) = 0`,
`mean(x′²) = 1` — so `θ0` and `θ1` are independent and each step shrinks the
remaining error by a factor of `|1 − learningRate|`. At `0.1` that is `0.9` per
step, which is why the cost is flat after a hundred iterations; from `2` on
the factor is `1` or more and the error never shrinks again.

### Learning rate bound

That factor makes the range of working learning rates exact: `0 < rate < 2`,
whatever the dataset, because standardization always produces the same
identity curvature. `train` checks the rate against it before the first step
instead of watching the cost. A rate of exactly `2` shows why: the parameters
jump between `(0, 0)` and twice the optimum, the cost is the same at both
points, and a check for a rising cost would never fire. During the descent
only an overflowing cost (values too large for a float) stops the run.

### De-standardization — the saved line

```
θ0′ + θ1′ · (km − μ) / σ   ==   (θ0′ − θ1′ · μ / σ)  +  (θ1′ / σ) · km
                                 └──── θ0 ────┘        └─ θ1 ─┘
```

Only the mileage is rescaled, never the price, so the cost printed during
training is a real squared price error and the two conversions above are the
whole bridge between the training space and the raw one. `predict` never
learns that a standardization happened.

### Extrapolation past the data

```
p0 = θ0 + θ1 · km_max                                  the line's price where the data ends
extrapolated(km) = p0 · e^(θ1 · (km − km_max) / p0)    for km > km_max
```

At `km_max` the curve has the same value and the same slope as the line, so
it leaves it without a jump or a kink; after that it keeps falling but only
approaches zero. The line reaches zero at about 396 000 km, the curve is still
at 1 204 at 400 000 km. The shape of the tail is an assumption, not something
learned from the data: it only replaces an answer the line cannot give. It is
skipped when the line does not go down or is already at or below zero at
`km_max`.

### Result

```
θ0 = 8499.599650      θ1 = −0.021449      cost = 445645.25
```

Gradient descent lands on the same line as the closed-form least-squares
solution, to within `10⁻¹¹`.

<br>

</details>

## 💡 Key Learning Outcomes

- **Supervised learning from scratch**: a hypothesis, a cost function and an
  optimiser are all a linear regression is
- **Gradient descent**: derivatives point uphill, the learning rate sets the
  stride, and the two parameters must be updated from the same snapshot
- **Feature scaling**: convergence depends on the scale of the input, not on
  the algorithm; standardizing the feature decouples the parameters
- **Reversible preprocessing**: whatever is done to the data before training
  has to be undone on the parameters afterwards
- **Convergence limits**: on a standardized feature each step scales the
  error by `|1 − learningRate|`, so the working rates are known in advance
- **Defensive input handling**: an untrusted CSV, a hand-edited parameter file
  and a free-text prompt all fail with a message, not a traceback

## ⚙️ Technical Specifications

- **Language**: Python 3, standard library only
- **Dependencies**: `python3` 3.8 or later — nothing to install
- **Model**: `price = θ0 + θ1 · km`, two parameters stored as JSON in
  `information/thetas.json` together with the largest training mileage,
  used for the extrapolated estimate past the data
- **Training**: batch gradient descent, learning rate `0.1`, `1000`
  iterations by default, learning rate restricted to `(0, 2)`
- **Preprocessing**: mileage standardized with the population mean and
  standard deviation; parameters converted back before saving
- **Dataset**: 24 cars, mileage from 22 899 to 240 000 km, price from 3 650
  to 8 290
- **Result**: `θ0 = 8499.599650`, `θ1 = −0.021449`, final cost (MSE)
  `445645.25`
- **Measured**: a full training run, cost logging included, takes about 50 ms
- **Interface**: `predict` prompts on standard input; both programs exit with
  status 1 and a `program: message` line on standard error on any failure,
  except malformed command-line options, which get `argparse`'s usage
  message and status 2

---

> [!NOTE]
> ft_linear_regression is a small program with the whole of machine learning
> in miniature: a hypothesis, a cost, and an optimiser that walks downhill.
> Everything that makes larger models work — scaling the inputs, choosing a
> learning rate, watching the cost — is already here, on two parameters and
> twenty-four cars.
