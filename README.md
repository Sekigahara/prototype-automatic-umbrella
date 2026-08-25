# Weather Analytics & Automated Umbrella

This project analyzes weather data and trains a Hidden Markov Model (HMM)
to estimate whether an automated umbrella should be **OPEN** or **CLOSED**.

The model uses minute-level weather observations and evaluates predictions
using rolling/walk-forward time blocks.

## 1. Requirements

- Python 3.9
- Conda
- Jupyter Notebook

The project uses a Conda environment named `weather_env`.

## 2. Create the Conda Environment

```bash
conda create -n weather_env python=3.9
conda activate weather_env
pip install -r requirements.txt
```

If additional packages are required by the project:

```bash
pip install <package-name>
```

## 3. Start Jupyter

```bash
conda activate weather_env
jupyter notebook
```

or:

```bash
jupyter lab
```

If the kernel is not available:

```bash
pip install ipykernel
python -m ipykernel install --user --name weather_env --display-name "Python (weather_env)"
```

Then select **Python (weather_env)** as the notebook kernel.

## 4. Project Workflow

The current workflow consists mainly of:

```text
analytics.ipynb
plot.ipynb
```

### `analytics.ipynb`

Main notebook for:

1. Loading weather data
2. Combining monthly datasets
3. Preparing weather features
4. Creating temporal features
5. Training the HMM
6. Performing rolling/walk-forward evaluation
7. Generating predictions
8. Calculating metrics
9. Saving evaluation results

The model operates on **1-minute weather observations**.

Current weather features:

```python
weather_cols = [
    "Temperature (°C)",
    "1-minute Precipitation (mm)",
    "Precipitation Presence (Presence/Absence)",
    "Wind Direction (deg)",
    "Wind Speed (m/s)",
    "Local Pressure (hPa)",
    "Humidity (%)"
]
```

### `plot.ipynb`

Used after training/evaluation to inspect the results.

It can be used to:

- Plot actual vs predicted umbrella states
- Plot HMM OPEN probabilities
- Visualize prediction errors
- Inspect individual evaluation folds
- Compare rainy periods
- Analyze OPEN/CLOSED prediction accuracy

The evaluation is performed at **1-minute resolution**.

## 5. Evaluation Strategy

The project uses chronological walk-forward evaluation rather than random
train/test splitting.

The general procedure is:

```text
Initial Training
      |
      v
Train HMM
      |
      v
Evaluate next day
      |
      v
Add new data to training
      |
      v
Train/update HMM
      |
      v
Evaluate next day
      |
      v
Repeat
```

This prevents future weather observations from leaking into the past.

Current configuration:

```python
N_STATES = 5
INITIAL_TRAIN_DAYS = 5
TRAIN_INTERVAL_DAYS = 5
HORIZON = 5
N_ITER = 50
```

Where:

- `N_STATES` = number of hidden HMM states
- `INITIAL_TRAIN_DAYS` = initial amount of training data
- `TRAIN_INTERVAL_DAYS` = amount of additional historical data before the next evaluation
- `HORIZON` = future prediction horizon in minutes
- `N_ITER` = maximum HMM training iterations

## 6. Running the Project

### Step 1 — Training and Evaluation

Open:

```text
analytics.ipynb
```

Run the notebook from the beginning.

The workflow is:

```text
Weather Data
     |
     v
Feature Preparation
     |
     v
Temporal Features
     |
     v
HMM Training
     |
     v
Walk-forward Evaluation
     |
     v
Metrics + Predictions
     |
     v
Saved Results
```

### Step 2 — Visual Evaluation

After `analytics.ipynb` finishes, open:

```text
plot.ipynb
```

Load the generated evaluation results and visualize the folds.

The plots should allow inspection of:

- Actual umbrella state
- Predicted umbrella state
- HMM probability of OPEN
- Prediction threshold
- Correct predictions
- Incorrect predictions
- Rain events

## 7. Important Notes

### Keep the data chronological

Do not randomly shuffle the weather data.

Weather is a time-series problem, so temporal ordering must be preserved.

### Keep minute-level resolution

The original weather observations are maintained at:

```text
1 observation = 1 minute
```

The model and evaluation should preserve this resolution.

### Evaluation data must remain unseen

The future evaluation day must not be included in HMM training before
prediction.

### Missing values

Weather data may contain missing values (`NaN`).

Before training, make sure the feature matrix does not contain:

```text
NaN
inf
-inf
```

## 8. Recommended Project Structure

```text
weather-analytics/
|
├── analytics.ipynb
├── plot.ipynb
├── README.md
|
├── data/
|   ├── January.csv
|   ├── February.csv
|   ├── March.csv
|   └── ...
|
├── results/
|   ├── metrics/
|   ├── predictions/
|   └── logs/
|
└── modules/
    └── model.py
```

## 9. Quick Start

For a new machine:

```bash
conda create -n weather_env python=3.9
conda activate weather_env

pip install numpy pandas matplotlib scikit-learn hmmlearn tqdm jupyter ipykernel

python -m ipykernel install --user     --name weather_env     --display-name "Python (weather_env)"

jupyter notebook
```

Then:

```text
1. Open analytics.ipynb
2. Select Python (weather_env)
3. Run the training/evaluation pipeline
4. Open plot.ipynb
5. Analyze the evaluation folds
```

## 10. Current Model Goal

The final objective is to determine whether the automated umbrella should be:

```text
OPEN
```

or:

```text
CLOSED
```

using historical and current weather patterns rather than relying only
on a deterministic rainfall threshold.

The HMM is intended to capture the **temporal/stochastic behavior of weather**
and provide a probability of the umbrella being in the OPEN state.

Currently, the prototype only considers water drop/raining condition, future development, we will cover the temperature effect that triggers the umbrella to be open and further performance improvement.
