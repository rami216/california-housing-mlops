from prometheus_client import Counter, Histogram, Gauge, Info

# how many predictions, split by outcome
PREDICTIONS = Counter(
    "predictions_total",
    "Total prediction requests",
    ["status"],          # label: "success" or "error"
)

# how long they take
LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent producing a prediction",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0),
)

# what values the model is outputting
PREDICTION_VALUE = Histogram(
    "prediction_value",
    "Distribution of predicted house values",
    buckets=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
)

# what values are coming in
FEATURE_VALUE = Histogram(
    "feature_value",
    "Distribution of incoming feature values",
    ["feature"],
    buckets=(-3, -2, -1, 0, 1, 2, 3, 5, 10),
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "1 if the model is loaded and ready, 0 otherwise",
)

MODEL_INFO = Info(
    "model",
    "Version and provenance of the loaded model",
)

DRIFT_PSI = Gauge(
    "drift_psi",
    "PSI of recent traffic versus training data, per feature",
    ["feature"],
)

DRIFT_WORST_PSI = Gauge(
    "drift_worst_psi",
    "Highest PSI across all features",
)

EXTRAPOLATIONS = Counter(
    "extrapolations_total",
    "Predictions outside the training target range",
)