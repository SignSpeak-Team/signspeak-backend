"""Prometheus metrics for Vision Service monitoring."""

from prometheus_client import Counter, Histogram

# Prediction counters
PREDICTIONS_TOTAL = Counter(
    "signspeak_predictions_total",
    "Total predictions by model and status",
    ["model_type", "status"],
)

# Latency histograms (buckets in seconds)
PREDICTION_LATENCY = Histogram(
    "signspeak_prediction_latency_seconds",
    "Prediction processing time",
    ["model_type"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Confidence distribution
PREDICTION_CONFIDENCE = Histogram(
    "signspeak_prediction_confidence",
    "Prediction confidence distribution",
    ["model_type"],
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
