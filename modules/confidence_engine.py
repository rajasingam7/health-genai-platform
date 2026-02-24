import numpy as np


def calculate_confidence(df, analytics_result):
    """
    Calculate confidence score based on:
    - Sample size
    - Statistical stability
    - Dataset coverage
    """

    total_records = len(df)

    # Extract sample sizes
    sample_sizes = [
        value for key, value in analytics_result.items()
        if "Sample_Size" in key
    ]

    if not sample_sizes:
        return 50.0  # Neutral fallback

    avg_sample_size = np.mean(sample_sizes)

    # Normalize sample size strength
    sample_score = min(avg_sample_size / total_records, 1.0)

    # Stability score from variance of rates
    rates = [
        value for key, value in analytics_result.items()
        if "Rate" in key
    ]

    if rates:
        variance = np.var(rates)
        stability_score = 1 - min(variance, 1)
    else:
        stability_score = 0.5

    confidence = (
        0.5 * sample_score +
        0.3 * stability_score +
        0.2 * 1
    )

    return round(confidence * 100, 2)