import re

# Broader and more aggressive patterns
INJECTION_PATTERNS = [
    r"ignore.*instruction",
    r"disregard.*prompt",
    r"reveal.*patient",
    r"show.*dataset",
    r"export.*data",
    r"list.*patient",
    r"give.*raw",
    r"override",
    r"bypass",
    r"pretend.*allowed",
]


def detect_prompt_injection(user_input: str) -> bool:
    lower_input = user_input.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower_input):
            return True

    return False


def validate_user_query(user_input: str):
    if detect_prompt_injection(user_input):
        raise ValueError(
            "⚠️ Query blocked due to security policy violation."
        )

    return True