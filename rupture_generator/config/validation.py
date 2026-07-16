def is_positive(x: float) -> float:
    if x <= 0:
        raise ValueError(f"must be positive, got {x}")
    return x


def is_non_negative(x: float) -> float:
    if x < 0:
        raise ValueError(f"must be non-negative (>= 0), got {x}")
    return x


def is_proportion(x: float | int) -> float:
    if not (0 <= x <= 1):
        raise ValueError(f"must be in [0, 1], got {x}")
    return float(x)
