import re
from typing import List



def average_scores(scores: List[float]) -> float:
    """
    Calculate the average of a list of scores.
    Args:
        scores (List[float]): A list of numerical scores.
    Returns:
        float: Average score or 0.0 if the list is empty.
    """
    return sum(scores) / len(scores) if scores else 0.0

