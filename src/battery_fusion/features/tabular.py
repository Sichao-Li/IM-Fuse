from collections.abc import Sequence

import chemparse
import numpy as np


def vocabulary_from_formulas(formulas: Sequence[str]) -> list[str]:
    elements: set[str] = set()
    for formula in formulas:
        elements.update(chemparse.parse_formula(str(formula)).keys())
    return sorted(elements)


def formula_vector(formula: str, vocabulary: Sequence[str]) -> np.ndarray:
    parsed = chemparse.parse_formula(str(formula))
    return np.array([float(parsed.get(element, 0.0)) for element in vocabulary])
