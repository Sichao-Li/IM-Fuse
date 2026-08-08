from __future__ import annotations

from pathlib import Path

import chemparse
import pandas as pd


ANION_FAMILIES = (
    "oxide",
    "sulfide",
    "halide",
    "phosphate_or_polyanion",
    "other",
)
WORKING_ION_GROUPS = ("Li", "Na", "Mg", "K", "Zn", "Ca", "Al", "other")


def parse_formula_elements(formula: str | float | None) -> set[str]:
    if formula is None or pd.isna(formula):
        return set()
    parsed = chemparse.parse_formula(str(formula))
    return {element for element, amount in parsed.items() if float(amount) > 0}


def assign_anion_family(formula: str | float | None) -> str:
    """Assign an approximate anion family for evaluation.

    The rules are intentionally transparent and conservative. They are meant for
    chemistry-aware split and audit analyses, not for crystallographic prototype
    classification or mechanistic interpretation.
    """

    elements = parse_formula_elements(formula)
    if not elements:
        return "other"

    if elements.intersection({"F", "Cl", "Br", "I"}):
        return "halide"
    if "O" in elements and elements.intersection({"P", "Si", "B"}):
        return "phosphate_or_polyanion"
    if "S" in elements:
        return "sulfide"
    if "O" in elements:
        return "oxide"
    return "other"


def normalize_working_ion(value: str | float | None) -> str:
    if value is None or pd.isna(value):
        return "other"
    normalized = str(value).strip()
    return normalized if normalized in WORKING_ION_GROUPS[:-1] else "other"


def assign_chemistry_groups(
    frame: pd.DataFrame,
    sample_id_col: str = "id_discharge",
    formula_col: str = "formula_discharge",
    target_col: str = "target",
    working_ion_col: str | None = "working_ion",
) -> pd.DataFrame:
    required = {sample_id_col, formula_col, target_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    output = pd.DataFrame(
        {
            "sample_id": frame[sample_id_col].astype(str),
            "formula": frame[formula_col].astype(str),
            "target": frame[target_col],
        }
    )
    if working_ion_col and working_ion_col in frame.columns:
        output["working_ion"] = frame[working_ion_col].map(normalize_working_ion)
    else:
        output["working_ion"] = "other"
    output["anion_family"] = output["formula"].map(assign_anion_family)
    return output[["sample_id", "formula", "working_ion", "anion_family", "target"]]


def load_assignments(path: Path) -> pd.DataFrame:
    assignments = pd.read_csv(path)
    required = {"sample_id", "formula", "working_ion", "anion_family"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    assignments["sample_id"] = assignments["sample_id"].astype(str)
    return assignments
