from collections.abc import Sequence

VALID_MODALITIES = ("rdf", "structure", "tabular")
MODALITY_ORDER = {name: i for i, name in enumerate(VALID_MODALITIES)}


def normalize_modalities(modalities: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(modalities), key=lambda item: MODALITY_ORDER.get(item, 99)))
    invalid = [item for item in normalized if item not in MODALITY_ORDER]
    if invalid:
        raise ValueError(f"Unsupported modalities: {invalid}")
    if not 1 <= len(normalized) <= 3:
        raise ValueError("Choose one, two, or three modalities")
    return normalized
