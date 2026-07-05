from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized paths for one repository checkout."""

    root: Path = Path(".")

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def raw_mp_total(self) -> Path:
        return self.raw_dir / "mp_total.csv"

    @property
    def raw_cif_dir(self) -> Path:
        return self.raw_dir / "cifs"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    def processed_split_dir(self, split_name: str) -> Path:
        return self.processed_dir / split_name

    def processed_modality_dir(
        self, split_name: str, modality: str, split: str
    ) -> Path:
        return self.processed_split_dir(split_name) / modality / split
