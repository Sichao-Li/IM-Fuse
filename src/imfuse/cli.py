from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

from imfuse import __version__


COMMANDS: dict[str, tuple[str, str]] = {
    "check": (
        "imfuse.release_check",
        "Validate the environment and tracked publication data contract.",
    ),
    "prepare-data": (
        "imfuse.data.foundation",
        "Stage mp_total/CIF inputs, labels, splits, and optional modality caches.",
    ),
    "preprocess": (
        "imfuse.data.preprocess",
        "Build aligned composition, RDF, and structure feature caches.",
    ),
    "random-split": (
        "imfuse.data.splits",
        "Create deterministic random train/val/test split manifests.",
    ),
    "split-ood": ("imfuse.experiments.ood_splits", "Create composition-cluster or working-ion OOD splits."),
    "train": ("imfuse.experiments.publication", "Run the publication fusion model matrix."),
    "baseline-classical": ("imfuse.experiments.classical_baselines", "Run RF/XGBoost composition baselines."),
    "baseline-alignn": ("imfuse.experiments.alignn_pretrained_baseline", "Run pretrained ALIGNN readout + RF baseline."),
    "dropout": ("imfuse.experiments.modality_dropout", "Evaluate inference-time modality dropout."),
    "subgroups": ("imfuse.experiments.subgroups", "Compute anion-family and working-ion subgroup metrics."),
    "tables": ("imfuse.experiments.final_publication_tables", "Build final publication summary tables."),
    "figures": ("imfuse.experiments.publication_figures", "Generate publication figures."),
    "parity": ("imfuse.experiments.parity_plots", "Generate train/test parity plots."),
    "explain-composition": ("imfuse.explain.composition_importance", "Run composition perturbation attributions."),
    "explain-fusion": ("imfuse.explain.fusion_importance", "Run multimodal fusion attributions."),
    "explain-permutation": ("imfuse.explain.permutation_matrix", "Run permutation-importance matrix experiments."),
    "explain-permutation-single": ("imfuse.explain.permutation", "Run one permutation-importance job."),
    "explain-structure": ("imfuse.explain.structure_ablation", "Run atom/edge ablation for structural attribution."),
    "explain-deletion": ("imfuse.explain.deletion_curves", "Generate deletion-curve figures from explanation outputs."),
    "explain-faithfulness": ("imfuse.explain.faithfulness", "Validate attribution faithfulness with deletion curves."),
    "plot-interpretability": ("imfuse.explain.plotting", "Plot retained interpretability summary figures."),
}


def _print_help() -> None:
    print(f"IM-Fuse {__version__}")
    print("")
    print("Usage:")
    print("  imfuse <command> [command options]")
    print("  imfuse --list")
    print("")
    print("Commands:")
    width = max(len(command) for command in COMMANDS)
    for command, (_module, description) in COMMANDS.items():
        print(f"  {command:<{width}}  {description}")
    print("")
    print("Examples:")
    print("  imfuse check")
    print("  imfuse train --target_col average_voltage --seeds 0 1 2 3 4 ...")
    print("  imfuse dropout --target_name average_voltage --seeds 0 1 2 3 4 ...")
    print("  imfuse split-ood composition-cluster --target_col average_voltage ...")
    print("  imfuse figures --output_dir figures/final_publication/main")


def _run_module(command: str, args: Sequence[str]) -> int:
    module_name = COMMANDS[command][0]
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise SystemExit(f"Command {command!r} maps to {module_name}, but that module has no main().")
    previous_argv = sys.argv[:]
    sys.argv = [f"imfuse {command}", *args]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv
    return int(result) if isinstance(result, int) else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if args[0] in {"--list", "list"}:
        for command in COMMANDS:
            print(command)
        return 0
    if args[0] in {"-V", "--version"}:
        print(__version__)
        return 0
    command = args.pop(0)
    if command not in COMMANDS:
        _print_help()
        raise SystemExit(f"\nUnknown imfuse command: {command}")
    return _run_module(command, args)


if __name__ == "__main__":
    raise SystemExit(main())
