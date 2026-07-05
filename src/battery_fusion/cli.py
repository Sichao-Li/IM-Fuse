from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


COMMANDS: dict[str, tuple[str, str]] = {
    "prepare-data": (
        "battery_fusion.data.foundation",
        "Stage mp_total/CIF inputs, labels, splits, and optional modality caches.",
    ),
    "preprocess": (
        "battery_fusion.data.preprocess",
        "Build aligned composition, RDF, and structure feature caches.",
    ),
    "random-split": (
        "battery_fusion.data.splits",
        "Create deterministic random train/val/test split manifests.",
    ),
    "split": ("battery_fusion.experiments.chemistry_splits", "Create chemistry-aware split files."),
    "train": ("battery_fusion.experiments.publication", "Run the publication fusion model matrix."),
    "baseline-classical": ("battery_fusion.experiments.classical_baselines", "Run RF/XGBoost composition baselines."),
    "baseline-alignn": ("battery_fusion.experiments.alignn_pretrained_baseline", "Run pretrained ALIGNN readout + RF baseline."),
    "dropout": ("battery_fusion.experiments.modality_dropout", "Evaluate inference-time modality dropout."),
    "holdout": ("battery_fusion.experiments.anion_holdout", "Train/evaluate leave-one-anion-family-out models."),
    "subgroups": ("battery_fusion.experiments.subgroups", "Compute anion-family and working-ion subgroup metrics."),
    "tables": ("battery_fusion.experiments.final_publication_tables", "Build final publication summary tables."),
    "figures": ("battery_fusion.experiments.cell_reports_figures", "Generate publication B/C/D figures."),
    "parity": ("battery_fusion.experiments.parity_plots", "Generate train/test parity plots."),
    "explain-composition": ("battery_fusion.explain.composition_importance", "Run composition perturbation attributions."),
    "explain-fusion": ("battery_fusion.explain.fusion_importance", "Run multimodal fusion attributions."),
    "explain-permutation": ("battery_fusion.explain.permutation_matrix", "Run permutation-importance matrix experiments."),
    "explain-permutation-single": ("battery_fusion.explain.permutation", "Run one permutation-importance job."),
    "explain-structure": ("battery_fusion.explain.structure_ablation", "Run atom/edge ablation for structural attribution."),
    "explain-deletion": ("battery_fusion.explain.deletion_curves", "Generate deletion-curve figures from explanation outputs."),
    "explain-faithfulness": ("battery_fusion.explain.faithfulness", "Validate attribution faithfulness with deletion curves."),
    "plot-interpretability": ("battery_fusion.explain.plotting", "Plot retained interpretability summary figures."),
}


def _print_help() -> None:
    print("IM-Fuse command line")
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
    print("  imfuse train --target_col average_voltage --seeds 0 1 2 3 4 ...")
    print("  imfuse dropout --target_name average_voltage --seeds 0 1 2 3 4 ...")
    print("  imfuse figures --output_dir figures/final_publication/cell_reports")


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
    command = args.pop(0)
    if command not in COMMANDS:
        _print_help()
        raise SystemExit(f"\nUnknown imfuse command: {command}")
    return _run_module(command, args)


if __name__ == "__main__":
    raise SystemExit(main())
