#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-mps}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
TARGETS="${TARGETS:-average_voltage capacity_vol}"
MODELS="${MODELS:-unimodal_rdf_sequence unimodal_tabular unimodal_structure late_dual_rdf_tabular late_dual_rdf_structure late_dual_tabular_structure early_tri_rdf_tabular_structure mid_tri_rdf_tabular_structure late_tri_rdf_tabular_structure}"
TARGET_TRANSFORM="${TARGET_TRANSFORM:-none}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
RDF_EPOCHS="${RDF_EPOCHS:-1000}"
TABULAR_EPOCHS="${TABULAR_EPOCHS:-1000}"
STRUCTURE_EPOCHS="${STRUCTURE_EPOCHS:-1000}"
TRI_EPOCHS="${TRI_EPOCHS:-1000}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
RDF_WEIGHT_DECAY="${RDF_WEIGHT_DECAY:-1e-4}"
SCHEDULER_MILESTONE="${SCHEDULER_MILESTONE:-20}"
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed/legacy_rdf_split_seed_42}"
RAW_DATA="${RAW_DATA:-data/raw/mp_total.csv}"
CIF_DIR="${CIF_DIR:-data/raw/cifs}"
RESULTS_ROOT="${RESULTS_ROOT:-results/final_publication}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-results/predictions}"
FIGURES_ROOT="${FIGURES_ROOT:-figures/final_publication}"
LOG_DIR="${LOG_DIR:-logs/final_publication}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
RUN_RANDOM="${RUN_RANDOM:-1}"
RUN_CLASSICAL="${RUN_CLASSICAL:-1}"
RUN_ALIGNN_PRETRAINED="${RUN_ALIGNN_PRETRAINED:-1}"
RUN_EXPERIMENT_B="${RUN_EXPERIMENT_B:-1}"
RUN_EXPERIMENT_C="${RUN_EXPERIMENT_C:-1}"
RUN_EXPERIMENT_D="${RUN_EXPERIMENT_D:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
RUN_SUMMARIES="${RUN_SUMMARIES:-1}"
INCLUDE_XGBOOST="${INCLUDE_XGBOOST:-1}"
ALIGNN_PYTHON="${ALIGNN_PYTHON:-}"
ALIGNN_FILE_FORMAT="${ALIGNN_FILE_FORMAT:-poscar}"
ALIGNN_PRETRAINED_MODELS="${ALIGNN_PRETRAINED_MODELS:-mp_e_form_alignn}"
ALIGNN_PRETRAINED_FEATURE_MODE="${ALIGNN_PRETRAINED_FEATURE_MODE:-readout}"
ALIGNN_PRETRAINED_FEATURE_CACHE_DIR="${ALIGNN_PRETRAINED_FEATURE_CACHE_DIR:-results/final_publication/alignn_pretrained_features}"
DROPOUT_MODEL_NAME="${DROPOUT_MODEL_NAME:-mid_tri_rdf_tabular_structure}"
DROPOUT_OUTPUT_DIR_NAME="${DROPOUT_OUTPUT_DIR_NAME:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
OOD_SEEDS="${OOD_SEEDS:-0}"
OOD_RESULTS_ROOT="${OOD_RESULTS_ROOT:-results/final_publication_ood}"
OOD_SPLIT_ROOT="${OOD_SPLIT_ROOT:-data/splits/publication_ood}"
OOD_LOG_DIR="${OOD_LOG_DIR:-logs/publication_ood}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR
IMFUSE=("${PYTHON}" -m battery_fusion.cli)

if [[ "${DEVICE}" == "cuda" ]]; then
  "${PYTHON}" - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
    print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
else:
    sys.exit("CUDA is not available. Refusing to run final publication jobs on CPU.")
PY
fi

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a TARGET_ARGS <<< "${TARGETS}"
read -r -a MODEL_ARGS <<< "${MODELS}"
read -r -a ALIGNN_PRETRAINED_MODEL_ARGS <<< "${ALIGNN_PRETRAINED_MODELS}"

if [[ -z "${DROPOUT_OUTPUT_DIR_NAME}" ]]; then
  if [[ "${DROPOUT_MODEL_NAME}" == "mid_tri_rdf_tabular_structure" ]]; then
    DROPOUT_OUTPUT_DIR_NAME="modality_dropout_mid_tri"
  else
    DROPOUT_OUTPUT_DIR_NAME="modality_dropout_${DROPOUT_MODEL_NAME}"
  fi
fi

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}" "${FIGURES_ROOT}"

optional_limit_args=()
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  optional_limit_args+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
  optional_limit_args+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
fi

for target in "${TARGET_ARGS[@]}"; do
  random_dir="${RESULTS_ROOT}/${target}/random_split"
  split_dir="data/splits/publication/${target}"
  assignment_path="${random_dir}/anion_family_assignments.csv"
  mkdir -p "${random_dir}"

  if [[ "${RUN_RANDOM}" == "1" ]]; then
    "${IMFUSE[@]}" train \
      --processed_root "${PROCESSED_ROOT}" \
      --raw_data "${RAW_DATA}" \
      --target_col "${target}" \
      --split_dir "${split_dir}" \
      --output_dir "${random_dir}" \
      --assignment_output "${assignment_path}" \
      --seeds "${SEED_ARGS[@]}" \
      --epochs "${MAX_EPOCHS}" \
      --rdf_epochs "${RDF_EPOCHS}" \
      --tabular_epochs "${TABULAR_EPOCHS}" \
      --structure_epochs "${STRUCTURE_EPOCHS}" \
      --tri_epochs "${TRI_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --learning_rate "${LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --rdf_weight_decay "${RDF_WEIGHT_DECAY}" \
      --scheduler_milestone "${SCHEDULER_MILESTONE}" \
      --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
      --device "${DEVICE}" \
      --models "${MODEL_ARGS[@]}" \
      --target_transform "${TARGET_TRANSFORM}" \
      --experiment_name final_publication_random \
      --predictions_root "${PREDICTIONS_ROOT}" \
      --skip_split_creation \
      ${optional_limit_args[@]+"${optional_limit_args[@]}"} \
      ${OVERWRITE_FLAG} \
      2>&1 | tee "${LOG_DIR}/${target}_random_split.log"

  fi

  if [[ "${RUN_CLASSICAL}" == "1" ]]; then
    classical_args=()
    if [[ "${INCLUDE_XGBOOST}" == "1" ]]; then
      classical_args+=(--include_xgboost)
    fi
    "${IMFUSE[@]}" baseline-classical \
      --split_dir "${split_dir}" \
      --output_dir "${RESULTS_ROOT}/${target}/classical_baselines" \
      --target_col "${target}" \
      --experiment_name final_publication_classical_random \
      --predictions_root "${PREDICTIONS_ROOT}" \
      --seeds "${SEED_ARGS[@]}" \
      --n_estimators 500 \
      --n_jobs -1 \
      --vocabulary_csv "${RAW_DATA}" \
      --vocabulary_formula_col formula_discharge \
      ${classical_args[@]+"${classical_args[@]}"} \
      ${OVERWRITE_FLAG} \
      2>&1 | tee "${LOG_DIR}/${target}_classical_baselines.log"
  fi

  if [[ "${RUN_ALIGNN_PRETRAINED}" == "1" ]]; then
    "${IMFUSE[@]}" baseline-alignn \
      --split_dir "${split_dir}" \
      --cif_dir "${CIF_DIR}" \
      --output_dir "${RESULTS_ROOT}/${target}/alignn_pretrained_rf" \
      --target_col "${target}" \
      --seeds "${SEED_ARGS[@]}" \
      --pretrained_models "${ALIGNN_PRETRAINED_MODEL_ARGS[@]}" \
      --feature_mode "${ALIGNN_PRETRAINED_FEATURE_MODE}" \
      --feature_cache_dir "${ALIGNN_PRETRAINED_FEATURE_CACHE_DIR}" \
      --experiment_name final_publication_alignn_pretrained \
      --predictions_root "${PREDICTIONS_ROOT}" \
      --alignn_python "${ALIGNN_PYTHON:-.venv-alignn/bin/python}" \
      --file_format "${ALIGNN_FILE_FORMAT}" \
      --n_estimators 500 \
      --n_jobs -1 \
      ${OVERWRITE_FLAG} \
      2>&1 | tee "${LOG_DIR}/${target}_alignn_pretrained_rf.log"
  fi

  if [[ "${RUN_EXPERIMENT_B}" == "1" ]]; then
    "${IMFUSE[@]}" dropout \
      --target_name "${target}" \
      --processed_root "${PROCESSED_ROOT}" \
      --checkpoint_dir "${random_dir}/checkpoints/${DROPOUT_MODEL_NAME}" \
      --split_dir "${split_dir}" \
      --output_dir "${RESULTS_ROOT}/${target}/${DROPOUT_OUTPUT_DIR_NAME}" \
      --model_name "${DROPOUT_MODEL_NAME}" \
      --metadata "${assignment_path}" \
      --predictions_root "${PREDICTIONS_ROOT}" \
      --experiment_name final_publication_modality_dropout \
      --device "${DEVICE}" \
      --seeds "${SEED_ARGS[@]}" \
      ${OVERWRITE_FLAG} \
      2>&1 | tee "${LOG_DIR}/${target}_experiment_b_${DROPOUT_MODEL_NAME}_modality_dropout.log"

  fi

  if [[ "${RUN_EXPERIMENT_D}" == "1" ]]; then
    "${IMFUSE[@]}" subgroups \
      --predictions_dir "${PREDICTIONS_ROOT}/final_publication_random/${target}" \
      --metadata "${assignment_path}" \
      --target_col "${target}" \
      --output_dir "${RESULTS_ROOT}/${target}/subgroup_analysis" \
      --min_group_size 30 \
      --split test \
      ${OVERWRITE_FLAG} \
      2>&1 | tee "${LOG_DIR}/${target}_experiment_d_subgroups.log"

    if [[ "${RUN_CLASSICAL}" == "1" ]]; then
      "${IMFUSE[@]}" subgroups \
        --predictions_dir "${PREDICTIONS_ROOT}/final_publication_classical_random/${target}" \
        --metadata "${assignment_path}" \
        --target_col "${target}" \
        --output_dir "${RESULTS_ROOT}/${target}/classical_subgroup_analysis" \
        --min_group_size 30 \
        --split test \
        ${OVERWRITE_FLAG} \
        2>&1 | tee "${LOG_DIR}/${target}_experiment_d_classical_subgroups.log"
    fi

    if [[ "${RUN_ALIGNN_PRETRAINED}" == "1" ]]; then
      "${IMFUSE[@]}" subgroups \
        --predictions_dir "${PREDICTIONS_ROOT}/final_publication_alignn_pretrained/${target}" \
        --metadata "${assignment_path}" \
        --target_col "${target}" \
        --output_dir "${RESULTS_ROOT}/${target}/alignn_pretrained_subgroup_analysis" \
        --min_group_size 30 \
        --split test \
        ${OVERWRITE_FLAG} \
        2>&1 | tee "${LOG_DIR}/${target}_experiment_d_alignn_pretrained_subgroups.log"
    fi

  fi
done

if [[ "${RUN_EXPERIMENT_C}" == "1" ]]; then
  TARGETS="${TARGETS}" \
  OOD_SEEDS="${OOD_SEEDS}" \
  DEVICE="${DEVICE}" \
  PYTHON="${PYTHON}" \
  PROCESSED_ROOT="${PROCESSED_ROOT}" \
  RAW_DATA="${RAW_DATA}" \
  CIF_DIR="${CIF_DIR}" \
  SPLIT_ROOT="${OOD_SPLIT_ROOT}" \
  RESULTS_ROOT="${OOD_RESULTS_ROOT}" \
  PREDICTIONS_ROOT="${PREDICTIONS_ROOT}" \
  LOG_DIR="${OOD_LOG_DIR}" \
  MODELS="${MODELS}" \
  MAX_EPOCHS="${MAX_EPOCHS}" \
  RDF_EPOCHS="${RDF_EPOCHS}" \
  TABULAR_EPOCHS="${TABULAR_EPOCHS}" \
  STRUCTURE_EPOCHS="${STRUCTURE_EPOCHS}" \
  TRI_EPOCHS="${TRI_EPOCHS}" \
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  LEARNING_RATE="${LEARNING_RATE}" \
  WEIGHT_DECAY="${WEIGHT_DECAY}" \
  RDF_WEIGHT_DECAY="${RDF_WEIGHT_DECAY}" \
  SCHEDULER_MILESTONE="${SCHEDULER_MILESTONE}" \
  RUN_NEURAL="${RUN_RANDOM}" \
  RUN_CLASSICAL="${RUN_CLASSICAL}" \
  RUN_ALIGNN_PRETRAINED="${RUN_ALIGNN_PRETRAINED}" \
  INCLUDE_XGBOOST="${INCLUDE_XGBOOST}" \
  ALIGNN_PYTHON="${ALIGNN_PYTHON:-.venv-alignn/bin/python}" \
  ALIGNN_PRETRAINED_MODELS="${ALIGNN_PRETRAINED_MODELS}" \
  ALIGNN_PRETRAINED_FEATURE_MODE="${ALIGNN_PRETRAINED_FEATURE_MODE}" \
  ALIGNN_PRETRAINED_FEATURE_CACHE_DIR="${ALIGNN_PRETRAINED_FEATURE_CACHE_DIR}" \
  OVERWRITE_FLAG="${OVERWRITE_FLAG}" \
  bash scripts/run_ood_publication.sh
fi

if [[ "${RUN_SUMMARIES}" == "1" ]]; then
  "${IMFUSE[@]}" tables \
    --results_root "${RESULTS_ROOT}" \
    --ood_results_root "${OOD_RESULTS_ROOT}" \
    --output_dir "${RESULTS_ROOT}" \
    ${OVERWRITE_FLAG} \
    2>&1 | tee "${LOG_DIR}/final_publication_summary_tables.log"
fi

if [[ "${RUN_PLOTS}" == "1" ]]; then
  "${IMFUSE[@]}" figures \
    --results_root "${RESULTS_ROOT}" \
    --output_dir "${FIGURES_ROOT}/cell_reports" \
    --data_output_dir "${RESULTS_ROOT}/cell_reports_figure_data" \
    ${OVERWRITE_FLAG} \
    2>&1 | tee "${LOG_DIR}/cell_reports_figures.log"

  "${IMFUSE[@]}" parity \
    --predictions_root "${PREDICTIONS_ROOT}" \
    --output_dir "${FIGURES_ROOT}/parity_plots" \
    --summary_output "${RESULTS_ROOT}/parity_plot_summary.csv" \
    --splits train test \
    ${OVERWRITE_FLAG} \
    2>&1 | tee "${LOG_DIR}/publication_parity_plots.log"
fi
