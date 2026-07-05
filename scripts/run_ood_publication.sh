#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-mps}"
OOD_SEEDS="${OOD_SEEDS:-0}"
TARGETS="${TARGETS:-average_voltage capacity_vol}"
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed/publication}"
RAW_DATA="${RAW_DATA:-data/raw/mp_total.csv}"
CIF_DIR="${CIF_DIR:-data/raw/cifs}"
SPLIT_ROOT="${SPLIT_ROOT:-data/splits/publication_ood}"
RESULTS_ROOT="${RESULTS_ROOT:-results/final_publication_ood}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-results/predictions}"
LOG_DIR="${LOG_DIR:-logs/publication_ood}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

N_CLUSTERS="${N_CLUSTERS:-3}"
CLUSTER_SEED="${CLUSTER_SEED:-0}"
CLUSTERS="${CLUSTERS:-0 1 2}"
MIN_TEST_SIZE="${MIN_TEST_SIZE:-50}"
VAL_RATIO="${VAL_RATIO:-0.1}"
WORKING_ION_HOLDOUT_SPECS="${WORKING_ION_HOLDOUT_SPECS:-Na;Mg Ca Zn}"

MODELS="${MODELS:-unimodal_rdf_sequence unimodal_tabular unimodal_structure late_dual_rdf_tabular late_dual_rdf_structure late_dual_tabular_structure early_tri_rdf_tabular_structure mid_tri_rdf_tabular_structure late_tri_rdf_tabular_structure}"
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

RUN_COMPOSITION_CLUSTER="${RUN_COMPOSITION_CLUSTER:-1}"
RUN_WORKING_ION="${RUN_WORKING_ION:-1}"
RUN_NEURAL="${RUN_NEURAL:-1}"
RUN_CLASSICAL="${RUN_CLASSICAL:-1}"
RUN_ALIGNN_PRETRAINED="${RUN_ALIGNN_PRETRAINED:-1}"
INCLUDE_XGBOOST="${INCLUDE_XGBOOST:-1}"
ALIGNN_PYTHON="${ALIGNN_PYTHON:-.venv-alignn/bin/python}"
ALIGNN_PRETRAINED_MODELS="${ALIGNN_PRETRAINED_MODELS:-mp_e_form_alignn}"
ALIGNN_PRETRAINED_FEATURE_MODE="${ALIGNN_PRETRAINED_FEATURE_MODE:-readout}"
ALIGNN_PRETRAINED_FEATURE_CACHE_DIR="${ALIGNN_PRETRAINED_FEATURE_CACHE_DIR:-results/final_publication/alignn_pretrained_features}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"

export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR
IMFUSE=("${PYTHON}" -m battery_fusion.cli)

read -r -a SEED_ARGS <<< "${OOD_SEEDS}"
read -r -a TARGET_ARGS <<< "${TARGETS}"
read -r -a CLUSTER_ARGS <<< "${CLUSTERS}"
read -r -a MODEL_ARGS <<< "${MODELS}"
read -r -a ALIGNN_MODEL_ARGS <<< "${ALIGNN_PRETRAINED_MODELS}"
IFS=';' read -r -a ION_HOLDOUT_SPECS <<< "${WORKING_ION_HOLDOUT_SPECS}"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}" "${SPLIT_ROOT}"

split_is_complete() {
  local split_dir="$1"
  local seed
  for seed in "${SEED_ARGS[@]}"; do
    if [[ ! -f "${split_dir}/seed_${seed}/train.csv" ]] || \
       [[ ! -f "${split_dir}/seed_${seed}/val.csv" ]] || \
       [[ ! -f "${split_dir}/seed_${seed}/test.csv" ]]; then
      return 1
    fi
  done
  return 0
}

run_model_family() {
  local target="$1"
  local split_dir="$2"
  local output_dir="$3"
  local experiment_name="$4"

  if ! split_is_complete "${split_dir}"; then
    echo "Skipping ${experiment_name}: incomplete split directory ${split_dir}" >&2
    return 0
  fi

  mkdir -p "${output_dir}"

  if [[ "${RUN_NEURAL}" == "1" ]]; then
    "${IMFUSE[@]}" train \
      --processed_root "${PROCESSED_ROOT}" \
      --raw_data "${RAW_DATA}" \
      --target_col "${target}" \
      --split_dir "${split_dir}" \
      --output_dir "${output_dir}/neural" \
      --assignment_output "${output_dir}/neural/anion_family_assignments.csv" \
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
      --target_transform none \
      --experiment_name "${experiment_name}_neural" \
      --predictions_root "${PREDICTIONS_ROOT}" \
      --skip_split_creation \
      ${OVERWRITE_FLAG} \
      2>&1 | tee "${LOG_DIR}/${target}_${experiment_name}_neural.log"
  fi

  if [[ "${RUN_CLASSICAL}" == "1" ]]; then
    classical_args=()
    if [[ "${INCLUDE_XGBOOST}" == "1" ]]; then
      classical_args+=(--include_xgboost)
    fi
    "${IMFUSE[@]}" baseline-classical \
      --target_col "${target}" \
      --split_dir "${split_dir}" \
      --output_dir "${output_dir}/classical_baselines" \
      --experiment_name "${experiment_name}_classical" \
      --predictions_root "${PREDICTIONS_ROOT}" \
      --seeds "${SEED_ARGS[@]}" \
      --n_estimators 500 \
      --n_jobs -1 \
      --vocabulary_csv "${RAW_DATA}" \
      --vocabulary_formula_col formula_discharge \
      ${classical_args[@]+"${classical_args[@]}"} \
      ${OVERWRITE_FLAG} \
      2>&1 | tee "${LOG_DIR}/${target}_${experiment_name}_classical.log"
  fi

  if [[ "${RUN_ALIGNN_PRETRAINED}" == "1" ]]; then
    "${IMFUSE[@]}" baseline-alignn \
      --target_col "${target}" \
      --split_dir "${split_dir}" \
      --cif_dir "${CIF_DIR}" \
      --output_dir "${output_dir}/alignn_pretrained_rf" \
      --experiment_name "${experiment_name}_alignn_pretrained" \
      --predictions_root "${PREDICTIONS_ROOT}" \
      --alignn_python "${ALIGNN_PYTHON}" \
      --pretrained_models "${ALIGNN_MODEL_ARGS[@]}" \
      --feature_mode "${ALIGNN_PRETRAINED_FEATURE_MODE}" \
      --feature_cache_dir "${ALIGNN_PRETRAINED_FEATURE_CACHE_DIR}" \
      --seeds "${SEED_ARGS[@]}" \
      --n_estimators 500 \
      --n_jobs -1 \
      ${OVERWRITE_FLAG} \
      2>&1 | tee "${LOG_DIR}/${target}_${experiment_name}_alignn_pretrained.log"
  fi
}

for target in "${TARGET_ARGS[@]}"; do
  if [[ "${RUN_COMPOSITION_CLUSTER}" == "1" ]]; then
    cluster_root="${SPLIT_ROOT}/composition_cluster_holdout/${target}/k_${N_CLUSTERS}"
    "${IMFUSE[@]}" split-ood composition-cluster \
      --input_data "${RAW_DATA}" \
      --sample_id_col id_discharge \
      --formula_col formula_discharge \
      --target_col "${target}" \
      --working_ion_col working_ion \
      --output_dir "${cluster_root}" \
      --seeds "${SEED_ARGS[@]}" \
      --n_clusters "${N_CLUSTERS}" \
      --cluster_seed "${CLUSTER_SEED}" \
      --min_test_size "${MIN_TEST_SIZE}" \
      --val_ratio "${VAL_RATIO}" \
      ${OVERWRITE_FLAG} \
      2>&1 | tee "${LOG_DIR}/${target}_create_composition_cluster_splits.log"

    for cluster_id in "${CLUSTER_ARGS[@]}"; do
      split_dir="${cluster_root}/cluster_${cluster_id}"
      output_dir="${RESULTS_ROOT}/${target}/composition_cluster_holdout/k_${N_CLUSTERS}/cluster_${cluster_id}"
      experiment_name="publication_ood_composition_cluster_k${N_CLUSTERS}_cluster_${cluster_id}"
      run_model_family "${target}" "${split_dir}" "${output_dir}" "${experiment_name}"
    done
  fi

  if [[ "${RUN_WORKING_ION}" == "1" ]]; then
    for ion_spec in "${ION_HOLDOUT_SPECS[@]}"; do
      read -r -a HELDOUT_ION_ARGS <<< "${ion_spec}"
      holdout_name="$(printf '%s_' "${HELDOUT_ION_ARGS[@]}")"
      holdout_name="${holdout_name%_}"
      split_dir="${SPLIT_ROOT}/working_ion_holdout/${target}/${holdout_name}"
      "${IMFUSE[@]}" split-ood working-ion \
        --input_data "${RAW_DATA}" \
        --sample_id_col id_discharge \
        --formula_col formula_discharge \
        --target_col "${target}" \
        --working_ion_col working_ion \
        --heldout_ions "${HELDOUT_ION_ARGS[@]}" \
        --output_dir "${split_dir}" \
        --seeds "${SEED_ARGS[@]}" \
        --min_test_size "${MIN_TEST_SIZE}" \
        --val_ratio "${VAL_RATIO}" \
        ${OVERWRITE_FLAG} \
        2>&1 | tee "${LOG_DIR}/${target}_create_working_ion_${holdout_name}_splits.log"

      output_dir="${RESULTS_ROOT}/${target}/working_ion_holdout/${holdout_name}"
      experiment_name="publication_ood_working_ion_${holdout_name}"
      run_model_family "${target}" "${split_dir}" "${output_dir}" "${experiment_name}"
    done
  fi
done
