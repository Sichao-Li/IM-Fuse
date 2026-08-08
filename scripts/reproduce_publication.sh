#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-${repo_dir}/src}"
"${PYTHON}" -m imfuse.cli check --root "${repo_dir}" --strict-artifacts
exec bash "${script_dir}/run_final_publication_gpu.sh" "$@"
