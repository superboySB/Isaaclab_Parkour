#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Prefer IsaacLab's bundled Python (via isaaclab.sh) since the system python may not have pip.
# - Provide IsaacLab root via ISAACLAB_PATH, e.g.:
#     ISAACLAB_PATH=/workspace/isaaclab ./install.sh
# - Or directly provide the launcher via ISAACLAB_SH.
ISAACLAB_PATH="${ISAACLAB_PATH:-}"
ISAACLAB_SH="${ISAACLAB_SH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  ./install.sh install    # default
  ./install.sh uninstall

Environment:
  ISAACLAB_PATH=/path/to/IsaacLab   # prefers {ISAACLAB_PATH}/isaaclab.sh -p -m pip
  ISAACLAB_SH=/path/to/isaaclab.sh  # direct path to isaaclab.sh (overrides ISAACLAB_PATH)
  PYTHON_BIN=python3.10             # fallback python executable (if pip is available)
EOF
}

resolve_pip_runner() {
  if [[ -n "${ISAACLAB_SH}" ]]; then
    if [[ -f "${ISAACLAB_SH}" ]]; then
      echo "${ISAACLAB_SH}"
      return 0
    fi
    echo "[ERROR] ISAACLAB_SH is set but does not exist: ${ISAACLAB_SH}" >&2
    return 2
  fi

  if [[ -n "${ISAACLAB_PATH}" ]]; then
    local candidate="${ISAACLAB_PATH%/}/isaaclab.sh"
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
    echo "[ERROR] ISAACLAB_PATH is set but isaaclab.sh not found at: ${candidate}" >&2
    return 2
  fi

  if command -v isaaclab.sh >/dev/null 2>&1; then
    echo "isaaclab.sh"
    return 0
  fi

  echo ""
  return 0
}

pip() {
  local isaaclab_sh
  isaaclab_sh="$(resolve_pip_runner)"
  if [[ -n "${isaaclab_sh}" ]]; then
    bash "${isaaclab_sh}" -p -m pip "$@"
  else
    "${PYTHON_BIN}" -m pip "$@"
  fi
}

do_install() {
  pip install -e .
  pip install -e parkour_tasks
}

do_uninstall() {
  pip uninstall -y Isaaclab_Parkour parkour_tasks || true
  rm -rf Isaaclab_Parkour.egg-info
  rm -rf parkour_tasks/*.egg-info
}

cmd="${1:-install}"
case "${cmd}" in
  install)
    do_install
    ;;
  uninstall|remove)
    do_uninstall
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[ERROR] Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
