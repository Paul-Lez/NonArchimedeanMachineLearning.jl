#!/usr/bin/env bash
# ==============================================================================
# run_and_deploy_tables.sh
#
# Runs all paper experiments, regenerates LaTeX tables, and copies them to
# the arXiv draft directory.
#
# Usage (from repo root):
#   bash experiments/paper/run_and_deploy_tables.sh [--quick] [--epochs N] [--samples N] [--selection-mode M] [--degree D]
#
# Flags are forwarded to generate_paper_tables.sh:
#   --quick           Use reduced epochs/simulations for a fast smoke-test run
#   --epochs N        Override number of epochs (default: 20)
#   --samples N       Override number of samples per config (default: 30)
#   --selection-mode  MCTS/DAG-MCTS selection mode: BestValue, VisitCount, or BestLoss (default: BestValue)
#   --degree D        Override tree branching degree for MCTS/DAG-MCTS/DOO optimizers (default: auto from dims)
#   --verbose         Include per-configuration detailed tables (default: aggregate only)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PAPER_DIR="$HOME/Documents/65a7dd3183827d4e7b6d0f36/arXiv_draft/tables"

step() { echo; echo "==== $* ===="; }
ok()   { echo "  OK: $*"; }
err()  { echo "  ERROR: $*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# Parse flags (intercept --move-only; forward the rest)
# ----------------------------------------------------------------------------

MOVE_ONLY=false
FORWARD_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --move-only) MOVE_ONLY=true ;;
        *)           FORWARD_ARGS+=("$arg") ;;
    esac
done

# ----------------------------------------------------------------------------
# Run experiments and generate tables (unless --move-only)
# ----------------------------------------------------------------------------

if [ "$MOVE_ONLY" = false ]; then
    step "Running experiments and generating tables"
    bash "$SCRIPT_DIR/generate_paper_tables.sh" "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"
else
    step "Skipping experiments (--move-only)"
fi

# ----------------------------------------------------------------------------
# Copy tables to paper directory
# ----------------------------------------------------------------------------

step "Copying tables to arXiv draft"

if [ ! -d "$PAPER_DIR" ]; then
    err "Paper directory not found: $PAPER_DIR"
fi

cp "$SCRIPT_DIR/absolute_sum_minimization/absolute_sum_tables.tex" \
   "$PAPER_DIR/absolute_sum_tables.tex"
ok "absolute_sum_tables.tex"

cp "$SCRIPT_DIR/function_learning/function_learning_tables.tex" \
   "$PAPER_DIR/function_learning_tables.tex"
ok "function_learning_tables.tex"

cp "$SCRIPT_DIR/polynomial_learning/polynomial_learning_tables.tex" \
   "$PAPER_DIR/polynomial_learning_tables.tex"
ok "polynomial_learning_tables.tex"

cp "$SCRIPT_DIR/polynomial_solving/polynomial_solving_tables.tex" \
   "$PAPER_DIR/polynomial_solving_tables.tex"
ok "polynomial_solving_tables.tex"

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------

echo
echo "======================================================================"
echo "Tables deployed to: $PAPER_DIR/"
echo "  absolute_sum_tables.tex"
echo "  function_learning_tables.tex"
echo "  polynomial_learning_tables.tex"
echo "  polynomial_solving_tables.tex"
echo "======================================================================"
