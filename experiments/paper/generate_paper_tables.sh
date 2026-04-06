#!/usr/bin/env bash
# ==============================================================================
# generate_paper_tables.sh
#
# Runs all paper experiments and regenerates the corresponding LaTeX tables.
#
# Usage:
#   ./experiments/paper/generate_paper_tables.sh [--quick] [--epochs N] [--samples N] [--selection-mode M] [--degree D]
#
# Flags:
#   --quick           Use reduced epochs/simulations for a fast smoke-test run
#   --epochs N        Override number of epochs (default: 20)
#   --samples N       Override number of samples per config (default: 30)
#   --selection-mode  MCTS/DAG-MCTS selection mode: BestValue, VisitCount, or BestLoss (default: BestValue)
#   --degree D        Override tree branching degree for MCTS/DAG-MCTS/DOO optimizers (default: auto from dims)
#   --verbose         Include per-configuration detailed tables (default: aggregate only)
#
# The script must be run from the repository root, e.g.:
#   bash experiments/paper/generate_paper_tables.sh
# ==============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Parse flags
# ----------------------------------------------------------------------------
QUICK_FLAG=""
EPOCHS_FLAG=""
SAMPLES_FLAG="--samples 30"
SELECTION_MODE_FLAG=""
DEGREE_FLAG=""
VERBOSE_FLAG=""

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --quick)
            QUICK_FLAG="--quick"
            i=$((i+1))
            ;;
        --epochs)
            i=$((i+1))
            EPOCHS_FLAG="--epochs ${!i}"
            i=$((i+1))
            ;;
        --epochs=*)
            EPOCHS_FLAG="--epochs ${arg#*=}"
            i=$((i+1))
            ;;
        --samples)
            i=$((i+1))
            SAMPLES_FLAG="--samples ${!i}"
            i=$((i+1))
            ;;
        --samples=*)
            SAMPLES_FLAG="--samples ${arg#*=}"
            i=$((i+1))
            ;;
        --selection-mode)
            i=$((i+1))
            SELECTION_MODE_FLAG="--selection-mode ${!i}"
            i=$((i+1))
            ;;
        --selection-mode=*)
            SELECTION_MODE_FLAG="--selection-mode ${arg#*=}"
            i=$((i+1))
            ;;
        --degree)
            i=$((i+1))
            DEGREE_FLAG="--degree ${!i}"
            i=$((i+1))
            ;;
        --degree=*)
            DEGREE_FLAG="--degree ${arg#*=}"
            i=$((i+1))
            ;;
        --verbose)
            VERBOSE_FLAG="--verbose"
            i=$((i+1))
            ;;
        *)
            i=$((i+1))
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

step() { echo; echo "==== $* ===="; }
ok()   { echo "  OK: $*"; }
err()  { echo "  ERROR: $*" >&2; exit 1; }

START_TIME=$(date +%s)

# ----------------------------------------------------------------------------
# Absolute sum minimization
# ----------------------------------------------------------------------------

ABSSUM_DIR="$SCRIPT_DIR/absolute_sum_minimization"
ABSSUM_RESULTS="$ABSSUM_DIR/absolute_sum_results_paper.json"

step "Running absolute_sum_minimization experiments"
julia --project="$REPO_ROOT" \
    "$ABSSUM_DIR/run_experiments.jl" \
    --paper --save \
    --output absolute_sum_results_paper.json \
    $QUICK_FLAG $EPOCHS_FLAG $SAMPLES_FLAG $SELECTION_MODE_FLAG $DEGREE_FLAG
ok "Experiments done"

step "Generating absolute_sum_minimization tables"
if [ ! -f "$ABSSUM_RESULTS" ]; then
    err "Expected results file not found: $ABSSUM_RESULTS"
fi
julia --project="$REPO_ROOT" \
    "$ABSSUM_DIR/generate_tables.jl" \
    "$ABSSUM_RESULTS" \
    --output absolute_sum_tables.tex \
    $VERBOSE_FLAG
ok "Tables written to $ABSSUM_DIR/absolute_sum_tables.tex"

# ----------------------------------------------------------------------------
# Function learning
# ----------------------------------------------------------------------------

FUNCLEARN_DIR="$SCRIPT_DIR/function_learning"
FUNCLEARN_RESULTS="$FUNCLEARN_DIR/function_learning_results_paper.json"

step "Running function_learning experiments"
julia --project="$REPO_ROOT" \
    "$FUNCLEARN_DIR/run_experiments.jl" \
    --paper --save \
    --output function_learning_results_paper.json \
    $QUICK_FLAG $EPOCHS_FLAG $SAMPLES_FLAG $SELECTION_MODE_FLAG $DEGREE_FLAG
ok "Experiments done"

step "Generating function_learning tables"
if [ ! -f "$FUNCLEARN_RESULTS" ]; then
    err "Expected results file not found: $FUNCLEARN_RESULTS"
fi
julia --project="$REPO_ROOT" \
    "$FUNCLEARN_DIR/generate_tables.jl" \
    "$FUNCLEARN_RESULTS" \
    --output function_learning_tables.tex \
    $VERBOSE_FLAG
ok "Tables written to $FUNCLEARN_DIR/function_learning_tables.tex"

# ----------------------------------------------------------------------------
# Polynomial learning
# ----------------------------------------------------------------------------

POLYLEARN_DIR="$SCRIPT_DIR/polynomial_learning"
POLYLEARN_RESULTS="$POLYLEARN_DIR/poly_learning_results_paper.json"

step "Running polynomial_learning experiments"
julia --project="$REPO_ROOT" \
    "$POLYLEARN_DIR/run_experiments.jl" \
    --paper --save \
    --output poly_learning_results_paper.json \
    $QUICK_FLAG $EPOCHS_FLAG $SAMPLES_FLAG $SELECTION_MODE_FLAG $DEGREE_FLAG
ok "Experiments done"

step "Generating polynomial_learning tables"
if [ ! -f "$POLYLEARN_RESULTS" ]; then
    err "Expected results file not found: $POLYLEARN_RESULTS"
fi
julia --project="$REPO_ROOT" \
    "$POLYLEARN_DIR/generate_tables.jl" \
    "$POLYLEARN_RESULTS" \
    --output polynomial_learning_tables.tex \
    $VERBOSE_FLAG
ok "Tables written to $POLYLEARN_DIR/polynomial_learning_tables.tex"

# ----------------------------------------------------------------------------
# Polynomial solving
# ----------------------------------------------------------------------------

POLYSOLVE_DIR="$SCRIPT_DIR/polynomial_solving"
POLYSOLVE_RESULTS="$POLYSOLVE_DIR/polynomial_solving_results_paper.json"

step "Running polynomial_solving experiments"
julia --project="$REPO_ROOT" \
    "$POLYSOLVE_DIR/run_experiments.jl" \
    --paper --save \
    --output polynomial_solving_results_paper.json \
    $QUICK_FLAG $EPOCHS_FLAG $SAMPLES_FLAG $SELECTION_MODE_FLAG $DEGREE_FLAG
ok "Experiments done"

step "Generating polynomial_solving tables"
if [ ! -f "$POLYSOLVE_RESULTS" ]; then
    err "Expected results file not found: $POLYSOLVE_RESULTS"
fi
julia --project="$REPO_ROOT" \
    "$POLYSOLVE_DIR/generate_tables.jl" \
    "$POLYSOLVE_RESULTS" \
    --output polynomial_solving_tables.tex \
    $VERBOSE_FLAG
ok "Tables written to $POLYSOLVE_DIR/polynomial_solving_tables.tex"

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------

ELAPSED=$(( $(date +%s) - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo
echo "======================================================================"
echo "All paper experiments complete and tables regenerated."
echo "  $ABSSUM_DIR/absolute_sum_tables.tex"
echo "  $FUNCLEARN_DIR/function_learning_tables.tex"
echo "  $POLYLEARN_DIR/polynomial_learning_tables.tex"
echo "  $POLYSOLVE_DIR/polynomial_solving_tables.tex"
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "======================================================================"
