#!/usr/bin/env bash
# ==============================================================================
# generate_paper_tables.sh
#
# Three-stage pipeline for paper experiments:
#   1. run_experiments.jl → raw JSON (per-sample results, no aggregation)
#   2. make_stats.jl      → stats JSON (adds rankings, aggregates, global ranking)
#   3. generate_tables.jl → LaTeX tables (reads stats JSON)
#
# Usage:
#   bash experiments/paper/generate_paper_tables.sh [--quick] [--epochs N] [--samples N] [--selection-mode M] [--degree D] [--verbose] [-p N]
#
# Flags:
#   --quick           Reduced epochs/simulations for smoke testing
#   --epochs N        Override epochs (default: 20)
#   --samples N       Override samples per config (default: 30)
#   --selection-mode  MCTS/DAG-MCTS selection mode (default: BestValue)
#   --degree D        Override tree branching degree (default: auto)
#   --verbose         Include per-configuration detailed tables
#   -p N, --procs N   Launch Julia with N additional worker processes (passed as `julia -p N`)
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
PROCS_FLAG=""

# Suite flags
SUITE_FLAGS=""

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
        --paper-optimizer-comparison|--paper)
            SUITE_FLAGS="$SUITE_FLAGS --paper-optimizer-comparison"
            i=$((i+1))
            ;;
        --paper-mcts-branching)
            SUITE_FLAGS="$SUITE_FLAGS --paper-mcts-branching"
            i=$((i+1))
            ;;
        --paper-dag-mcts-branching)
            SUITE_FLAGS="$SUITE_FLAGS --paper-dag-mcts-branching"
            i=$((i+1))
            ;;
        --paper-greedy-descent-branching)
            SUITE_FLAGS="$SUITE_FLAGS --paper-greedy-descent-branching"
            i=$((i+1))
            ;;
        --paper-gradient-descent-branching)
            SUITE_FLAGS="$SUITE_FLAGS --paper-gradient-descent-branching"
            i=$((i+1))
            ;;
        --paper-mcts-number-of-simulations)
            SUITE_FLAGS="$SUITE_FLAGS --paper-mcts-number-of-simulations"
            i=$((i+1))
            ;;
        --paper-dag-mcts-number-of-simulations)
            SUITE_FLAGS="$SUITE_FLAGS --paper-dag-mcts-number-of-simulations"
            i=$((i+1))
            ;;
        --paper-mcts-exploration-constant)
            SUITE_FLAGS="$SUITE_FLAGS --paper-mcts-exploration-constant"
            i=$((i+1))
            ;;
        --paper-dag-mcts-exploration-constant)
            SUITE_FLAGS="$SUITE_FLAGS --paper-dag-mcts-exploration-constant"
            i=$((i+1))
            ;;
        --verbose)
            VERBOSE_FLAG="--verbose"
            i=$((i+1))
            ;;
        -p)
            i=$((i+1))
            PROCS_FLAG="-p ${!i}"
            i=$((i+1))
            ;;
        -p=*)
            PROCS_FLAG="-p ${arg#*=}"
            i=$((i+1))
            ;;
        --procs)
            i=$((i+1))
            PROCS_FLAG="-p ${!i}"
            i=$((i+1))
            ;;
        --procs=*)
            PROCS_FLAG="-p ${arg#*=}"
            i=$((i+1))
            ;;
        *)
            i=$((i+1))
            ;;
    esac
done

# If no suites specified, default to optimizer comparison
if [ -z "$SUITE_FLAGS" ]; then
    SUITE_FLAGS="--paper-optimizer-comparison"
fi

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

step() { echo; echo "==== $* ===="; }
ok()   { echo "  OK: $*"; }
err()  { echo "  ERROR: $*" >&2; exit 1; }

START_TIME=$(date +%s)

# Helper function to run the 3-stage pipeline for one experiment
run_pipeline() {
    local DIR="$1"
    local NAME="$2"
    local RAW_FILE="$3"
    local STATS_FILE="$4"
    local TEX_FILE="$5"

    # Stage 1: Run experiments → raw JSON
    step "[$NAME] Stage 1: Running experiments"
    julia --project="$REPO_ROOT" $PROCS_FLAG \
        "$DIR/run_experiments.jl" \
        --save \
        --output "$RAW_FILE" \
        $SUITE_FLAGS $QUICK_FLAG $EPOCHS_FLAG $SAMPLES_FLAG $SELECTION_MODE_FLAG $DEGREE_FLAG
    ok "Raw results: $DIR/$RAW_FILE"

    local RAW_PATH="$DIR/$RAW_FILE"
    if [ ! -f "$RAW_PATH" ]; then
        err "Expected raw results not found: $RAW_PATH"
    fi

    # Stage 2: Compute statistics → stats JSON
    step "[$NAME] Stage 2: Computing statistics"
    julia --project="$REPO_ROOT" \
        "$SCRIPT_DIR/make_stats.jl" \
        "$RAW_PATH" \
        --output "$DIR/$STATS_FILE"
    ok "Stats: $DIR/$STATS_FILE"

    local STATS_PATH="$DIR/$STATS_FILE"
    if [ ! -f "$STATS_PATH" ]; then
        err "Expected stats file not found: $STATS_PATH"
    fi

    # Stage 3: Generate tables → LaTeX
    step "[$NAME] Stage 3: Generating tables"
    julia --project="$REPO_ROOT" \
        "$DIR/generate_tables.jl" \
        "$STATS_PATH" \
        --output "$TEX_FILE" \
        $VERBOSE_FLAG
    ok "Tables: $DIR/$TEX_FILE"
}

# ----------------------------------------------------------------------------
# Run all experiments through the pipeline
# ----------------------------------------------------------------------------

run_pipeline \
    "$SCRIPT_DIR/absolute_sum_minimization" \
    "absolute_sum" \
    "absolute_sum_results_raw.json" \
    "absolute_sum_results_stats.json" \
    "absolute_sum_tables.tex"

run_pipeline \
    "$SCRIPT_DIR/function_learning" \
    "function_learning" \
    "function_learning_results_raw.json" \
    "function_learning_results_stats.json" \
    "function_learning_tables.tex"

run_pipeline \
    "$SCRIPT_DIR/polynomial_learning" \
    "polynomial_learning" \
    "poly_learning_results_raw.json" \
    "poly_learning_results_stats.json" \
    "polynomial_learning_tables.tex"

run_pipeline \
    "$SCRIPT_DIR/polynomial_solving" \
    "polynomial_solving" \
    "polynomial_solving_results_raw.json" \
    "polynomial_solving_results_stats.json" \
    "polynomial_solving_tables.tex"

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------

ELAPSED=$(( $(date +%s) - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo
echo "======================================================================"
echo "All paper experiments complete and tables regenerated."
echo "  $SCRIPT_DIR/absolute_sum_minimization/absolute_sum_tables.tex"
echo "  $SCRIPT_DIR/function_learning/function_learning_tables.tex"
echo "  $SCRIPT_DIR/polynomial_learning/polynomial_learning_tables.tex"
echo "  $SCRIPT_DIR/polynomial_solving/polynomial_solving_tables.tex"
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "======================================================================"
