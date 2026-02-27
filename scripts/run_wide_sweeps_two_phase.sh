#!/usr/bin/env bash
# Two-phase wide sweeps: 3 jobs in parallel (6 cores each), then 2 sweeps + 1 single run.
# Run from project root in WSL: ./scripts/run_wide_sweeps_two_phase.sh
# Or: bash scripts/run_wide_sweeps_two_phase.sh
#
# Phase 1: linear regression sweep, Bayesian Ridge sweep, GPR sweep (3 parallel, each 6 cores).
# Phase 2 (after wait): GMM sweep, logistic regression sweep (2 parallel); then ListMLE standing-rank single run.
#
# Total: 3 workers x 6 threads = 18 cores in phase 1; 2 x 6 in phase 2; then 1 x 6 for single run.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

N_TRIALS="${N_TRIALS:-40}"
BATCH_LINEAR="${BATCH_LINEAR:-team_stats_linear_wide}"
BATCH_BAYES="${BATCH_BAYES:-team_stats_bayesian_ridge_wide}"
BATCH_GPR="${BATCH_GPR:-team_stats_gpr_wide}"
BATCH_GMM="${BATCH_GMM:-team_stats_gmm_wide}"
BATCH_LR="${BATCH_LR:-logistic_regression_wide}"

echo "Phase 1: starting 3 sweeps in parallel (linear, bayesian_ridge, gpr), each 6 cores, n_trials=$N_TRIALS"

python -m scripts.sweep_hparams --config config/team_stats_linear_sweep.yaml --method optuna --objective spearman --n-trials "$N_TRIALS" --n-jobs 1 --listmle-target playoff_outcome --batch-id "$BATCH_LINEAR" &
PID1=$!
python -m scripts.sweep_hparams --config config/team_stats_bayesian_ridge_sweep.yaml --method optuna --objective spearman --n-trials "$N_TRIALS" --n-jobs 1 --listmle-target playoff_outcome --batch-id "$BATCH_BAYES" &
PID2=$!
python -m scripts.sweep_hparams --config config/team_stats_gpr_sweep.yaml --method optuna --objective spearman --n-trials "$N_TRIALS" --n-jobs 1 --listmle-target playoff_outcome --batch-id "$BATCH_GPR" &
PID3=$!

wait $PID1 $PID2 $PID3
echo "Phase 1 done."

echo "Phase 2: starting 2 sweeps in parallel (gmm, logistic_regression), each 6 cores, n_trials=$N_TRIALS"

python -m scripts.sweep_hparams --config config/team_stats_gmm_sweep.yaml --method optuna --objective spearman --n-trials "$N_TRIALS" --n-jobs 1 --listmle-target playoff_outcome --batch-id "$BATCH_GMM" &
PID4=$!
python -m scripts.sweep_hparams --config config/logistic_regression_sweep.yaml --method optuna --objective spearman --n-trials "$N_TRIALS" --n-jobs 1 --listmle-target playoff_outcome --batch-id "$BATCH_LR" &
PID5=$!

wait $PID4 $PID5
echo "Phase 2 sweeps done."

echo "Running ListMLE with standing rank single run (output/11_listmle_standing_rank)."
python -m scripts.run_pipeline_from_model_a --config config/outputs11_sweep_listmle_standing_rank.yaml --outputs output/11_listmle_standing_rank/listmle_standing_rank_single

echo "All done."
