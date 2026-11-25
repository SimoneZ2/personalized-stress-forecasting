#!/bin/bash
# Reproduces all the results of the paper, in order. Run from the repository root:
#   CES_DIR=/path/to/CES/dataset SSAQS_DIR=/path/to/SSAQS bash run_all.sh [step ...]
# Steps: data ces ces_eval ihope phq4 ssaqs (default: all, in this order).
# Outputs go to results/; the tables are printed to the .txt files listed in README.md.
set -eu
export PYTHONHASHSEED=0          # the walk-forward iterates a set of user ids

CES_DIR=${CES_DIR:?set CES_DIR to the CES folder containing EMA/ and Sensing/}
SSAQS_DIR=${SSAQS_DIR:-}
SEEDS="42 43 44"
EMA=$CES_DIR/EMA/general_ema.csv
SENS=$CES_DIR/Sensing/sensing.csv
STRESS_NPZ=ces_stress_forecast.npz
PHQ4_NPZ=ces_phq4_forecast.npz
R=results
STEPS=${*:-data ces ces_eval ihope phq4 ssaqs}
mkdir -p $R/tables $R/figures

for step in $STEPS; do
case $step in

data)   # feature engineering: CES stress and PHQ-4 datasets
  python features/build_forecast_dataset.py --ema $EMA --sens $SENS --out $STRESS_NPZ
  python features/build_forecast_dataset.py --ema $EMA --sens $SENS --target-col phq4_score --out $PHQ4_NPZ
  ;;

ces)    # RQ1-RQ2: LSTM + GBM system, walk-forward and leave-users-out, three seeds (GPU)
  for s in $SEEDS; do
    python -u pipeline/run_walk_forward.py --data $STRESS_NPZ --seed $s --out $R/wf_seed$s
    python -u pipeline/run_leave_users_out.py --data $STRESS_NPZ --seed $s --out $R/luo_seed$s
  done
  ;;

ces_eval)   # Table 1, paired tests, Table 2, figures, seed stability
  WF=$R/wf_seed42/predictions_wf.csv
  LUO=$R/luo_seed42/predictions_luo.csv
  python evaluation/rq1_table.py --wf $WF --luo $LUO --npz $STRESS_NPZ > $R/tables/rq1_table1.txt
  python evaluation/rq1_tests.py --wf-pattern $WF --luo-pattern $LUO --npz $STRESS_NPZ > $R/tables/rq1_tests.txt
  python evaluation/rq2_table.py --wf $WF --luo $LUO --npz $STRESS_NPZ > $R/tables/rq2_table2.txt
  python evaluation/figures.py --wf $WF --luo $LUO --npz $STRESS_NPZ --out-dir $R/figures > $R/tables/figures_values.txt
  python evaluation/seed_stability.py --seeds $SEEDS --npz $STRESS_NPZ --wf-pattern "$R/wf_seed{s}/predictions_wf.csv" \
      --luo-pattern "$R/luo_seed{s}/predictions_luo.csv" --out $R/tables/seed_stability.csv > $R/tables/seed_stability.txt
  ;;

ihope)  # RQ3: I-HOPE reimplementation (CPU, several hours). TensorFlow slows down within one
        # process, so the script stops every 15 users (exit code 3) and is restarted; it resumes
        # from the users already saved. Run inside tmux/screen.
  while true; do
    rc=0
    python -u comparisons/ihope/ihope_regressor.py --sensing $SENS --ema $EMA \
        --rows comparisons/ihope/ihope_rows.csv --out $R/ihope --max-users 15 || rc=$?
    [ $rc -eq 3 ] || break
  done
  [ $rc -eq 0 ]
  python evaluation/rq3_ihope.py --ihope $R/ihope/ihope_predictions.csv --wf $R/wf_seed42/predictions_wf.csv \
      --npz $STRESS_NPZ > $R/tables/rq3_ihope.txt
  ;;

phq4)   # RQ3: PHQ-4 severity, comparison with Gala et al. (120 users, seed 42)
  for c in sparse weekly; do
    stats=dense; [ $c = sparse ] && stats=sparse
    python -u pipeline/run_leave_users_out.py --data $PHQ4_NPZ --target phq4 --folds 5 --no-cutoff \
        --users comparisons/gala/phq4_users.txt --self-reports $c --all-rows --out $R/phq4_${c}_seed42
    python comparisons/gala/severity_stacking.py --predictions $R/phq4_${c}_seed42/predictions_luo_all.csv \
        --stats $stats --out $R/phq4_${c}_seed42
  done
  python comparisons/gala/evaluate_gala.py --npz $PHQ4_NPZ \
      --run sparse=$R/phq4_sparse_seed42/predictions_severity.csv \
      --run weekly=$R/phq4_weekly_seed42/predictions_severity.csv > $R/tables/rq3_gala.txt
  ;;

ssaqs)  # RQ4: transfer to SSAQS (CPU). WF and LUO run in the same process, in this order.
  SSAQS_DIR=${SSAQS_DIR:?set SSAQS_DIR to the SSAQS folder with one sub-folder per user}
  S=$R/ssaqs
  mkdir -p $S
  python ssaqs/build_dataset.py --data $SSAQS_DIR --out-dir $S
  for s in $SEEDS; do
    python -u ssaqs/run_ensemble.py --lstm-input $S/ssaqs_lstm_input_ces.csv --gbm-input $S/ssaqs_gbm_input_ces.csv \
        --seed $s --out $S/runs/seed$s
  done
  for scheme in WF LUO; do
    sch=$(echo $scheme | tr 'A-Z' 'a-z')
    preds=""
    for s in $SEEDS; do preds="$preds $S/runs/seed$s/ensemble_forecast_${sch}_adaptive_preds.csv"; done
    python ssaqs/evaluate.py --schema $scheme --data $SSAQS_DIR --input $S/ssaqs_lstm_input_ces.csv \
        --preds $preds --out $S/rq4_eval
  done
  python ssaqs/rq4_table.py --dir $S/rq4_eval > $R/tables/rq4_table.txt
  python ssaqs/rq4_figure.py --rows $S/rq4_eval/rq4_rows_LUO_seed42__ensemble_forecast_luo_adaptive_preds.csv \
      --out $R/figures
  ;;

*) echo "unknown step: $step"; exit 1 ;;
esac
echo "== step $step done"
done
