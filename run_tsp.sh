cd ~/sadrl
for e in mae eps; do for s in 0 1 2; do
  [ -f results/tsp/sel_${e}_s${s}.pt ] && [ "$e$s" != "mae0" ] && continue
  [ "$e$s" == "mae0" ] && [ -f results/tsp/train_mae_s0.json ] && python3 -c "import json;assert len(json.load(open('results/tsp/train_mae_s0.json'))['hist'])==400" 2>/dev/null && continue
  python3 tsp_exp.py train --seed $s --explore $e 2>&1 | grep -v Warn | tail -1
done; done
python3 tsp_exp.py ortools --secs 10 2>&1 | tail -2
python3 tsp_exp.py eval --runs 10 2>&1 | grep -v Warn
