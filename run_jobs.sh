#!/bin/bash
cd ~/sadrl
cat jobs_rl.txt | xargs -P 2 -L 1 bash -c 'python3 run_rl.py --env $0 --method $1 --seeds $2 --out results/rl >> results/rl_log.txt 2>&1'
