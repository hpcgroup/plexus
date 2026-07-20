#!/bin/bash
# Time-to-accuracy leaderboard for Wave-1 hyperparameter runs (eval_every=1).
cd /pscratch/sd/c/cunyang/gnn/plexus
echo "config | epoch_ms | 63.0% | 63.41% | 64.0% | best"
for f in result/papers_acc/papers_train_hub*wu*ep*.log; do
  [ -f "$f" ] || continue
  n=$(basename $f .log | sed 's/papers_train_hub_//;s/_g4x4x8//;s/_g4x4x4//;s/_do0.0_c1.0_cosine//;s/_s0//')
  e9=$(grep -oE "'epoch 9 \| Max Time: [0-9.]+" $f | grep -oE "[0-9.]+$" | head -1)
  [ -z "$e9" ] && { echo "$n | INCOMPLETE"; continue; }
  ee=$(grep -m1 -oE "'eval_every': [0-9]+" $f | grep -oE "[0-9]+")
  best=$(grep "TEST: acc" $f | awk '{print $3}' | tr -d ',' | sort -rn | head -1)
  cross=$(grep "TEST: acc" $f | awk '{print $3}' | tr -d ',' | awk -v e=$e9 -v k=${ee:-1} '{ep=NR*k-1; if($1>=0.63 && !a){printf "%.0fs@ep%d | ", ep*e/1000, ep; a=1} if($1>=0.6341 && !b){printf "%.0fs@ep%d | ", ep*e/1000, ep; b=1} if($1>=0.64 && !c){printf "%.0fs@ep%d", ep*e/1000, ep; c=1}} END{if(!a)printf "- | - | "; else if(!b)printf "- | -"; else if(!c)printf "-"}')
  echo "$n | ${e9}ms | $cross | $best"
done | sort -t'|' -k3 -n
