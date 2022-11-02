#!/bin/bash


source /glade/work/wchapman/miniconda3/etc/profile.d/conda.sh
conda activate gputorch

arrayhor=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
arrayseed=(1 2 3 6)

for i in "${arrayhor[@]}"
do
for b in "${arrayseed[@]}"
  do
      python run_deeplim.py --horizon $i --seed $b --epochs 30 --gpu_id 0 >> out_epochs30_{$i}_{$b}.txt &
      pid1=$!
      wait $pid1
done
done
