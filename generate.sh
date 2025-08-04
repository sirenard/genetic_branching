#!/usr/bin/bash

for i in {0..25}
do
    letter=$(printf \\$(printf '%03o' $((65+i))))
    start_at=$((12+i))
#    echo $i $letter
    python python/khalil.py --sample_prefix $letter --seed $start_at > "out$letter" &
done

for job in `jobs -p`
do
    wait $job
done
