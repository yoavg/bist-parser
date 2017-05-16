#!/bin/bash

TRAIN=../../data/en-ud-train.conllu
GPUID=2

mkdir timings

#python parser.py --train $TRAIN --outdir . --params tmp.pkl  --model tmp.model --epochs 1 --dynet-mem 4096 --dynet-seed 1 --userlmost --userl --bibi-lstm --batch-size 1 --dynet-autobatch 0  > timings/default.CPU.ab0
#python parser.py --train $TRAIN --outdir . --params tmp.pkl  --model tmp.model --epochs 1 --dynet-mem 4096 --dynet-seed 1 --userlmost --userl --bibi-lstm --batch-size 1 --dynet-autobatch 0  --dynet-gpu-ids $GPUID > timings/default.GPU.ab0

for BATCH_SIZE in 1 2 4 8 16 32 64; do
   python parser.py --train $TRAIN --outdir . --params tmp.pkl  --model tmp.model --epochs 1 --dynet-mem 4096 --dynet-seed 1 --userlmost --userl --bibi-lstm --batch-size $BATCH_SIZE --dynet-autobatch 1  > timings/default.CPU.ab-$BATCH_SIZE
   python parser.py --train $TRAIN --outdir . --params tmp.pkl  --model tmp.model --epochs 1 --dynet-mem 4096 --dynet-seed 1 --userlmost --userl --bibi-lstm --batch-size $BATCH_SIZE --dynet-autobatch 1  --dynet-gpu-ids $GPUID > timings/default.GPU.ab-$BATCH_SIZE
done
