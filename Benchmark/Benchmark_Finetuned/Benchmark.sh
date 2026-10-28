#!/bin/bash

module load Python
source /home/x/venvs/sft-test/bin/activate

cd /home/x/Code
python Benchmark/Benchmark_Finetuned/Benchmark.py
