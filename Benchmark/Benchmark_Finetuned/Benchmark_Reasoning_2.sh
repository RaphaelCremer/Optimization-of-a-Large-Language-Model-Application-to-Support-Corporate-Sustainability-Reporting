#!/bin/bash

module load Python
source /home/x/venvs/benchmark_reasoning/bin/activate

export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/x/Code
python Benchmark/Benchmark_Finetuned/Benchmark_Reasoning_2.py
