#!/bin/bash

mkdir -p log

sbatch \
    --job-name=Qwen14B_Benchmark_Reasoning_2 \
    --partition=c23g \
    --nodes=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=00:30:00 \
    --output=log/benchmark_Reasoning_2_%j.out \
    --error=log/benchmark_Reasoning_2_%j.out \
    Benchmark/Benchmark_Finetuned/Benchmark_Reasoning_2.sh
