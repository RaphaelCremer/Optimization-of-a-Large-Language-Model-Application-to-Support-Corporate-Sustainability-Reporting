#!/usr/bin/zsh

#Adapted and modified from Muennighoff, N, Yang, Z, Shi, W, Li, XL, Fei-Fei, L, Hajishirzi, H, Zettlemoyer, L, Liang, P, Candès, E & Hashimoto, T 2025, s1: Simple test-time scaling. Available from: https://github.com/simplescaling/s1 [19. September 2026] under Apache License Version 2.0, January 2004 found under /Third_Party_Licenses

uid="$(date +%Y%m%d-%H%M%S)"

mkdir -p log

sbatch --job-name=TestrunQwen2.5_14B_Instruct \
    --nodes=1 \
    --ntasks-per-node=1 \
    --partition=c23g \
    --gres=gpu:4 \
    --mem=128G \
    --time=01:00:00 \
    --cpus-per-task=16 \
    SFT/SFT/sft_slurm.sh \
        --uid ${uid} \
        > log/sft_${uid}.txt 2>&1 &