#!/usr/bin/env bash
#SBATCH --job-name=ic27_data
#SBATCH --account=mst114566
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-1%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
BACKBONES=(llama32_3b llama31_8b)
rgcls prepare-data --backbone "${BACKBONES[$SLURM_ARRAY_TASK_ID]}"
