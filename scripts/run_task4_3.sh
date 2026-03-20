#!/bin/bash
#SBATCH --job-name=prof_3
#SBATCH --partition=all
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/fs/scratch/tl0463/Courses/COS568/COS568-DistLM-SP26/scripts/%x.out

set -euo pipefail

REPO_ROOT=/n/fs/scratch/tl0463/Courses/COS568/COS568-DistLM-SP26
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
MASTER_PORT=$((12000 + RANDOM % 50000))

source "$REPO_ROOT/.venv/bin/activate"

echo "[$(date -Is)] batch host=$(hostname) job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES ntasks=$SLURM_NTASKS master=${MASTER_ADDR}:${MASTER_PORT}"

srun --label --kill-on-bad-exit=1 \
    --export=ALL,REPO_ROOT="$REPO_ROOT",MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT",WORLD_SIZE="$SLURM_NTASKS" \
    bash -lc '
echo "[$(date -Is)] task host=$(hostname) procid=${SLURM_PROCID} localid=${SLURM_LOCALID} nodeid=${SLURM_NODEID} master=${MASTER_ADDR}:${MASTER_PORT}"
python -u "$REPO_ROOT/task4/task3/run_glue.py" \
    --model_type bert --model_name_or_path bert-base-cased \
    --task_name RTE --do_train \
    --data_dir "$REPO_ROOT/glue_data/RTE" \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir "$REPO_ROOT/task4/task3/output/" --overwrite_output_dir \
    --local_rank "${SLURM_PROCID}" \
    --master_ip "${MASTER_ADDR}" --master_port "${MASTER_PORT}" --world_size "${WORLD_SIZE}"
'
