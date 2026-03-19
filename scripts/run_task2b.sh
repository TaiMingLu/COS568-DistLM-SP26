#!/bin/bash
#SBATCH --job-name=task2b
#SBATCH --partition=cs
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=task2b_%j.out

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
MASTER_PORT=12345

source /n/fs/scratch/tl0463/Courses/COS568/COS568-DistLM-SP26/.venv/bin/activate

srun python /n/fs/scratch/tl0463/Courses/COS568/COS568-DistLM-SP26/task2b/run_glue.py \
    --model_type bert --model_name_or_path bert-base-cased \
    --task_name RTE --do_train --do_eval \
    --data_dir /n/fs/scratch/tl0463/Courses/COS568/COS568-DistLM-SP26/glue_data/RTE \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir /n/fs/scratch/tl0463/Courses/COS568/COS568-DistLM-SP26/output/task2b/ --overwrite_output_dir \
    --local_rank \$SLURM_PROCID \
    --master_ip $MASTER_ADDR --master_port $MASTER_PORT --world_size 4
