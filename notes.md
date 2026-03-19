  python run_glue.py --model_type bert --model_name_or_path bert-base-cased --task_name RTE --do_train --do_eval --data_dir ./glue_data/RTE --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate 2e-5  --num_train_epochs 3 --output_dir /tmp/RTE/ --overwrite_output_dir


cd /n/fs/scratch/tl0463/Courses/COS568/COS568-DistLM-SP26/task2a                                                 
  for rank in 0 1 2 3; do python run_glue.py --model_type bert --model_name_or_path bert-base-cased --task_name RTE --do_train --do_eval --data_dir ../glue_data/RTE --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 1 --output_dir /tmp/RTE_2a --overwrite_output_dir --local_rank $rank --master_ip 127.0.0.1 --master_port 12345 --world_size 4 & done; wait 