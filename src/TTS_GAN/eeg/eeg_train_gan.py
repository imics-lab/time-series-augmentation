import os

for id in range(2):
    filename = f'eeg_ttsgan_class{id}'
    os.system(f"python main.py \
        --gpu 1 \
        --dataset EEG \
        --class_id {id} \
        --GAN_type TTS_GAN \
        --max_iter 1000 \
        --n_critic 1 \
        --num_workers 16 \
        --g_lr 0.0001 \
        --d_lr 0.0003 \
        --optimizer adam \
        --loss lsgan \
        --beta1 0.5 \
        --beta2 0.999 \
        --grow_steps 0 0 \
        --batch_size 16 \
        --latent_dim 100 \
        --g_seq_len 178 \
        --g_channel 1 \
        --d_seq_length 178 \
        --d_patch_size 1 \
        --d_channel 1 \
        --d_n_classes 1 \
        --exp_name {filename}")