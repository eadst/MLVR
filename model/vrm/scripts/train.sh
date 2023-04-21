# ViT-B/16
job_name="train_vsm_vit16_test01" # train job name
DATA_PATH="/dfs/data/others/aicity23/dataset" # data path
python -m torch.distributed.launch --nproc_per_node=4 \
    train_vrm.py --do_train --num_thread_reader=8 \
    --epochs=50 --batch_size=40 --n_display=10 \
    --data_path ${DATA_PATH}/aicity \
    --features_path ${DATA_PATH}/aicity/testvideobox \
    --output_dir ckpts/${job_name} \
    --lr 1e-4 --max_words 32 --max_frames 20 --batch_size_val 40 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 2>&1 | tee -a log/${job_name}
