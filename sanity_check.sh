python train_baseline.py \
    --n_epochs 20 \
    --device cpu \
    --output_folder consistency_checkpoints/ \
    --eval_every 10 \
    --observed_mask_pct 50 \
    --batch_size 16