# Run the simulated push experiment with default hyperparameters
python main_reptile.py --experiment=sim_push --demo_file=data/sim_push --demo_gif_dir=data/sim_push/ --gif_prefix=object \
                   --T=2 --im_width=125 --im_height=125 --val_set_size=76 --metatrain_iterations=3 --init=xavier \
                  --meta_batch_size=2 --train_update_lr=0.01 --clip=True --clip_min=-10 --clip_max=10 \
                  --fp=True --num_filters=16 --filter_size=5 --num_conv_layers=4 --num_strides=4 --all_fc_bt=False --bt_dim=20 \
                  --num_fc_layers=3 --layer_size=200 --loss_multiplier=50.0 --two_head=True --log_dir=logs/sim_push \
                  --use_noisy_demos=False
