#!/bin/bash

python3 polynet.py --polyhedron cube --num_channels 28 --batch_size 64 --dataset happi20_poly/cube_48/ --dropout_rate 0.1 --learning_rate 3e-3 --num_epochs -30 --sched_gamma 0.7 --sched_step_size 20 --task segmentation --frac_hier 0.5 --act_fn ReLU --use_dataparallel true --normalize_data true --use_cube_pad true --seed 1
python3 polynet.py --polyhedron cube --num_channels 28 --batch_size 64 --dataset happi20_poly/cube_48/ --dropout_rate 0.1 --learning_rate 3e-3 --num_epochs -30 --sched_gamma 0.7 --sched_step_size 20 --task segmentation --frac_hier 0.5 --act_fn ReLU --use_dataparallel true --normalize_data true --use_cube_pad true --seed 2
python3 polynet.py --polyhedron cube --num_channels 28 --batch_size 64 --dataset happi20_poly/cube_48/ --dropout_rate 0.1 --learning_rate 3e-3 --num_epochs -30 --sched_gamma 0.7 --sched_step_size 20 --task segmentation --frac_hier 0.5 --act_fn ReLU --use_dataparallel true --normalize_data true --use_cube_pad true --seed 3
