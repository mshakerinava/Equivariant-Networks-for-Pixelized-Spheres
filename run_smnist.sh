#!/bin/bash

python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.25 --use_cube_pad true --seed 1
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.25 --use_cube_pad true --seed 2
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.25 --use_cube_pad true --seed 3


python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0 --use_cube_pad false --seed 1
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0 --use_cube_pad false --seed 2
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0 --use_cube_pad false --seed 3
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0 --use_cube_pad false --seed 4
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0 --use_cube_pad false --seed 5

# python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.25 --use_cube_pad false --seed 1
# python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.25 --use_cube_pad false --seed 2
# python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.25 --use_cube_pad false --seed 3
# python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.25 --use_cube_pad false --seed 4
# python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.25 --use_cube_pad false --seed 5

python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.5 --use_cube_pad false --seed 1
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.5 --use_cube_pad false --seed 2
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.5 --use_cube_pad false --seed 3
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.5 --use_cube_pad false --seed 4
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.5 --use_cube_pad false --seed 5

# python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.75 --use_cube_pad false --seed 1
# python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.75 --use_cube_pad false --seed 2
# python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.75 --use_cube_pad false --seed 3
# python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.75 --use_cube_pad false --seed 4
# python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 0.75 --use_cube_pad false --seed 5

python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 1.0 --use_cube_pad false --seed 1
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 1.0 --use_cube_pad false --seed 2
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 1.0 --use_cube_pad false --seed 3
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 1.0 --use_cube_pad false --seed 4
python3 polynet.py --polyhedron cube --num_channels 20 --batch_size 128 --dataset smnist_poly/cube_24_rr/ --dropout_rate 0.333 --learning_rate 1e-3 --num_epochs 50 --sched_gamma 0.1 --sched_step_size 20 --task classification --frac_hier 1.0 --use_cube_pad false --seed 5
