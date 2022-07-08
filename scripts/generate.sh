#!/bin/bash

overrides=(
    'object=butter dataset.num_scenes=20 num_objs=50 pack_type=random'
    "object=cookies dataset.num_scenes=20 num_objs=20 pack_type=random"
    "object=cheese dataset.num_scenes=20 num_objs=50 pack_type=random"
    "object=mac_cheese dataset.num_scenes=20 num_objs=30 pack_type=random"
    "object=raisins dataset.num_scenes=20 num_objs=40 pack_type=random"
    'object=butter dataset.num_scenes=4 num_objs=100 pack_type=ordered'
    "object=cookies dataset.num_scenes=4 num_objs=20 pack_type=ordered"
    "object=cheese dataset.num_scenes=4 num_objs=50 pack_type=ordered"
    "object=mac_cheese dataset.num_scenes=4 num_objs=30 pack_type=ordered"
    "object=raisins dataset.num_scenes=4 num_objs=80 pack_type=ordered"
    "object=pudding dataset.num_scenes=2 num_objs=40 pack_type=random dataset.output_dir=/home/mujin/sandbox/mujin/repos/datasets/bop/ucn_test"
    "object=granola dataset.num_scenes=2 num_objs=40 pack_type=random dataset.output_dir=/home/mujin/sandbox/mujin/repos/datasets/bop/ucn_test"
    "object=popcorn dataset.num_scenes=2 num_objs=40 pack_type=random dataset.output_dir=/home/mujin/sandbox/mujin/repos/datasets/bop/ucn_test"
    "object=spaghetti dataset.num_scenes=2 num_objs=40 pack_type=random dataset.output_dir=/home/mujin/sandbox/mujin/repos/datasets/bop/ucn_test"
    "object=pudding dataset.num_scenes=2 num_objs=100 pack_type=ordered dataset.output_dir=/home/mujin/sandbox/mujin/repos/datasets/bop/ucn_test"
    "object=granola dataset.num_scenes=2 num_objs=20 pack_type=ordered dataset.output_dir=/home/mujin/sandbox/mujin/repos/datasets/bop/ucn_test"
    "object=popcorn dataset.num_scenes=2 num_objs=60 pack_type=ordered dataset.output_dir=/home/mujin/sandbox/mujin/repos/datasets/bop/ucn_test"
    "object=spaghetti dataset.num_scenes=2 num_objs=20 pack_type=ordered dataset.output_dir=/home/mujin/sandbox/mujin/repos/datasets/bop/ucn_test"
)

for opt in "${overrides[@]}";do
    echo "running: blenderproc run ./src/gen_data.py scene=scene2 dataset.num_poses_per_scene=5 ${opt}"
    blenderproc run ./src/gen_data.py ${opt} || true
done