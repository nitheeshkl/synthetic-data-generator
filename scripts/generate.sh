#!/bin/bash

overrides=(
    'object=butter dataset.num_images=100 num_objs=50 pack_type=random'
    'object=butter dataset.num_images=50 num_objs=150 pack_type=ordered'
    "object=cookies dataset.num_images=100 num_objs=10 pack_type=random"
    "object=cookies dataset.num_images=50 num_objs=30 pack_type=ordered"
    "object=cheese dataset.num_images=100 num_objs=50 pack_type=random"
    "object=cheese dataset.num_images=50 num_objs=150 pack_type=ordered"
    "object=mac_cheese dataset.num_images=100 num_objs=30 pack_type=random"
    "object=mac_cheese dataset.num_images=50 num_objs=50 pack_type=ordered"
    "object=raisins dataset.num_images=100 num_objs=40 pack_type=random"
    "object=raisins dataset.num_images=50 num_objs=100 pack_type=ordered"
)

for opt in "${overrides[@]}";do
    echo "running: blenderproc run ./src/gen_data.py ${opt}"
    blenderproc run ./src/gen_data.py ${opt} || true
done