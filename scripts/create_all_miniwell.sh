#!/bin/bash

# Sorted by size:
datasets=(
    "turbulent_radiative_layer_2D"
    "active_matter"
    "helmholtz_staircase"
    "viscoelastic_instability"
    "MHD_64"
    "post_neutron_star_merger"
    "shear_flow"
    "gray_scott_reaction_diffusion"
    "acoustic_scattering_discontinuous"
    "planetswe"
    "rayleigh_taylor_instability"
    "supernova_explosion_64"
    "acoustic_scattering_inclusions"
    "acoustic_scattering_maze"
    "rayleigh_benard"
    "turbulence_gravity_cooling"
    "euler_multi_quadrants_openBC"
    "euler_multi_quadrants_periodicBC"
    ## These ones are too large:
    # "convective_envelope_rsg"
    # "MHD_256"
)

SPATIAL=4
TIME=2
TIME_FRACTION=1.0

# Create the logging directory if it does not exist
mkdir -p ./logs

for dataset in ${datasets[@]}; do
    echo "Processing dataset: $dataset"
    nohup python ./scripts/create_miniwell.py \
        "/mnt/ceph/users/mcranmer/the_well/mini_v2_spatial${SPATIAL}x_time${TIME}x" \
        --dataset $dataset \
        --max-trajectories-per-train 20 \
        --max-trajectories-per-val 5 \
        --max-trajectories-per-test 5 \
        --time-fraction $TIME_FRACTION \
        --spatial-downsample-factor $SPATIAL \
        --time-downsample-factor $TIME > ./logs/${dataset}.log 2>&1 &
    sleep 1
done

wait
