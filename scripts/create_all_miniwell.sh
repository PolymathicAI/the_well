#!/bin/bash

# Sorted by size:
#     "turbulent_radiative_layer_2D", #     "active_matter", #     "helmholtz_staircase", #     "viscoelastic_instability",
#     "MHD_64", #     "post_neutron_star_merger", #     "shear_flow", #     "gray_scott_reaction_diffusion", #     "acoustic_scattering_discontinuous",
#     "planetswe", #     "rayleigh_taylor_instability", #     "supernova_explosion_64", #     "acoustic_scattering_inclusions", #     "acoustic_scattering_maze",
#     "rayleigh_benard", #     "convective_envelope_rsg", #     "turbulent_radiative_layer_3D", #     "supernova_explosion_128", #     "turbulence_gravity_cooling",
#     "euler_multi_quadrants_openBC", #     "euler_multi_quadrants_periodicBC", #     "MHD_256"
# ]

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
)

SPATIAL=4
TIME=2
TIME_FRACTION=1.0

for dataset in ${datasets[@]}; do
    echo "Processing dataset: $dataset"
    python ./scripts/create_miniwell.py \
        "/mnt/ceph/users/mcranmer/the_well/mini_spatial${SPATIAL}x_time${TIME}x_frac${TIME_FRACTION}" \
        --dataset $dataset \
        --max-files-per-train 10 \
        --max-files-per-val 2 \
        --max-files-per-test 2 \
        --time-fraction $TIME_FRACTION \
        --spatial-downsample-factor $SPATIAL \
        --time-downsample-factor $TIME
done

