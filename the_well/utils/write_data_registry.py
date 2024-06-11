import json
import os


def generate_file_lists(root_dirs):
    """
    Generates lists of all files under each specified root directory.
    Returns a dictionary where each key is a root directory and the value is a list of file paths.

    :param root_dirs: List of directories to index.
    :return: Dictionary of file lists indexed by directory.
    """
    base_url = "https://users.flatironinstitute.org/~polymathic/data/the_well/"
    all_files = {}
    for root_dir in root_dirs:
        file_list = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # Append the full path or just the filename
                file_list.append(
                    os.path.join(root, file).replace(
                        "/mnt/home/polymathic/ceph/the_well/", base_url
                    )
                )
        # Use the last part of the directory path as the key to make it more readable
        # key = os.path.basename(root_dir) if os.path.basename(root_dir) else os.path.basename(os.path.dirname(root_dir))
        key = os.path.dirname(root_dir.rstrip("/")).split("/")[-1]
        print(key)

        all_files[key] = file_list
    return all_files


def write_to_json(data, output_file):
    """
    Writes the given data to a JSON file.

    :param data: Data to write (dictionary format).
    :param output_file: Path to the output JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    # List of directories you want to index
    directories_to_index = [
        "/mnt/home/polymathic/ceph/the_well/datasets/acoustic_scattering_discontinuous_2d/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/acoustic_scattering_inclusions_2d/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/acoustic_scattering_maze_2d/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/active_matter/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/convective_envelope_rsg/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/euler_quadrants/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/helmholtz_staircase/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/MHD_64/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/MHD_256/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/pattern_formation/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/planetswe/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/post_neutron_star_merger/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/rayleigh_benard/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/rayleigh_taylor_instability/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/shear_flow/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/supernova_explosion_64/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/supernova_explosion_128/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/turbulence_gravity_cooling/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/turbulent_radiative_layer_2D/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/turbulent_radiative_layer_3D/data",
        "/mnt/home/polymathic/ceph/the_well/datasets/viscoelastic_instability/data",
    ]
    # Generate file lists
    directory_files = generate_file_lists(directories_to_index)
    # Output JSON file
    json_file_path = "data_registry.json"
    # Write to JSON
    write_to_json(directory_files, json_file_path)
    print(f"File lists have been written to {json_file_path}")
