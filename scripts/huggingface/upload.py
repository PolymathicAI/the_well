"""Script to upload dataset to Hugging Face hub."""

import argparse
import logging
import multiprocessing
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
from functools import partial

import jinja2
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def edit_readme(
    input_filename: str, output_filename: str, dataset_tag: str, dataset_name: str
):
    """Edit the original Well dataset ReadMe file in two ways:
    - Convert MathJax to KaTeX
    - Add rendered template as a header to the original ReadMe file

    Github renders LaTeX in markdown files using MathJax and `$...$` delimiters.
    c.f. https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions
    This is incompatible with HF relying on KaTeX with a different syntax
    c.f. https://huggingface.co/docs/hub/en/model-cards#can-i-write-latex-in-my-model-card
    This only concerns the inline syntax. Display syntax is the same for both MathJax and KaTex.

    Convert Github's MathJax syntax to HF's KaTex syntax.
    """
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(pathlib.Path(__file__).parent)
    )
    header_template = env.get_template("DATASET_README_HEADER_TEMPLATE.md")
    header = header_template.render(tag=dataset_tag, dataset_name=dataset_name)
    mathjax_syntax = re.compile(r"\$(?P<equation>[^$]+)\$")
    with open(output_filename, "w") as output_file:
        output_file.write(header + "\n\n")
        with open(input_filename, "r") as input_file:
            for line in input_file:
                new_line = re.sub(mathjax_syntax, r"\\\(\g<equation>\\\)", line)
                output_file.write(new_line)


def repack_h5(input_filename: str, output_filename: str):
    """Repack h5 file for the cloud.
    Gather metadata at the head of the file.
    """
    n_bytes = 8 * 1024 * 1024
    page_size_bytes = 8 * 1024 * 1024
    command = f"h5repack -S PAGE -G {page_size_bytes} --metadata_block_size={n_bytes} {input_filename} {output_filename}"
    args = shlex.split(command)
    subprocess.run(args, check=True)


def upload_folder(folder: str, repo_id: str):
    api = HfApi()
    api.upload_large_folder(
        repo_id=repo_id, folder_path=folder, repo_type="dataset", private=True
    )


def is_file_valid(filename: pathlib.Path) -> bool:
    if not filename.is_file():
        return False
    elif ".cache" in filename.parts:
        return False
    elif filename.suffix == ".lock":
        return False
    return True


def process_file(
    root_directory: pathlib.Path,
    file_path: pathlib.Path,
    output_directory: pathlib.Path,
    dataset_tag: str,
    dataset_name: str,
):
    in_dir_file_path = file_path.relative_to(root_directory)
    # Skip irrelevant files
    if not is_file_valid(file_path):
        return
    # Create corresponding directory
    target_dir = pathlib.Path(output_directory) / in_dir_file_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    target_filename = target_dir / in_dir_file_path.name
    # Process ReadMe
    if file_path.name.lower() == "readme.md":
        logger.debug(f"Convert ReadMe {file_path}")
        edit_readme(file_path, target_filename, dataset_tag, dataset_name)
    # Process HDF5
    elif file_path.suffix in [".hdf", ".h5", ".hdf5"]:
        logger.debug(f"Repack HDF5 {file_path}")
        repack_h5(file_path, target_filename)
    # Simply copy remaining files
    else:
        logger.debug(f"Link file {file_path}")
        target_filename.symlink_to(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("repo_id", type=str)
    parser.add_argument("-t", "--tag", default=None)
    parser.add_argument(
        "-n",
        "--n_proc",
        type=int,
        default=1,
        help="Number of workers for the file processing.",
    )
    args = parser.parse_args()
    directory = pathlib.Path(args.directory)
    repo_id = args.repo_id
    n_proc = args.n_proc
    dataset_tag = args.tag
    dataset_name = pathlib.Path(repo_id).name

    files = list(directory.rglob("*"))
    chunk_size = len(files) // n_proc
    with tempfile.TemporaryDirectory() as tmp_dirname:
        tmp_dirname = pathlib.Path(tmp_dirname)
        process_fn = partial(
            process_file,
            directory,
            output_directory=tmp_dirname,
            dataset_tag=dataset_tag,
            dataset_name=dataset_name,
        )
        with multiprocessing.Pool() as pool:
            pool.map(process_fn, files, chunksize=chunk_size)

        upload_folder(tmp_dirname, repo_id)
