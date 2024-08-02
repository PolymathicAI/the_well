from the_well.utils.interface import Interface
from the_well.benchmark.data import GenericWellDataset
from the_well.benchmark.models.unet_classic import UNetClassic


if __name__ == "__main__":
    dataset = GenericWellDataset(
        well_base_path="/mnt/home/polymathic/ceph/the_well/",
        well_dataset_name="active_matter",
        n_steps_input=4,
        n_steps_output=1,
        well_split_name="valid",
    )
    interface = Interface.from_dataset(dataset)
    model = UNetClassic(11, 11, dataset.metadata)
    print(interface.check(model, history=4, horizon=1))
