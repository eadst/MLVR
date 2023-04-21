from .vct_data import VCT_DATA


dataset_list = {
                "vct": VCT_DATA
                }


def build_dataset(dataset, root_path, data_path, shots):
    return dataset_list[dataset](root_path, data_path, shots)