import os
from scipy.io import loadmat
from .utils import Datum, DatasetBase, read_json

template = ['a photo of a {}.']

def read_split(filepath, path_prefix):
    def _convert(items):
        out = []
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            item = Datum(
                impath=impath,
                label=int(label),
                classname=classname
            )
            out.append(item)
        return out

    print(f'Reading split from {filepath}')
    split = read_json(filepath)
    train = _convert(split['train'])
    val = _convert(split['val'])
    test = _convert(split['test'])

    return train, val, test
    
class VCT_DATA(DatasetBase):

    def __init__(self, root, data_path, num_shots):
        self.dataset_dir = root
        self.split_path = data_path

        self.template = template

        train, val, test = read_split(self.split_path, self.dataset_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)['annotations'][0]
        meta_file = loadmat(meta_file)['class_names'][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]['fname'][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]['class'][0, 0]
            label = int(label) - 1 # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(' ')
            year = names.pop(-1)
            names.insert(0, year)
            classname = ' '.join(names)
            item = Datum(
                impath=impath,
                label=label,
                classname=classname
            )
            items.append(item)
        return items