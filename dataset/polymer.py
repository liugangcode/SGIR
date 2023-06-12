from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import logging
import os
import os.path as osp
import pandas as pd
import torch
import copy

from .data_utils import make_balanced_testset, read_graph_list
logger = logging.getLogger(__name__)

class PolymerRegDataset(InMemoryDataset):
    def __init__(self, name='plym-oxygen', root ='data', transform=None, pre_transform=None):
        self.name = name
        self.dir_name = '_'.join(name.split('-'))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.processed_root = osp.join(osp.abspath(self.root))

        self.num_tasks = 1
        self.eval_metric = 'rmse'
        self.task_type = 'regression'
        self.__num_classes__ = '-1'
        self.binary = 'False'

        super(PolymerRegDataset, self).__init__(self.processed_root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.total_data_len = len(self.data.y.view(-1))
        self.unlabeled_data_len = torch.isnan(self.data.y.view(-1)).sum().item()
        self.labeled_data_len = self.total_data_len - self.unlabeled_data_len
        print('# label: {}, # unlabeled: {}, # total: {}'.format(self.labeled_data_len, self.unlabeled_data_len, self.total_data_len))
    
    def get_unlabeled_idx(self):
        return torch.arange(self.labeled_data_len, self.total_data_len, dtype=torch.long)

    def get_idx_split(self, split_type = 'balance', regenerate=False):
        if split_type is None:
            split_type = 'balance'
            
        path = osp.join(self.root, 'split', split_type)
        if not os.path.exists(path):
            os.makedirs(path)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))
        if os.path.isfile(osp.join(path, 'train.csv.gz')) and os.path.isfile(osp.join(path, 'valid.csv.gz')) and os.path.isfile(osp.join(path, 'test.csv.gz')) and not regenerate:
            train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
            valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
            test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]
        else:
            train_idx, valid_idx, test_idx = make_balanced_testset(self.name, self.data.y[:self.labeled_data_len], max_size=15, seed=666, subsample_train = None, vis=False)
            df_train = pd.DataFrame({'train': train_idx})
            df_valid = pd.DataFrame({'valid': valid_idx})
            df_test = pd.DataFrame({'test': test_idx})
            df_train.to_csv(osp.join(path, 'train.csv.gz'), index=False, header=False, compression="gzip")
            df_valid.to_csv(osp.join(path, 'valid.csv.gz'), index=False, header=False, compression="gzip")
            df_test.to_csv(osp.join(path, 'test.csv.gz'), index=False, header=False, compression="gzip")
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        ### labeled data
        data_list = read_graph_list(osp.join(self.root, 'raw' ,self.name.split('-')[1]+'_raw.csv'), property_name=self.name, process_labeled=True)
        print(data_list[:3])
        self.labeled_data_len = len(data_list)

        data_list.extend(read_graph_list(osp.join(self.original_root,'plym_all.csv'), property_name=self.name, drop_property=True, process_labeled=False))
        self.total_data_len = len(data_list)
        data_list.extend(read_graph_list(osp.join(self.original_root,'QM9.txt'), property_name=self.name, process_labeled=False))
        self.total_data_len = len(data_list)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    pass