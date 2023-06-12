import logging
import pandas as pd
import shutil, os
import os.path as osp

import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg

from .data_utils import make_balanced_testset, read_graph_list
logger = logging.getLogger(__name__)

class PygGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root = 'dataset', transform=None, pre_transform = None, meta_dict = None):
        
        self.name = name ## original name, e.g., ogbg-molhiv
        self.dir_name = '_'.join(name.split('-')) 
        
        # check if previously-downloaded folder exists.
        # If so, use that one.
        if osp.exists(osp.join(root, self.dir_name + '_pyg')):
            self.dir_name = self.dir_name + '_pyg'

        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

        self.download_name = name.split('-')[1].replace('mol','')
        if self.download_name == 'esol':
            self.url = 'http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/esol.zip'
        elif self.download_name == 'freesolv':
            self.url = 'http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/freesolv.zip'
        elif self.download_name == 'lipo':
            self.download_name = 'lipophilicity'
            self.url = 'http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/lipophilicity.zip'
        else:
            pass
        self.num_tasks = 1
        self.eval_metric = 'rmse'
        self.task_type = 'regression'
        self.__num_classes__ = -1
        self.binary = False

        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)

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
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            file_names.append('node-feat')
            file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        url = self.url
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        ### read pyg graph list
        add_inverse_edge = True
        additional_node_files = []
        additional_edge_files = []
        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)
        if self.binary:
            graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
        else:
            graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
        for i, g in enumerate(data_list):
            g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)
        self.labeled_data_len = len(data_list)
        print('Labeled Finished with length ', self.labeled_data_len)
        data_list.extend(read_graph_list(self.original_root+'/QM9.txt', property_name=self.name, process_labeled=False))
        self.total_data_len = len(data_list)
        print('Label + Unlabeled data length', self.total_data_len)
        data_list.extend(read_graph_list(self.original_root+'/plym_all.csv', property_name=self.name, process_labeled=False))
        self.total_data_len = len(data_list)
        print('Label + Unlabeled data length', self.total_data_len)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
