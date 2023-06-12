
from ogb.utils.features import (atom_to_feature_vector,bond_to_feature_vector) 
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
import torch

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))

def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

def log_base(base, x):
    return np.log(x) / np.log(base) 

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    atom_label = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
        atom_label.append(atom.GetSymbol())

    x = np.array(atom_features_list, dtype = np.int64)
    atom_label = np.array(atom_label, dtype = np.str)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    
    return graph 

import copy
import pathlib
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

def labeled2graphs(raw_dir):
    '''
        - raw_dir: the position where property csv stored,  
    '''
    path_suffix = pathlib.Path(raw_dir).suffix
    if path_suffix == '.csv':
        df_full = pd.read_csv(raw_dir, engine='python')
        df_full.set_index('SMILES', inplace=True)
        print(df_full[:5])
    else:
        raise ValueError("Support only csv.")
    graph_list = []
    for smiles_idx in tqdm(df_full.index[:]):
        graph_dict = smiles2graph(smiles_idx)
        props = df_full.loc[smiles_idx]
        for (name,value) in props.iteritems():
            graph_dict[name] = np.array([[value]])
        graph_list.append(graph_dict)
    return graph_list

def unlabel2graphs(raw_dir, property_name=None, drop_property=False):
    '''
        - raw_dir: the position where property csv stored,  
    '''
    path_suffix = pathlib.Path(raw_dir).suffix
    if path_suffix == '.csv':
        df_full = pd.read_csv(raw_dir, engine='python')
        # select data without current property
        if drop_property:
            df_full = df_full[df_full[property_name.split('-')[1]].isna()]
        df_full = df_full.dropna(subset=['SMILES'])
    elif path_suffix == '.txt':
        df_full = pd.read_csv(raw_dir, sep=" ", header=None, names=['SMILES'])
    else:
        raise ValueError("Support only csv and txt.")
    graph_list = []
    for smiles_idx in tqdm(df_full['SMILES']):
        graph_dict = smiles2graph(smiles_idx)
        graph_dict[property_name.split('-')[1]] = np.array([[np.nan]])
        graph_list.append(graph_dict)
    return graph_list
def read_graph_list(raw_dir, property_name=None, drop_property=False, process_labeled=False):
    print('raw_dir', raw_dir)
    if process_labeled:
        graph_list = labeled2graphs(raw_dir)
    else:
        graph_list = unlabel2graphs(raw_dir, property_name=property_name, drop_property=drop_property)
    pyg_graph_list = []
    print('Converting graphs into PyG objects...')
    for graph in graph_list:
        g = Data()
        g.__num_nodes__ = graph['num_nodes']
        g.edge_index = torch.from_numpy(graph['edge_index'])
        del graph['num_nodes']
        del graph['edge_index']
        # if graph[self.name.split('-')[1]] is not None:
        g.y = torch.from_numpy(graph[property_name.split('-')[1]])
        del graph[property_name.split('-')[1]]

        if graph['edge_feat'] is not None:
            g.edge_attr = torch.from_numpy(graph['edge_feat'])
            del graph['edge_feat']

        if graph['node_feat'] is not None:
            g.x = torch.from_numpy(graph['node_feat'])
            del graph['node_feat']

        addition_prop = copy.deepcopy(graph)
        for key in addition_prop.keys():
            g[key] = torch.tensor(graph[key])
            del graph[key]

        pyg_graph_list.append(g)

    return pyg_graph_list

def make_balanced_testset(dataset_name, labels, max_size=150, seed=666, base = None, subsample_train = None, vis=False):
    prop_name = dataset_name.split('-')[1]

    labels = labels.numpy().reshape(-1)
    if prop_name in ['oxygen']:
        base = 10
        interval = 0.2
        max_size = 8
    elif prop_name in ['density']:
        interval = 0.02
    elif prop_name in ['melting']:
        interval = 10
    elif prop_name in ['molesol']:
        interval = 0.1
    elif prop_name in ['molfreesolv']:
        interval = 0.2
    elif prop_name in ['mollipo']:
        interval = 0.05
    else:
        interval = 1
    if base != None:
        labels = log_base(base, labels)

    max_label_value = np.ceil(max(labels)) + interval
    min_label_value = np.floor(min(labels)) - interval / 2
    bins = np.arange(min_label_value, max_label_value, interval)
    inds = np.digitize(labels, bins)
    u_inds, counts = np.unique(inds, return_counts=True)
    max_size = int(max_size)
    selected_bins = u_inds

    train_inds = []
    valid_inds = []
    test_inds = []
    np.random.seed(seed)
    for i in selected_bins:
        candidates_inds = (inds == i).nonzero()[0]
        each_sample_per_bin_val_test = min(len(candidates_inds) // 3, max_size)
        sample_reorder = np.arange(len(candidates_inds))
        np.random.shuffle(sample_reorder)
        test_inds.append(candidates_inds[sample_reorder][:each_sample_per_bin_val_test])
        valid_inds.append(candidates_inds[sample_reorder][each_sample_per_bin_val_test:each_sample_per_bin_val_test*2])
    test_inds = np.concatenate(test_inds)
    valid_inds = np.concatenate(valid_inds)

    train_inds = np.arange(len(labels))
    train_inds = np.setdiff1d(train_inds,test_inds)
    train_inds = np.setdiff1d(train_inds, valid_inds)
    test_num, valid_num = len(test_inds), len(valid_inds)
    train_num = len(labels) - test_num - valid_num
    if subsample_train is not None:
        train_num = test_num * 3

    train_subsampling = np.arange(len(train_inds))
    np.random.shuffle(train_subsampling)
    train_inds = train_inds[train_subsampling[:int(train_num)]]
    new_train_inds = []

    if subsample_train is not None and subsample_train != 'auto':
        divide_ratio = max(1, int(subsample_train))
        train_bin_inds = np.digitize(labels[train_inds], bins)
        train_bin_u_inds, train_counts = np.unique(train_bin_inds, return_counts=True)
        max_train_freq = max(train_counts)
        for selected_bin in train_bin_u_inds:
            train_inds_candidates = (train_bin_inds == selected_bin).nonzero()[0]
            # to keep region unchanged
            if len(train_inds_candidates) >= max_train_freq / 2:
                cur_sample_num = max(len(train_inds_candidates) // divide_ratio, 20)
            elif len(train_inds_candidates) >= max_train_freq / 10 and len(train_inds_candidates) < max_train_freq / 2:
                cur_sample_num = max(len(train_inds_candidates) // divide_ratio, 5)
            elif len(train_inds_candidates) < max_train_freq / 10:
                cur_sample_num = max(len(train_inds_candidates) // divide_ratio, 1)
            new_train_inds.append(train_inds[train_inds_candidates[:cur_sample_num]])
        new_train_inds = np.concatenate(new_train_inds).reshape(-1)
        train_inds = new_train_inds
        train_num = len(train_inds)


    if vis:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(3, figsize=(6, 9), sharex='all')
        ax[0].hist(labels[train_inds], bins)
        ax[0].set_title(f"[{dataset_name.upper()}] train: {train_inds.shape[0]}")
        ax[1].hist(labels[valid_inds], bins)
        ax[1].set_title(f"[{dataset_name.upper()}] val: {valid_inds.shape[0]}")
        ax[2].hist(labels[valid_inds], bins)
        ax[2].set_title(f"[{dataset_name.upper()}] test: {test_inds.shape[0]}")
        ax[0].set_xlim([min(labels)-interval/2, max(labels)+interval])
        plt.savefig('{}_temp_split_vis.png'.format(dataset_name),bbox_inches='tight')

    return train_inds, np.sort(valid_inds, axis=None), np.sort(test_inds, axis=None)