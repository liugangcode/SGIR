from .polymer import PolymerRegDataset
from .ogbg import PygGraphPropPredDataset

DATASET_GETTERS = {
    'plym-oxygen': PolymerRegDataset,
    'plym-density': PolymerRegDataset,
    'plym-melting': PolymerRegDataset,
    'ogbg-molesol': PygGraphPropPredDataset,
    'ogbg-molfreesolv': PygGraphPropPredDataset,
    'ogbg-mollipo': PygGraphPropPredDataset
    }

def get_dataset(args, load_path):
    return DATASET_GETTERS[args.dataset](args.dataset, load_path)