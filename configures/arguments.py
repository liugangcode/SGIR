import yaml
import argparse

def load_arguments_from_yaml(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_args():
    parser = argparse.ArgumentParser(description='[KDD 2023] Semi-supervised Learning for Imbalace Graph Regression.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--dataset', default="plym-glass", type=str,
                        choices=['plym-density', 'plym-oxygen', 'plym-melting', 
                                'ogbg-mollipo', 'ogbg-molfreesolv', 'ogbg-molesol'],
                        help='dataset name (default: plym-, ogbg-)')
    parser.add_argument('--bin-base', type=float, default=0.,
                        help='For evaluation. log base of bin to divide the label interval. 0 indicates no use')
    parser.add_argument('--bw', type=float, default=0.,
                        help='For evaluation. bin width to divide the label interval. 0 indicates no use.')
    parser.add_argument('--no-print', action='store_true', default=False,
                        help="don't use progress bar")
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--logname', type=str, default='',
                        help="output log file name")
    parser.add_argument('--many-threshold', type=float, default=2, 
                        help='many shot region >=  max_freq/many_threshold')
    parser.add_argument('--medium-threshold', type=float, default=5, 
                        help='few-short range <  max_freq/medium_threshold; \
                              max_freq/medium_threshold <= medium-short range < max_freq/many_threshold')

    # model
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop-ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num-layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb-dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--gamma', default=0.4, type=float,
                        help='gamma for rationales')
    parser.add_argument('--lw-Rreg', default=0.1, type=float,
                        help='loss weight for rationale regularization')
    parser.add_argument('--lw-xenvs', default=1, type=float,
                        help='loss weight for envrioment augmentation loss')
    parser.add_argument('--temperature', default=2, type=float,
                        help='temperature for GREA reweighting')    

    # training
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--initw-name', type=str, default='default',
                        choices=['default','orthogonal','normal','xavier','kaiming'],
                        help='method name to initialize neural weights')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stop (default: 50)')
    parser.add_argument('--warmup-scheduler', default=0, type=float,
                        help='scheduler warmup epochs')
    parser.add_argument('--trails', type=int, default=5,
                        help='nubmer of experiments')
    parser.add_argument('--eval-metric', type=str, default='mae',
                        choices=['mae','rmse','gm'],
                        help='regression evaluation metrics used for early stopping (default: mae)')     
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-2,
                        help='Learning rate (default: 1e-2)')
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train (default: 400)')

    # self-training: augmentation
    parser.add_argument('--update-aug', default=50, type=int,
                        help='number of supervised epoch before update the augmentation')
    parser.add_argument('--anchor-aug', default=100, type=int,
                        help='number of anchors in the label space for augmentation')
    parser.add_argument('--lw-aug', default=1, type=float,
                        help='loss weight for imbalance augmentation')

    # self-training: selection
    parser.add_argument('--update-select', default=30, type=int,
                        help='number of supervised epoch before update the augmentation')
    parser.add_argument('--anchor-select', default=50, type=int,
                        help='number of anchors in the label space for selection')
    parser.add_argument('--var-threshold', default=0.1, type=float,
                        help='variance/uncertainty threshold for unlabeled selection')
    

    args = parser.parse_args()
        
    if args.bin_base == 0:
        args.bin_base = None

    ### only used for interval masking and evaluation
    ### should be fixed for all methods
    if args.bw == 0:
        prop_name = args.dataset.split('-')[1]
        if prop_name in ['oxygen']:
            args.bw = 0.2
            args.bin_base = 10
        elif prop_name in ['density']:
            args.bw = 0.02
            args.many_threshold = 1.5
            args.medium_threshold = 3
        elif prop_name in ['melting']:
            args.bw = 10
            args.many_threshold = 1.5
            args.medium_threshold = 3
        elif prop_name in ['molesol']:
            args.bw = 0.1
            args.many_threshold = 1.5
            args.medium_threshold = 2
        elif prop_name in ['molfreesolv']:
            args.bw = 0.2
            args.many_threshold = 1.5
            args.medium_threshold = 3
        elif prop_name in ['mollipo']:
            args.bw = 0.05
        else:
            args.bw = 1

    return args
