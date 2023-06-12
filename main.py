import os
import time
import logging
from tqdm import tqdm
from datetime import datetime

import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from utils import weighted_l1_loss
from dataset.get_datasets import get_dataset
from utils import AverageMeter, validate, print_info, IntervalMasker, init_weights
from utils import build_augment_dataset, build_selection_dataset
from configures.arguments import load_arguments_from_yaml, get_args

reg_criterion = weighted_l1_loss

def get_logger(name, logfile=None):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
    return logger

def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        # return max(0., math.cos(math.pi * num_cycles * no_progress))
        return max(0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def train(args, model, train_loaders, optimizer, scheduler, epoch):
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_xaug = AverageMeter()
    device = args.device
    
    model.train()
    if train_loaders['augmented_reps'] is not None and train_loaders['augmented_labels'] is not None and args.lw_aug != 0:
        aug_reps = train_loaders['augmented_reps']
        aug_targets = train_loaders['augmented_labels']
        random_inds = torch.randperm(aug_reps.size(0))
        aug_reps = aug_reps[random_inds]
        aug_targets = aug_targets[random_inds]
        aug_batch_size = aug_reps.size(0) // args.steps
        aug_inputs = list(torch.split(aug_reps, aug_batch_size))
        aug_outputs = list(torch.split(aug_targets, aug_batch_size))
    else:
        aug_inputs = None
        aug_outputs = None

    for batch_idx in range(args.steps):
        end = time.time()
        model.zero_grad()
        
        ### augmentation loss
        if aug_inputs is not None and aug_outputs is not None and aug_inputs[batch_idx].size(0) != 1:
            model._disable_batchnorm_tracking(model)
            pred_aug = model.predictor(aug_inputs[batch_idx])
            model._enable_batchnorm_tracking(model)
            targets_aug = aug_outputs[batch_idx]
            Laug = reg_criterion(pred_aug.view(targets_aug.size()).to(torch.float32), targets_aug, weights=None)
            Laug = Laug.mean()
        else:
            Laug = torch.tensor(0.)            
        

        ### labeled loss
        try:
            batch_labeled = train_loaders['labeled_iter'].next()
        except:
            train_loaders['labeled_iter'] = iter(train_loaders['labeled_trainloader'])
            batch_labeled = train_loaders['labeled_iter'].next()
        batch_labeled =  batch_labeled.to(device)
        targets = batch_labeled.y.to(torch.float32)

        if batch_labeled.x.shape[0] == 1 or batch_labeled.batch[-1] == 0:
            continue
        else:
            output = model(batch_labeled)
            pred_labeled, pred_rep = output['pred_rem'], output['pred_rep']

            Losses_x = reg_criterion(pred_labeled.view(targets.size()).to(torch.float32), targets)
            Lx = Losses_x.mean()

            Lx += output['loss_reg'] * args.lw_Rreg
            
            target_rep = targets.repeat_interleave(batch_labeled.batch[-1]+1,dim=0)
            losses_xrep_envs = reg_criterion(pred_rep.view(target_rep.size()).to(torch.float32), target_rep)

            losses_xrep_envs = losses_xrep_envs.view(-1).view(-1,batch_labeled.batch[-1]+1)
            losses_xrep_var, losses_xrep_mean = losses_xrep_envs.var(dim=1), losses_xrep_envs.mean(dim=1)

            pos_w = F.normalize(torch.abs(targets.view(-1,1) - targets.view(1,-1)).mean(dim=1) / args.temperature, dim=0).softmax(dim=0)
            Lx += args.lw_xenvs * torch.matmul(losses_xrep_var, pos_w)
            Lx += args.lw_xenvs * torch.matmul(losses_xrep_mean, pos_w)

        loss = Lx + Laug * args.lw_aug
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_xaug.update(Laug.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if not args.no_print:
            # print('scheduler.get_last_lr()[0]', scheduler.get_last_lr()[0])
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_xaug: {losses_xaug:.4f}. ".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.steps,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                losses_xaug=losses_xaug.avg,
                ))
            p_bar.update()
    if not args.no_print:
        p_bar.close()

    return train_loaders


def main(args):
    def create_model(args):
        from models.grea import GraphEnvAug
        model = GraphEnvAug(gnn_type = args.gnn, num_tasks = dataset.num_tasks, num_layer = args.num_layer,
                            emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma = args.gamma).to(device)
        init_weights(model, args.initw_name, init_gain=0.02)   
        return model
    
    device = torch.device('cuda', args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    os.makedirs(args.out, exist_ok=True)

    dataset = get_dataset(args, './raw_data')
    label_split_idx = dataset.get_idx_split(split_type = 'balance', regenerate=False)

    args.num_unlabeled = dataset.unlabeled_data_len
    args.num_labeled = dataset.labeled_data_len
    args.num_trained = len(label_split_idx["train"])

    interval_masker = IntervalMasker(
        args.dataset, 
        dataset.data.y[label_split_idx["train"]],
        base=args.bin_base,
        bin_width=args.bw, 
        medium_t=args.medium_threshold,
        many_t=args.many_threshold)


    labeled_trainloader = DataLoader(
        dataset[label_split_idx["train"]], 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers = args.num_workers)
    
    valid_loader = DataLoader(
        dataset[label_split_idx["valid"]],
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers = args.num_workers)
    
    test_loader = DataLoader(
        dataset[label_split_idx["test"]], 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers = args.num_workers)
    

    model = create_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    args.steps = args.num_trained // args.batch_size + 1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_scheduler, args.epochs * args.steps)

    logging.warning(
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        )
    logger.info(dict(args._get_kwargs()))
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_trained}/{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.epochs * args.steps}")
    logger.info(f"  Evaluation metric = {args.eval_metric}")

    labeled_iter = iter(labeled_trainloader)

    augmented_reps, augmented_labels = None, None
    train_loaders = {
        'labeled_iter': labeled_iter,
        'labeled_trainloader': labeled_trainloader,
        'augmented_reps': augmented_reps,
        'augmented_labels': augmented_labels,
    }

    for epoch in range(0, args.epochs):
        train_loaders = train(args, model, train_loaders, optimizer, scheduler, epoch)

        if epoch >= 50 and epoch % args.update_select == 0:
            new_trainloader = build_selection_dataset(args, model, dataset)
            train_loaders['labeled_trainloader'] = new_trainloader
            train_loaders['labeled_iter'] = iter(new_trainloader)
            args.num_trained = len(new_trainloader.dataset)
            args.steps = args.num_trained // args.batch_size + 1

        if epoch >= 50 and epoch % args.update_aug == 0:
            augmented_reps, augmented_labels = build_augment_dataset(args, model, dataset)
            train_loaders['augmented_reps'] = augmented_reps
            train_loaders['augmented_labels'] = augmented_labels

        train_perf = validate(args, model, labeled_trainloader, interval_masker)
        valid_perf = validate(args, model, valid_loader, interval_masker)

        update_test = False
        if epoch != 0 and valid_perf[args.eval_metric]['all'] < best_valid_perf[args.eval_metric]['all']:
            update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            best_train_perf = train_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perf = validate(args, model, test_loader, interval_masker)

            if not args.no_print:
                print_info('Train', train_perf)
                print_info('Valid', valid_perf)
                print_info('Test', test_perf)
        else:
            # not update
            if not args.no_print:
                print_info('Train', train_perf)
                print_info('Valid', valid_perf)
            if epoch > 200: 
                cnt_wait += 1
                if cnt_wait > args.patience:
                    break
    print('Finished training! Best validation results from epoch {}.'.format(best_epoch))
    # print_info('train', best_train_perf)
    # print_info('valid', best_valid_perf)
    # print_info('test', test_perf)

    return best_train_perf, best_valid_perf, test_perf

if __name__ == '__main__':
    args = get_args()
    config = load_arguments_from_yaml(f'configures/{args.dataset}.yaml')
    for arg, value in config.items():
        setattr(args, arg, value)

    datetime_now = datetime.now().strftime("%Y%m%d.%H%M%S")
    if args.logname != '':
        fname = f'{args.dataset.replace("-", "_")}-training-{args.logname}-{datetime_now}.log'
        logfile = os.path.join(args.out, fname)
    else:
        logfile = None

    logger = get_logger(__name__, logfile=logfile)

    val_results = dict()
    test_results = dict()
    print(args)
    for exp_num in range(args.trails):
        seed_torch(exp_num)
        args.exp_num = exp_num
        train_perf, valid_perf, test_perf = main(args)
        if exp_num ==0:
            for mode in valid_perf.keys(): 
                val_results[mode] = dict()
                test_results[mode] = dict()
                for region, value in valid_perf[mode].items():
                    if region != 'Metric':
                        val_results[mode][region] = []
                        test_results[mode][region] = []
        
        for mode in valid_perf.keys():
            for region in val_results[mode].keys():
                val_results[mode][region].append(valid_perf[mode][region])
                test_results[mode][region].append(test_perf[mode][region])  

        for mode in val_results.keys():
            for region, nums in val_results[mode].items():
                logger.info('val {:<5} {:<5}\t: {:.3f}+-{:.4f} {}'.format(
                    mode, region, np.mean(nums), np.std(nums), nums))
        for mode in test_results.keys():
            for region, nums in test_results[mode].items():
                logger.info('test {:<5} {:<5}\t: {:.3f}+-{:.4f} {}'.format(
                    mode, region, np.mean(nums), np.std(nums), nums))
    
    for mode in test_results.keys():
        output_str = ''
        logger.info('-'*10 + '{}\n'.format(mode))
        for region, nums in test_results[mode].items():
            if args.dataset == 'plym-density':
                output_str += '{:>10}: {:.3f}+-{:.3f}\t'.format(region, np.mean(np.array(nums)*1000), np.std(np.array(nums)*1000))
            else:
                output_str += '{:>10}: {:.3f}+-{:.3f}\t'.format(region, np.mean(nums), np.std(nums))
            
        logger.info(output_str)