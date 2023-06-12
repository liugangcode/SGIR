import torch
from torch_geometric.data import Data
from torch.distributions.beta import Beta
from torch_geometric.loader import DataLoader

__all__ = ['build_augment_dataset', 'build_selection_dataset']

def build_selection_dataset(args, model, dataset):
    label_split_idx = dataset.get_idx_split(split_type = 'balance', regenerate=False)
    labeled_set = dataset[label_split_idx["train"]]

    unlabeled_set = dataset[dataset.get_unlabeled_idx()]
    unlabel_idx = torch.arange(len(unlabeled_set))

    unlabeled_trainloader = DataLoader(
        unlabeled_set, 
        batch_size= args.batch_size, 
        shuffle=False, 
        num_workers = args.num_workers)
    labeled_trainloader = DataLoader(
        labeled_set, 
        batch_size= args.batch_size, 
        shuffle=False, 
        num_workers = args.num_workers)
    
    model.eval()
    unlabeled_pred = []
    unlabeled_env_vars = []

    for step, batch in enumerate(unlabeled_trainloader):
        batch = batch.to(args.device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
                pred_rep = pred['pred_rep']
                pred = pred['pred_rem']
                var = pred_rep.view(-1).view(-1, batch.batch[-1]+1).var(dim=1)
                unlabeled_env_vars.append(var.detach())
                unlabeled_pred.append(pred.detach())

    unlabeled_pred = torch.cat(unlabeled_pred, dim = 0).cpu().view(-1)
    unlabeled_env_vars = torch.cat(unlabeled_env_vars, dim = 0).cpu().view(-1)
    labeled_true = dataset.data.y[label_split_idx["train"]].cpu().view(-1)

    var_asc_idx = torch.argsort(unlabeled_env_vars)
    unlabeled_env_vars = unlabeled_env_vars[var_asc_idx]
    unlabeled_pred = unlabeled_pred[var_asc_idx]
    unlabel_idx = unlabel_idx[var_asc_idx]

    labeled_env_vars = []
    for step, batch_labeled in enumerate(labeled_trainloader):
        batch_labeled = batch_labeled.to(args.device)
        if batch_labeled.x.shape[0] != 1:
            with torch.no_grad():
                pred = model(batch_labeled)
                pred_rep = pred['pred_rep']
                var = pred_rep.view(-1).view(-1, batch_labeled.batch[-1]+1).var(dim=1)
                labeled_env_vars.append(var.detach())
    labeled_env_vars = torch.cat(labeled_env_vars, dim = 0).cpu().view(-1)

    uncertainty_masks = unlabeled_env_vars <= torch.quantile(labeled_env_vars, args.var_threshold)
    unlabeled_pred = unlabeled_pred[uncertainty_masks]
    unlabel_idx = unlabel_idx[uncertainty_masks]

    start, end = labeled_true.min(), labeled_true.max()
    anchor_num = args.anchor_select
    if args.dataset == 'plym-oxygen':
        boundaries = torch.linspace(torch.log10(start), torch.log10(end), steps=anchor_num+1)
        boundaries = torch.pow(10, boundaries)
    elif args.dataset == 'plym-density':
        end = torch.topk(labeled_true.view(-1), 2).values[1] # remove the outlier
        boundaries = torch.linspace(start, end, steps=anchor_num+1)
    else:
        boundaries = torch.linspace(start, end, steps=anchor_num+1)

    bucket_id_per_label = torch.bucketize(labeled_true.view(-1), boundaries)
    bucket_id_per_label = torch.clamp(bucket_id_per_label, min=1, max=len(boundaries)-1)
    unique_train_buckets, count_per_bucket = torch.unique(bucket_id_per_label, sorted=True, return_counts=True)
    bucket_center = 1/2 * (boundaries[unique_train_buckets-1] + boundaries[unique_train_buckets])
        
    def get_sample_probs(counts):
        idx = torch.argsort(counts.view(-1), descending=False)
        sample_rate = counts[torch.flipud(idx)] / counts.max()
        probs = torch.zeros_like(sample_rate)
        probs[idx] = sample_rate
        return probs

    sampling_probs = get_sample_probs(count_per_bucket)
    new_labeled_idx = []
    new_labeled_y = []
    sample_record = {}

    sample_count = 0
    for idx, anchor in enumerate(bucket_center):
        upper_idx = (boundaries>anchor).nonzero().min()
        lower_idx = (boundaries<anchor).nonzero().max()
        width = boundaries[upper_idx] - boundaries[lower_idx]
        range_min = anchor - width / 2
        range_max = anchor + width / 2
        valid_mask = torch.logical_and(unlabeled_pred>=range_min, unlabeled_pred<range_max)
        num_picked = (valid_mask.sum() * torch.pow(sampling_probs[idx], 1)).to(torch.int).item()
        num_picked = min(num_picked, count_per_bucket.max())

        if num_picked > 0:
            label_dist = torch.abs(anchor - unlabeled_pred[valid_mask])
            inds_after_mask_asc = torch.argsort(label_dist, descending=False)
            new_labeled_idx.append(unlabel_idx[valid_mask][inds_after_mask_asc][:num_picked])
            cur_sample_len = (unlabel_idx[valid_mask][inds_after_mask_asc][:num_picked]).size(0)
            new_labeled_y.append(torch.ones(cur_sample_len)*anchor)
            sample_record[anchor.item()] = cur_sample_len
            sample_count += cur_sample_len

    if len(new_labeled_idx) != 0:
        new_labeled_idx = torch.cat(new_labeled_idx, dim=0)
        new_labeled_y = torch.cat(new_labeled_y, dim=0)

    new_trainset = []
    def get_datapoint_list(subset_data, unlabeled=True):
        datapoint_list = []
        ys = []
        for idx, datapoint in enumerate(subset_data):
            g = Data()
            g.edge_index = datapoint.edge_index
            if not ('image' in args.dataset):
                g.edge_attr = datapoint.edge_attr
            g.x = datapoint.x
            if unlabeled:
                g.y = torch.tensor([[new_labeled_y[idx]]])
            else:
                g.y = datapoint.y
            ys.append(g.y.item())
            datapoint_list.append(g)
        return datapoint_list, ys

    if len(new_labeled_idx) != 0:
        selected_trainset = get_datapoint_list(unlabeled_set[new_labeled_idx])[0]
        new_trainset.extend(selected_trainset)
    else:
        selected_trainset = None
    new_trainset.extend(get_datapoint_list(labeled_set, unlabeled=False)[0])

    new_trainloader = DataLoader(
        new_trainset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers = args.num_workers)

    return new_trainloader

def build_augment_dataset(args, model, dataset):
    label_split_idx = dataset.get_idx_split(split_type = 'balance', regenerate=False)
    labeled_set = dataset[label_split_idx["train"]]
    unlabeled_set = dataset[dataset.get_unlabeled_idx()]
    labeled_trainloader = DataLoader(
        labeled_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers = args.num_workers)

    unlabeled_trainloader = DataLoader(
        unlabeled_set, 
        batch_size= args.batch_size, 
        shuffle=False, 
        num_workers = args.num_workers)
    
    labeled_true, labeled_reps = [], []
    all_reps, all_labels = [], [] 
    cenrRep_labels, cenRep_reps = [], [] 

    model.eval()
    for step, batch in enumerate(labeled_trainloader):
        batch = batch.to(args.device)
        if batch.x.shape[0] != 1:
            with torch.no_grad():
                output = model(batch)
                labeled_p = output['pred_rem']
                labeled_h = output['reps']
            labeled_true.append(batch.y.view(labeled_p.shape).detach().to(torch.float32))
            labeled_reps.append(labeled_h.detach())

    labeled_true = torch.cat(labeled_true, dim = 0)
    labeled_reps = torch.cat(labeled_reps, dim = 0)
    all_labels, all_reps = torch.clone(labeled_true), torch.clone(labeled_reps)
    cenrRep_labels, cenRep_reps = torch.clone(labeled_true), torch.clone(labeled_reps)

    unlabeled_pred = []
    unlabeled_reps = []
    for step, batch in enumerate(unlabeled_trainloader):
        batch = batch.to(args.device)
        if batch.x.shape[0] != 1:
            with torch.no_grad():
                output = model(batch)
                unlabeled_p = output['pred_rem']
                unlabeled_h = output['reps']
            unlabeled_pred.append(unlabeled_p.detach()) 
            unlabeled_reps.append(unlabeled_h.detach())
    unlabeled_pred = torch.cat(unlabeled_pred, dim = 0)
    unlabeled_reps = torch.cat(unlabeled_reps, dim = 0)
    all_reps = torch.cat([all_reps, unlabeled_reps], dim=0)
    all_labels = torch.cat([all_labels, unlabeled_pred], dim=0)

    start, end = labeled_true.min(), labeled_true.max()
    anchor_num = args.anchor_aug

    if args.dataset == 'plym-oxygen':
        boundaries = torch.linspace(torch.log10(start), torch.log10(end), steps=anchor_num+1)
        boundaries = torch.pow(10, boundaries)
    elif args.dataset == 'plym-density':
        end = torch.topk(labeled_true.view(-1), 2).values[1]
        boundaries = torch.linspace(start, end, steps=anchor_num+1)
    else:
        boundaries = torch.linspace(start, end, steps=anchor_num+1)
    boundaries = boundaries.to(args.device)
    bucket_id_per_label = torch.bucketize(cenrRep_labels.view(-1), boundaries)
    bucket_id_per_label = torch.clamp(bucket_id_per_label, min=1, max=len(boundaries)-1)
    uni_buckets = torch.unique(bucket_id_per_label, sorted=True)
    bucket_center = 1/2*(boundaries[uni_buckets-1] + boundaries[uni_buckets])

    
    def mean_by_groups(sample_rep, groups):
        ''' select mean(samples), count() from samples group by labels order by labels asc '''
        weight = torch.zeros(groups.max()+1, sample_rep.shape[0]).to(sample_rep.device) # L, N
        weight[groups, torch.arange(sample_rep.shape[0])] = 1
        group_count = weight.sum(dim=1)
        weight = torch.nn.functional.normalize(weight, p=1, dim=1) # l1 normalization
        mean = torch.mm(weight, sample_rep) # L, F
        index = torch.arange(mean.shape[0]).to(mean.device)[group_count > 0]
        return mean[index], group_count[index]
   
    bucket_rep, bucket_count = mean_by_groups(cenRep_reps, bucket_id_per_label-1)

    buckets_preds_dist = torch.cdist(bucket_center.view(-1,1), all_labels.view(-1,1), p=2)
    rank_per_buckets = torch.argsort(buckets_preds_dist, dim=1, descending=False)
    
    def get_sample_probs(counts):
        idx = torch.argsort(counts.view(-1), descending=False)
        sample_rate = counts[torch.flipud(idx)] / counts.max()
        probs = torch.zeros_like(sample_rate)
        probs[idx] = sample_rate
        return probs
    sampling_probs = get_sample_probs(bucket_count)
    sample_to_compensate = torch.ceil(sampling_probs * bucket_count.max()).to(torch.int)

    sample_to_compensate = torch.clamp(sample_to_compensate, max=min(rank_per_buckets.size(1), 100))
    max_sample_num = max(sample_to_compensate)
    sample_inds = rank_per_buckets[:, :max_sample_num].contiguous().view(-1)
    # mixup
    m = Beta(torch.tensor([5.]).to(args.device), torch.tensor([1.]).to(args.device))
    lam = m.sample((bucket_rep.size(0),)).view(-1,1)
    lam = torch.max(torch.cat([lam, 1-lam], dim=1), dim=1).values

    mixed_buckets = lam.view(-1,1,1) * bucket_rep.unsqueeze(1) + (1.0 - lam).view(-1,1,1) * all_reps[sample_inds].view(bucket_rep.size(0), max_sample_num, bucket_rep.size(1))
    mixed_buckets = mixed_buckets.view(-1, bucket_rep.size(1))
    mixed_labels = lam.view(-1,1,1) * bucket_center.view(-1,1).unsqueeze(1) + (1.0 - lam).view(-1,1,1) * all_labels[sample_inds].view(bucket_center.size(0), max_sample_num, 1)
    mixed_labels = mixed_labels.view(-1, 1)

    augmented_reps = []
    augmented_labels = []
    for idx in range(bucket_rep.size(0)):
        current_sample_num = sample_to_compensate[idx]
        augmented_reps.append(mixed_buckets[idx * max_sample_num:idx * max_sample_num + current_sample_num, :])
        augmented_labels.append(mixed_labels[idx * max_sample_num:idx * max_sample_num + current_sample_num, :])

    augmented_reps = torch.cat(augmented_reps, dim=0)
    augmented_labels = torch.cat(augmented_labels, dim=0)

    return augmented_reps, augmented_labels


