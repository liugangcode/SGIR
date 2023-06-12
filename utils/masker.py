import logging
import numpy as np
__all__ = ['IntervalMasker']


logger = logging.getLogger(__name__)

def log_base(base, x):
    return np.log(x) / np.log(base)

class IntervalMasker(object):
    def __init__(self, dataset, labels, base=None, max_target=None, bin_width=None, medium_t=None, many_t=None):
        self.dataset = dataset
        self.labels = labels.numpy().reshape(-1)
        self.base = base
        self.interval = bin_width
        self.max_target = max_target
        
        self.bins = None
        self.train_label_freqs = None
        if medium_t is None:
            self.medium_t = 10
        else:
            self.medium_t = medium_t
        if many_t is None:
            self.many_t = 2
        else:
            self.many_t = many_t
        self._intialization()

    def get_region_masks(self, labels):
        assert self.train_label_freqs is not None, 'density missing'
        assert self.bins is not None, 'bins missing'

        region_bins = self.bins
        training_labels = self.labels
        if self.base != None:
            labels = log_base(self.base, labels)
            training_labels = log_base(self.base, training_labels)
        _, region_u_inds, bin_counts = self._assign_label_to_bins(training_labels, region_bins)
        region_freqs = np.zeros(region_bins.shape)
        region_freqs[region_u_inds] = bin_counts
        freqs4labels = np.zeros(labels.shape)
        in_train_region_mask = np.logical_and(labels>=min(region_bins), labels<max(region_bins))
        inds4labels = np.digitize(labels[in_train_region_mask], region_bins)
        freqs4labels[in_train_region_mask] = region_freqs[inds4labels]
        max_freq = max(region_freqs)
        many_mask = freqs4labels >= max_freq/self.many_t
        medium_mask = np.logical_and(freqs4labels>=max_freq/self.medium_t, freqs4labels<max_freq/self.many_t)
        few_mask = np.logical_and(freqs4labels>0, freqs4labels<max_freq/self.medium_t)
        zero_mask = freqs4labels == 0
        return many_mask, medium_mask, few_mask, zero_mask
    
    def _intialization(self):
        prop_name = self.dataset.split('-')[1]
        if self.interval is None:
            if prop_name in ['oxygen']:
                interval = 1
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
            self.interval = interval

        labels = self.labels
        if self.base != None:
            labels = log_base(self.base, labels)
        bins = self._get_bins()
        inds, u_inds, counts = self._assign_label_to_bins(labels, bins)
        freqs = np.zeros(bins.shape)
        freqs[u_inds] = counts
        self.train_label_freqs = freqs
        self.bins = bins

    def get_train_freqs(self, labels):
        labels = labels.reshape(-1)
        if self.base is not None:
            labels = log_base(self.base, labels)
        inds = np.digitize(labels, self.bins)
        assert self.train_label_freqs is not None, 'density missing'
        return self.train_label_freqs[inds]
    
    def _get_bins(self):
        training_labels = self.labels
        if self.base is not None:
            training_labels = log_base(self.base, training_labels)
        interval = self.interval
        max_label_value = np.ceil(max(training_labels)) + interval
        min_label_value = np.floor(min(training_labels)) - interval / 2
        if self.max_target is not None:
            max_label_value = self.max_target
        bins = np.arange(min_label_value, max_label_value, interval) #[start, end)
        return bins
    
    def _assign_label_to_bins(self, labels, bins):
        # Return the indices of the bins to which each value in input array belongs: return satisfying i: bins[i-1] <= x < bins[i]
        inds = np.digitize(labels, bins)
        u_inds, counts = np.unique(inds, return_counts=True)
        return inds, u_inds, counts
