import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import gmean
import torch

__all__ = ['print_info', 'validate', 'init_weights', 'AverageMeter']

def print_info(set_name, perf):
    output_str = '{} \t\t'.format(set_name)
    for metric_name in ['mae', 'rmse', 'mse', 'gm']:
        output_str += '{} all: {:<10.4f} \t'.format(metric_name, perf[metric_name]['all'])
    print(output_str)

def validate(args, model, loader, reg_evaluator, likelihood=None):
    y_true = []
    y_pred = []
    rmse_dict = {'Metric': 'RMSE'}
    mse_dict = {'Metric': 'MSE'}
    mae_dict = {'Metric': 'MAE'}
    gm_dict = {'Metric': 'GM'}
    device = args.device
    model.eval()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                output = model(batch)
                pred = output['pred_rem']

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy().reshape(-1)
    y_pred = torch.cat(y_pred, dim = 0).numpy().reshape(-1)
    
    rmse_dict['all'] = mean_squared_error(y_true, y_pred, squared=False)
    mse_dict['all'] = mean_squared_error(y_true, y_pred, squared=True)
    mae_dict['all'] = mean_absolute_error(y_true, y_pred)
    gm_dict['all'] = gmean(np.abs(y_true-y_pred), axis=None)
    many_mask, medium_mask, few_mask, zero_mask = reg_evaluator.get_region_masks(y_true)

    if many_mask.sum() != 0:
        rmse_dict['many'] = mean_squared_error(y_true[many_mask], y_pred[many_mask], squared=False)
        mse_dict['many'] = mean_squared_error(y_true[many_mask], y_pred[many_mask], squared=True)
        mae_dict['many'] = mean_absolute_error(y_true[many_mask], y_pred[many_mask])
        gm_dict['many'] = gmean(np.abs(y_true[many_mask]-y_pred[many_mask]), axis=None)
    if medium_mask.sum() != 0:  
        rmse_dict['medium'] = mean_squared_error(y_true[medium_mask], y_pred[medium_mask], squared=False)
        mse_dict['medium'] = mean_squared_error(y_true[medium_mask], y_pred[medium_mask], squared=True)
        mae_dict['medium'] = mean_absolute_error(y_true[medium_mask], y_pred[medium_mask])
        gm_dict['medium'] = gmean(np.abs(y_true[medium_mask]-y_pred[medium_mask]), axis=None)
    if few_mask.sum() != 0:
        rmse_dict['few'] = mean_squared_error(y_true[few_mask], y_pred[few_mask], squared=False)
        mse_dict['few'] = mean_squared_error(y_true[few_mask], y_pred[few_mask], squared=True)
        mae_dict['few'] = mean_absolute_error(y_true[few_mask], y_pred[few_mask])
        gm_dict['few'] = gmean(np.abs(y_true[few_mask]-y_pred[few_mask]), axis=None)
    if zero_mask.sum() != 0:
        rmse_dict['zero'] = mean_squared_error(y_true[zero_mask], y_pred[zero_mask], squared=False)
        mse_dict['zero'] = mean_squared_error(y_true[zero_mask], y_pred[zero_mask], squared=True)
        mae_dict['zero'] = mean_absolute_error(y_true[zero_mask], y_pred[zero_mask])
        gm_dict['zero'] = gmean(np.abs(y_true[zero_mask]-y_pred[zero_mask]), axis=None)

    perf ={
        'rmse': rmse_dict,
        'mse': mse_dict,
        'mae': mae_dict,
        'gm': gm_dict,
    }
    return perf


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
