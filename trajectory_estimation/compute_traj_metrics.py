# Compare saved trajectory with ground-truth and compute evaluation metrics.

import os
import sys
import glob
import numpy as np
import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.ntp_utils import *

def traj_metrics_without_table(traj, labels, traj_mask):
    traj = torch.from_numpy(traj)
    labels = torch.from_numpy(labels)
    traj_mask = torch.from_numpy(traj_mask)
        
    # ANCHOR: prepare for the trajectories with masks, reshape to (NxW)x3
    pred_traj_masked = traj * (traj_mask.unsqueeze(-1))
    gt_traj_masked = labels * (traj_mask.unsqueeze(-1))
    
    # ANCHOR: absolute mean traj error, should use all to compute mean
    rmse_mean = torch.sum((pred_traj_masked - gt_traj_masked) ** 2, -1)   # Nxtraj_len

    # ANCHOR: absolute median traj error, should use all to compute median
    mae_median = torch.sum(torch.abs(pred_traj_masked - gt_traj_masked), -1)   # Nxtraj_len
    
    # ANCHOR: trajectory rotation error
    unit_label = gt_traj_masked / gt_traj_masked.norm(dim=2, keepdim=True)   # shape: NxWx3
    unit_pred = pred_traj_masked / pred_traj_masked.norm(dim=2, keepdim=True)   # shape: NxWx3
    eps = 1e-7
    dot_product = (unit_label * unit_pred).sum(-1).clamp(min=-1+eps, max=1-eps)   # shape: NxW
    dot_product[dot_product != dot_product] = 0  # Remove NaNs
    angle_error = torch.mean(torch.acos(dot_product), 0)
    
    # ANCHOR: ground truth relative errors
    l2_norm = torch.sqrt(torch.sum((pred_traj_masked - gt_traj_masked) ** 2, -1)).cpu()   # Absolute distance error, NxW
    labels_norm = torch.sqrt(torch.sum(gt_traj_masked ** gt_traj_masked, -1)).cpu()   # NxW
    relative_err = l2_norm / (labels_norm + 1e-20)
    
    # ANCHOR: ACC_5
    error_lt_5 = torch.BoolTensor((l2_norm < 0.5))
    relative_err_lt_5 = torch.BoolTensor((relative_err < 0.05))
    acc3d_strict = torch.mean((error_lt_5 | relative_err_lt_5).float(), 0)
    
    # ANCHOR: ACC_10
    error_lt_10 = torch.BoolTensor((l2_norm < 1))
    relative_err_lt_10 = torch.BoolTensor((relative_err < 0.1))
    acc3d_relax = torch.mean((error_lt_10 | relative_err_lt_10).float(), 0)
    
    # ANCHOR: outliers
    l2_norm_gt_3 = torch.BoolTensor(l2_norm > 3)
    relative_err_gt_10 = torch.BoolTensor(relative_err > 0.1)
    outlier = torch.mean((l2_norm_gt_3 | relative_err_gt_10).float(), 0)
    
    return rmse_mean, mae_median, angle_error, acc3d_strict, acc3d_relax, outlier


def load_test_traj(traj_path, options, gt_data=None):
    if os.path.exists(traj_path + '/traj.npy'):
        # For outputs saved by foward-Euler integration.
        test_traj = np.load(traj_path + '/traj.npy')[:,:options.traj_len,:]
        
    elif os.path.exists(traj_path + '/traj_field_model.pth'):
        # For outputs saved by trajecotry field estimation.
        traj_field = load_pretrained_traj_field(os.path.join(traj_path, 'traj_field_model.pth'), options.traj_len, options)
        traj_field.cuda()
        data = np.load(gt_data, allow_pickle=True)
        test_traj = get_traj_field(data['pcs'][0], 0, traj_field) 
        test_traj = test_traj.cpu().numpy()
    
    else:
        print('Required trajectory/model files not found.')

    return test_traj
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute trajectory metrics.")
    parser.add_argument('--exp_name', type=str, default='fit_NTP_base_test', metavar='N', help='Name of the experiment.')
    parser.add_argument('--dataset_path', type=str, default='/mnt/088A6CBB8A6CA742/av1/av1_traj')

    # ANCHOR: settings for trajectory
    parser.add_argument('--hidden_units', type=int, default=128, metavar='N', help='Number of hidden units in neural prior')
    parser.add_argument('--act_fn', type=str, default='relu', metavar='AF', help='activation function for neural prior.')
    parser.add_argument('--traj_len', type=int, default=25, help='point cloud sequence length for the trajectory.')
    parser.add_argument('--traj_type', type=str, default='velocity', help='trajectory decoder type')
    parser.add_argument('--st_embed_type', type=str, default='cosine', help='type of spatial temporal embeddings')
    options = parser.parse_args()
    traj_len = options.traj_len

    test_fi_name = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '../', f"checkpoints/{options.exp_name}/*/")))
    gt_fi_name = sorted(glob.glob(os.path.join(options.dataset_path, '*.npz')))
    
    rmse_mean_list = []
    mae_median_list = []
    angle_error_list = []
    acc3d_strict_list = []
    acc3d_relax_list = []
    outlier_list = []
    for i in range(len(test_fi_name)):
        assert test_fi_name[i].split('/')[-2] == gt_fi_name[i].split('/')[-1][:-4]
        
        test_traj = load_test_traj(test_fi_name[i], options, gt_data=gt_fi_name[i])
        gt_fi = np.load(gt_fi_name[i])
        gt_traj = gt_fi['traj'][:,:traj_len,:]
        gt_traj_mask = gt_fi['traj_val_mask'][:,:traj_len]
        
        rmse_mean, mae_median, angle_error, acc3d_strict, acc3d_relax, outlier = traj_metrics_without_table(test_traj, gt_traj, gt_traj_mask)
        
        rmse_mean_list.append(rmse_mean)
        mae_median_list.append(mae_median)
        angle_error_list.append(angle_error)
        acc3d_strict_list.append(acc3d_strict)
        acc3d_relax_list.append(acc3d_relax)
        outlier_list.append(outlier)
  
    rmse_mean_list = torch.cat(rmse_mean_list, 0)
    mae_median_list = torch.cat(mae_median_list, 0)
    print(rmse_mean_list.shape, mae_median_list.shape)   # N x traj_len
    rmse_mean_list_mean = torch.sqrt(torch.mean(rmse_mean_list, 0))   # traj_len
    mae_median_list_mean, _ = torch.median(mae_median_list, 0)   # traj_len
    
    angle_error_list = torch.cat(angle_error_list, 0).reshape(-1, gt_traj.shape[1])
    acc3d_strict_list = torch.cat(acc3d_strict_list, 0).reshape(-1, gt_traj.shape[1])
    acc3d_relax_list = torch.cat(acc3d_relax_list, 0).reshape(-1, gt_traj.shape[1])
    outlier_list = torch.cat(outlier_list, 0).reshape(-1, gt_traj.shape[1])

    acc3d_strict_list_np = acc3d_strict_list.numpy()
    acc3d_relax_list_np = acc3d_relax_list.numpy()
    
    print(rmse_mean_list.shape, mae_median_list.shape, angle_error_list.shape, acc3d_strict_list.shape, acc3d_relax_list.shape, outlier_list.shape)
    
    angle_error_list_mean = torch.mean(angle_error_list, 0)
    acc3d_strict_list_mean = torch.mean(acc3d_strict_list, 0)
    acc3d_relax_list_mean = torch.mean(acc3d_relax_list, 0)
    outlier_list_mean = torch.mean(outlier_list, 0)
    
    print(rmse_mean_list_mean.shape, mae_median_list_mean.shape, angle_error_list_mean.shape, acc3d_strict_list_mean.shape, acc3d_relax_list_mean.shape, outlier_list_mean.shape)
    
    print('rmse mean is {}\n, mae median mean is {}\n, angle error mean is {}\n, acc3d strict mean is {}\n, acc3d relax mean is {}\n, outlier mean is {}\n'.format(rmse_mean_list_mean, mae_median_list_mean, angle_error_list_mean, acc3d_strict_list_mean, acc3d_relax_list_mean, outlier_list_mean))
    
    print(f'Acc(0.5) between 1st and 25th frame: {acc3d_strict_list_mean[-1]}')
    print(f'Acc(1.0) between 1st and 25th frame: {acc3d_relax_list_mean[-1]}')