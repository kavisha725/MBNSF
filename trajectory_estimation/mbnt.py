# Long-term trajectory estimation with MBNT.

import os, glob
import argparse
import logging
import csv
import numpy as np
import torch
import sys
import pytorch3d.loss as p3dloss

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.general_utils import *
from utils.ntp_utils import *
from utils.o3d_uitls import extract_clusters_dbscan
from utils.sc_utils import spatial_consistency_loss

logger = logging.getLogger(__name__)

def total_sc_loss(labels_t, label_ids, pc, pc_defored, d_thresh=0.03, max_points=3000):
    loss_sc = None
    for id in label_ids:
        cluster = pc[labels_t == id]
        cluster_deformed = pc_defored[labels_t == id]
        assert cluster.shape == cluster_deformed.shape
        cluster_cs_loss = spatial_consistency_loss(cluster.unsqueeze(0), cluster_deformed.unsqueeze(0), d_thre=d_thresh, max_points=max_points)
        if not loss_sc:
            loss_sc = cluster_cs_loss
        else: loss_sc += cluster_cs_loss
    loss_sc /= len(label_ids)
    return loss_sc.squeeze()

def fit_trajectory_field(
    exp_dir,
    pc_list,
    options,   
    flow_gt_list = None,
    traj_gt = None,
    traj_val_mask = None
    ):
    
    csv_file = open(f"{exp_dir}/metrics.csv", 'w')
    metric_labels = ['train_loss', 'train_chamfer_loss', 'train_sc_loss', 'train_consist_loss', 'traj_consist', 'epe', 'acc_strict', 'acc_relax', 'angle_error', 'outlier']
    csv_writer = csv.DictWriter(csv_file, ['itr'] + metric_labels + ['traj_metric'])
    csv_writer.writeheader()
    
    n_lidar_sweeps = len(pc_list)
    
    if traj_gt is not None and traj_val_mask is not None:
        traj_gt = torch.from_numpy(traj_gt).cuda()
        traj_val_mask = torch.from_numpy(traj_val_mask).cuda()

    # ANCHOR: Initialize the trajectory field
    net = NeuralTrajField(traj_len=n_lidar_sweeps,
                filter_size=options.hidden_units,
                act_fn=options.act_fn, traj_type=options.traj_type, st_embed_type=options.st_embed_type)
    net.to(options.device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,400,600,800], gamma=0.5)

    # Pre-compute clusters:
    labels_database, label_ids_database = [], []
    for fid in range(n_lidar_sweeps):
        labels = extract_clusters_dbscan(pc_list[fid], eps = options.sc_cluster_eps, min_points=options.sc_cluster_min_points, return_clusters= False, return_colored_pcd=False)
        labels_t = torch.from_numpy(labels).float().clone().to(options.device)
        labels_database.append(labels_t)
        label_ids_database.append(torch.unique(labels_t)[1:]) # Ignore unclustered points with label = -1.
    
    # SECTION: Training steps
    min_loss = 1e10
    for i in range(options.iters):
        # NOTE: Randomly sample  pc pairs
        do_val = np.mod(i, options.traj_len) == 0

        if do_val:
            rnd_ids = range(n_lidar_sweeps)
            cur_metrics = {}
            for label in metric_labels:
                cur_metrics[label] = np.zeros(n_lidar_sweeps-1)
        else:
            rnd_ids = np.random.choice(n_lidar_sweeps, options.traj_batch_size)
        
        # cur_metrics = {}
        for ref_id in rnd_ids:
            post_id = min(n_lidar_sweeps-1, ref_id+1)
            prev_id = max(0, ref_id-1)
            pc_ref = torch.from_numpy(pc_list[ref_id]).cuda()
            
            ref_traj_rt = net(pc_ref, ref_id, do_fwd_flow=True, do_bwd_flow=True, do_full_traj=True)

            pc_ref2post = net.transform_pts(ref_traj_rt['flow_fwd'], pc_ref)
            pc_ref2prev = net.transform_pts(ref_traj_rt['flow_bwd'], pc_ref)

            pc_prev = torch.from_numpy(pc_list[prev_id]).cuda()
            pc_post = torch.from_numpy(pc_list[post_id]).cuda()
            
            loss_chamfer_ref2prev, _ = my_chamfer_fn(pc_prev.unsqueeze(0), pc_ref2prev.unsqueeze(0), None, None)
            loss_chamfer_ref2post, _ = my_chamfer_fn(pc_post.unsqueeze(0), pc_ref2post.unsqueeze(0), None, None)
            loss_chamfer = loss_chamfer_ref2prev + loss_chamfer_ref2post

        
            # NOTE: Spatial-Consistency loss
            labels_t = labels_database[ref_id]
            label_ids = label_ids_database[ref_id]
            
            loss_sc = torch.zeros([], dtype=pc_ref.dtype, device=pc_ref.device,)
            loss_sc_ref2prev = torch.zeros([], dtype=pc_ref.dtype, device=pc_ref.device,)
            loss_sc_ref2post = torch.zeros([], dtype=pc_ref.dtype, device=pc_ref.device,)

            flow_ref2first = net.extract_flow(ref_id, 0, ref_traj_rt['traj'])
            pc_ref2first = net.transform_pts(flow_ref2first, pc_ref)
            dist_thresh_ref2first = options.sc_d_thresh_min + options.sc_d_thresh_max*((ref_id - 0)/n_lidar_sweeps)
            flow_ref2last = net.extract_flow(ref_id, n_lidar_sweeps-1, ref_traj_rt['traj'])
            pc_ref2last = net.transform_pts(flow_ref2last, pc_ref)
            dist_thresh_ref2last = options.sc_d_thresh_min + options.sc_d_thresh_max*((n_lidar_sweeps - ref_id)/n_lidar_sweeps)
            loss_sc_ref2prev = total_sc_loss(labels_t, label_ids, pc_ref, pc_ref2first, d_thresh=dist_thresh_ref2first, max_points=3000)
            loss_sc_ref2post = total_sc_loss(labels_t, label_ids, pc_ref, pc_ref2last, d_thresh=dist_thresh_ref2last, max_points=3000)
            loss_sc = loss_sc_ref2prev + loss_sc_ref2post
            

            # NOTE: Cycle-Consistency loss
            post_traj_rt = net(pc_ref2post, post_id, do_fwd_flow=False, do_bwd_flow=True, do_full_traj=True)    
            prev_traj_rt = net(pc_ref2prev, prev_id, do_fwd_flow=True, do_bwd_flow=False, do_full_traj=True)

            loss_traj_consist = ( (ref_traj_rt['traj'] - post_traj_rt['traj'])**2 ).mean() \
                + ( (ref_traj_rt['traj'] - prev_traj_rt['traj'])**2 ).mean()

            loss_consist = net.compute_traj_consist_loss(ref_traj_rt['traj'], post_traj_rt['traj'], pc_ref, pc_ref2post, ref_id, post_id, options.ctype) \
                + net.compute_traj_consist_loss(ref_traj_rt['traj'], prev_traj_rt['traj'], pc_ref, pc_ref2prev, ref_id, prev_id, options.ctype)

            tmp_id = n_lidar_sweeps // 2
            flow_ref2tmp = net.extract_flow(ref_id, tmp_id, ref_traj_rt['traj'])
            pc_ref2tmp = net.transform_pts(flow_ref2tmp, pc_ref)
            tmp_traj_rt = net(pc_ref2tmp, tmp_id, do_fwd_flow=False, do_bwd_flow=False, do_full_traj=True)
            loss_consist += net.compute_traj_consist_loss(ref_traj_rt['traj'], tmp_traj_rt['traj'], pc_ref, pc_ref2tmp, ref_id, tmp_id, options.ctype)
              
            loss = loss_chamfer + options.sc_loss_weight*loss_sc + options.w_consist*loss_consist

            loss.backward()

            if flow_gt_list is not None and ref_id < n_lidar_sweeps-1 and do_val:
                fwd_flow_gt = torch.from_numpy(flow_gt_list[ref_id]).to(pc_ref2post.device)
                EPE3D_1, acc3d_strict_1, acc3d_relax_1, outlier_1, angle_error_1 = scene_flow_metrics(
                    (pc_ref2post - pc_ref).unsqueeze(0), fwd_flow_gt.unsqueeze(0))
                logging.info(f" [EPE: {EPE3D_1:.3f}] [Acc strict: {acc3d_strict_1 * 100:.3f}%]"
                            f" [Acc relax: {acc3d_relax_1 * 100:.3f}%] [Angle error (rad): {angle_error_1:.3f}]"
                            f" [Outl.: {outlier_1 * 100:.3f}%]")

                if ref_id == 0:
                    traj_metric, _ = p3dloss.chamfer_distance(
                        torch.from_numpy(pc_list[-1]).cuda().unsqueeze(0),
                        net.transform_pts(ref_traj_rt['traj'][:, -1, :], pc_ref).unsqueeze(0)
                    )
                    logging.info(f" traj metric: {traj_metric:.3f}")
                
                cur_metrics['train_loss'][ref_id] = loss.item()
                cur_metrics['train_chamfer_loss'][ref_id] = loss_chamfer.item()
                cur_metrics['train_sc_loss'][ref_id] = loss_sc.item()
                cur_metrics['train_consist_loss'][ref_id] = loss_consist.item()
                cur_metrics['traj_consist'][ref_id] = loss_traj_consist.item()
                cur_metrics['epe'][ref_id] = EPE3D_1
                cur_metrics['acc_strict'][ref_id] = acc3d_strict_1
                cur_metrics['acc_relax'][ref_id] = acc3d_relax_1
                cur_metrics['angle_error'][ref_id] = angle_error_1
                cur_metrics['outlier'][ref_id] = outlier_1
                
        if do_val:
            cur_metrics = {label:round(cur_metrics[label].mean(), 4) for label in cur_metrics.keys()}
            cur_metrics['traj_metric'] = round(traj_metric.item(), 4)
            cur_metrics['itr'] = i

            csv_writer.writerow(cur_metrics)
            logging.info(cur_metrics)

            # ANCHOR: Save checkpoints
            logging.info('save checkpoints ...')
            cur_loss = torch.ones(1)*cur_metrics['train_loss']
            if cur_loss < min_loss:
                min_loss = cur_loss
                logging.info(f'Checkpoints saved at Itr: {i}')
                torch.save(net.state_dict(), f"{exp_dir}/traj_field_model.pth")
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        logging.info(f"[Itr: {i}]"
                    f"[ref: {ref_id}]"
                    f"[loss: {loss:.5f}]"
                    f"[chamf_post: {loss_chamfer_ref2post:.4f}]"
                    f"[chamf_prev: {loss_chamfer_ref2prev:.4f}]"
                    f"[sc_total: {loss_sc:.6f}]"
                    f"[sc_first: {loss_sc_ref2prev:.6f}]"
                    f"[sc_last: {loss_sc_ref2post:.6f}]"
                    f"[consist: {loss_consist:.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long-term trajectory estimation with MBNT.")

    parser.add_argument('--exp_name', type=str, default='fit_MBNT_base_test', metavar='N', help='Name of the experiment.')
    parser.add_argument('--dataset_path', type=str, default='/mnt/088A6CBB8A6CA742/av1/av1_traj', metavar='N', help='Dataset path.')
    parser.add_argument('--iters', type=int, default=1001, metavar='N', help='Number of iterations to optimize the model.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=('sgd', 'adam', 'lbfgs', 'lbfgs_custm', 'rmsprop'), help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0, metavar='M', help='SGD momentum (default: 0.9).')
    parser.add_argument('--device', default='cuda:0', type=str, help='device: cpu? cuda?')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='N', help='Weight decay.')
    parser.add_argument('--hidden_units', type=int, default=128, metavar='N', help='Number of hidden units in neural prior')
    parser.add_argument('--act_fn', type=str, default='relu', metavar='AF', help='activation function for neural prior.')

    # ANCHOR: settings for trajectory
    parser.add_argument('--traj_batch_size', type=int, default=4, metavar='batch_size', help='trajecotry training batch size.')
    parser.add_argument('--traj_len', type=int, default=25, help='point cloud sequence length for the trajectory.')
    parser.add_argument('--traj_type', type=str, default='velocity', help='trajectory decoder type')
    parser.add_argument('--st_embed_type', type=str, default='cosine', help='type of spatial temporal embeddings')
    parser.add_argument('--w_consist', type=float, default=1.0, help='the weight of the consistency loss')
    parser.add_argument('--ctype', type=str, default='velocity', help='consistency loss type')

    # For spatial consistency regularizer
    parser.add_argument('--sc_loss_weight', type=float, default=1.0, help='weigth for spatial consistency loss.')
    parser.add_argument('--sc_cluster_eps', type=float, default=0.8, help='Epsilon parameter of DBSCAN when extracting clusters.')
    parser.add_argument('--sc_cluster_min_points', type=int, default=30, help='minimum number of points per cluster in DBSCAN')
    parser.add_argument('--sc_d_thresh_min', type=float, default=0.10, help='sc distance threshold')#0.3
    parser.add_argument('--sc_d_thresh_max', type=float, default=0.39, help='sc distance threshold')#2 or 0.5 m/s
    
    options = parser.parse_args()

    exp_dir_path = os.path.join(os.path.dirname(__file__), '../', f"checkpoints/{options.exp_name}")
    os.makedirs(exp_dir_path, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.FileHandler(filename=f"{exp_dir_path}/run.log"), logging.StreamHandler()])        

    logging.info('\n' + ' '.join([sys.executable] + sys.argv))
    logging.info(options)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.info('---------------------------------------')
    print_options = vars(options)
    for key in print_options.keys():
        logging.info(key+': '+str(print_options[key]))
    logging.info('---------------------------------------')

    # load point cloud sequeence
    argoverse_tracking_val_log_ids = sorted(glob.glob(os.path.join(options.dataset_path, '*.npz')))

    for fi_name in argoverse_tracking_val_log_ids:
        log_id = fi_name.split('/')[-1].split('.')[0]
    
        data = np.load(fi_name, allow_pickle=True)
        pc_list = [data['pcs'][i] for i in range(options.traj_len)]
        flow_gt_list = [data['flos'][i] for i in range(options.traj_len-1)]
        traj_gt = data['traj'][:, :options.traj_len]
        traj_gt_val_mask = data['traj_val_mask'][:, :options.traj_len]

        cur_exp_dir = os.path.join(exp_dir_path, log_id)
        os.makedirs(cur_exp_dir, exist_ok=True)
        
        fit_trajectory_field(cur_exp_dir, pc_list, options, flow_gt_list, traj_gt, traj_gt_val_mask)
    
    