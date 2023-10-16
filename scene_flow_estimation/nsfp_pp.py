# Pairwise scene flow estimation with NSFP++.
# Unofficial implementation of NSFP++ - ECCV'2022 (https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136980416.pdf)

import os, sys
import argparse
import numpy as np
import torch
import time
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.general_utils import *
from utils.nsfp_utils import *
from utils.o3d_uitls import *

# ANCHOR: generator
class GeneratorWrap:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen

def sf_solver(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
    options: argparse.Namespace,
    net: torch.nn.Module,
    i: int,
):

    for param in net.parameters():
        param.requires_grad = True
    
    if options.backward_flow:
        net_inv = copy.deepcopy(net)
        params = [{'params': net.parameters(), 'lr': options.lr, 'weight_decay': options.weight_decay},
                {'params': net_inv.parameters(), 'lr': options.lr, 'weight_decay': options.weight_decay}]
    else:
        params = net.parameters()
    
    if options.optimizer == "sgd":
        print('using SGD.')
        optimizer = torch.optim.SGD(params, lr=options.lr, momentum=options.momentum, weight_decay=options.weight_decay)
    elif options.optimizer == "adam":
        print("Using Adam optimizer.")
        optimizer = torch.optim.Adam(params, lr=options.lr, weight_decay=0)

    total_losses = []
    chamfer_losses = []

    early_stopping = EarlyStopping(patience=options.early_patience, min_delta=0.0001)

    if options.time:
        timers = Timers()
        timers.tic("solver_timer")

    pc1 = pc1.cuda().contiguous()
    pc2 = pc2.cuda().contiguous()

    normal1 = None
    normal2 = None

    # ANCHOR: initialize best metrics
    best_loss_1 = 10.
    best_flow_1 = None
    best_epoch = 0
    best_state_dict = None
    
    for epoch in range(options.iters):
        optimizer.zero_grad()

        flow_pred_1 = net(pc1)
        pc1_deformed = pc1 + flow_pred_1

        loss_chamfer_1, _ = my_chamfer_fn(pc2, pc1_deformed, normal2, normal1)
        
        if options.backward_flow:
            flow_pred_1_prime = net_inv(pc1_deformed)
            pc1_prime_deformed = pc1_deformed - flow_pred_1_prime
            loss_chamfer_1_prime, _ = my_chamfer_fn(pc1_prime_deformed, pc1, normal2, normal1)
        
        if options.backward_flow:
            loss_chamfer = loss_chamfer_1 + loss_chamfer_1_prime
        else:
            loss_chamfer = loss_chamfer_1

        loss = loss_chamfer

        if options.flow_consistency:
            alpha = 0.1
            alpha = alpha*0.5
            flow_dist = torch.cdist(flow_pred_1, flow_pred_1, 2)
            flow_dist_avg = (flow_dist.sum() * alpha) / flow_pred_1.size()[1]
            loss += flow_dist_avg

        flow_pred_1_final = pc1_deformed - pc1

        # ANCHOR: get best metrics
        if loss <= best_loss_1:
            best_loss_1 = loss.item()
            best_flow_1 = flow_pred_1_final
            best_epoch = epoch
            best_state_dict = net.state_dict()
            
        if epoch % 50 == 0:
            logging.info(f"[Sample: {i}]"
                        f"[Ep: {epoch}] [Loss: {loss:.5f}] "
                        )
            
        total_losses.append(loss.item())
        chamfer_losses.append(loss_chamfer)

        if options.animation:
            yield flow_pred_1_final.detach().cpu().numpy()

        if early_stopping.step(loss):
            break
        
        loss.backward()
        optimizer.step()

    if options.time:
        timers.toc("solver_timer")
        time_avg = timers.get_avg("solver_timer")
        logging.info(timers.print())

    # ANCHOR: get the best metrics
    info_dict = {
        'loss': best_loss_1,
        'time': time_avg,
        'epoch': best_epoch
    }
        
    return info_dict, best_flow_1[0].cpu().detach().numpy(), best_state_dict


def fit_scene_flow_pp(
    pc1_dynamic, pc2,
    options: argparse.Namespace
):
    
    info_dict_forward = {'time': 0}
    start_time = time.time()

    labels, clusters = extract_clusters_dbscan(pc1_dynamic, eps = 0.8, min_points=30, return_clusters= True, return_colored_pcd=False)
    print("Has %d clusters" % clusters.shape[0])
    pc1_dynamic_deformed = copy.deepcopy(pc1_dynamic)

    if len(clusters) > 0:
        label_ids = np.unique(labels)[1:]
        for id in label_ids:
            cluster_ids = labels == id
            cluster = pc1_dynamic[cluster_ids]
            cluster_center, cluster_bb_dash = extract_single_cluster_info(cluster, axis_aligned = True, bev = True)

            # Query Expansion:
            inlier_points = get_inlier_points(pc2, cluster_bb_dash)

            if len(inlier_points) > 3:
                # Pruning:
                min_points = min(len(cluster), len(inlier_points)) - 1
                dist_to_center = np.linalg.norm(inlier_points - cluster_center, axis=1)
                min_values_arg = np.argpartition(dist_to_center, min_points)
                pruned_inlier_points = inlier_points[min_values_arg[:min_points]]

                if len(pruned_inlier_points) > 3:
                    # Local Flow:
                    net = Neural_Prior().cuda()
                    source = torch.from_numpy(cluster.astype('float32')).unsqueeze(0)
                    target = torch.from_numpy(pruned_inlier_points.astype('float32')).unsqueeze(0)
                    solver_generator = GeneratorWrap(sf_solver(source, target, options, net, id))
                    for _ in solver_generator: pass
                    info_dict, best_flow_1, model_state_dict = solver_generator.value
                    pc1_dynamic_deformed[cluster_ids] += best_flow_1

    # Final results:
    flow_pred = pc1_dynamic_deformed - pc1_dynamic
    flow_pred = torch.from_numpy(flow_pred).float().clone().unsqueeze(0)
    info_dict_forward['time'] = time.time() - start_time
    return flow_pred, info_dict_forward