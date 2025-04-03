import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import TGMFNet
from minedata import highwayTrajDataset
from utils import initLogging, maskedMSETest, maskedNLLTestnointention,maskedNLLTest,maskedTest
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


## Network Arguments
parser = argparse.ArgumentParser(description='Training: Planning-informed Trajectory Prediction for Autonomous Driving')
# General setting------------------------------------------
parser.add_argument('--use_cuda', action='store_false', help='if use cuda (default: True)', default = True)
parser.add_argument('--use_planning', action="store_false", help='if use planning coupled module (default: True)',default = True)
parser.add_argument('--use_fusion', action="store_false", help='if use targets fusion module (default: True)',default = False)
parser.add_argument('--train_output_flag', action="store_false", help='if concatenate with true maneuver label (default: True)', default = True)
parser.add_argument('--batch_size', type=int, help='batch size to use (default: 64)',  default=64)
parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-3)', default=0.001)
parser.add_argument('--tensorboard', action="store_true", help='if use tensorboard (default: True)', default = True)
# IO setting------------------------------------------
parser.add_argument('--grid_size', type=int,  help='default: (25,5)', nargs=2,    default = [25, 5])
parser.add_argument('--in_length', type=int,  help='History sequence (default: 16)',default = 16)    # 3s history traj at 5Hz
parser.add_argument('--out_length', type=int, help='Predict sequence (default: 25)',default = 25)    # 5s future traj at 5Hz
parser.add_argument('--num_lat_classes', type=int, help='Classes of lateral behaviors',     default = 3)
parser.add_argument('--num_lon_classes', type=int, help='Classes of longitute behaviors',   default = 2)
# Network hyperparameters------------------------------------------
parser.add_argument('--temporal_embedding_size', type=int,  help='Embedding size of the input traj', default = 32)
parser.add_argument('--encoder_size', type=int, help='lstm encoder size',  default = 64)
parser.add_argument('--decoder_size', type=int, help='lstm decoder size',  default = 128)
parser.add_argument('--soc_conv_depth', type=int, help='The 1st social conv depth',  default = 64)
parser.add_argument('--soc_conv2_depth', type=int, help='The 2nd social conv depth',  default = 16)
parser.add_argument('--dynamics_encoding_size', type=int,  help='Embedding size of the vehicle dynamic',  default = 32)
parser.add_argument('--social_context_size', type=int,  help='Embedding size of the social context tensor',  default = 80)
parser.add_argument('--fuse_enc_size', type=int,  help='Feature size to be fused',  default = 112)
## Evaluation setting------------------------------------------
parser.add_argument('--name',     type=str, help='model name', default="1")
parser.add_argument('--test_set', type=str, help='Path to test datasets')
parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
parser.add_argument('--metric',   type=str, help='RMSE & NLL is calculated by (agent/sample) based evaluation', default="agent")
parser.add_argument("--plan_info_ds", type=int, default=1, help="N, further downsampling planning information to N*0.2s")
# Dest setting------------------------------------------
parser.add_argument('--dest_dec_size', type=int, default=[128, 64, 32])
parser.add_argument('--dest_latent_size', type=int, default=[8,50])
parser.add_argument('--dest_enc_size', type=int, default=[8,16])
parser.add_argument('--zdim', type=int, default=16)
parser.add_argument('--fdim', type=int, default=16)
parser.add_argument('--sigma', type=float, default=1.3)
parser.add_argument('--order', type=int, default=3)
parser.add_argument('--best_of_n', type=int, default=6)
parser.add_argument('--use_attention', action="store_false", help='if use attention module (default: True)',default = True)
def model_evaluate():

    args = parser.parse_args()
    ## Initialize network
    TGMF = TGMFNet(args)
    TGMF.load_state_dict(torch.load('./trained_models/{}/{}.tar'.format((args.name).split('-')[0], args.name)))
    if args.use_cuda:
        TGMF = TGMF.cuda()
    ## Evaluation Mode
    TGMF.eval()
    TGMF.train_output_flag = False
    initLogging(log_file='./trained_models/{}/evaluationall6.log'.format((args.name).split('-')[0]))
    ## Intialize dataset
    logging.info("Loading test data from {}...".format(args.test_set))
    tsSet = highwayTrajDataset(path=args.test_set,
                               targ_enc_size=args.social_context_size+args.dynamics_encoding_size,
                               grid_size=args.grid_size,
                               fit_plan_traj=True,
                               fit_plan_further_ds=args.plan_info_ds)
    logging.info("TOTAL :: {} test data.".format(len(tsSet)) )
    tsDataloader = DataLoader(tsSet, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=tsSet.collate_fn)
    ## Loss statistic
    logging.info("<{}> evaluated by {}-based NLL & RMSE, with planning input of {}s step.".format(args.name, args.metric, args.plan_info_ds*0.2))
    if args.metric == 'agent':
        rmse_r_stat = torch.zeros((np.max(tsSet.Data[:, 0]).astype(int) + 1,
                                  np.max(tsSet.Data[:, 13:(13 + tsSet.grid_cells)]).astype(int) + 1, args.out_length)).cuda()
        rmse_loss_stat = torch.zeros((np.max(tsSet.Data[:, 0]).astype(int) + 1,
                                   np.max(tsSet.Data[:, 13:(13 + tsSet.grid_cells)]).astype(int) + 1, args.out_length)).cuda()
        both_count_stat = torch.zeros((np.max(tsSet.Data[:, 0]).astype(int) + 1,
                                    np.max(tsSet.Data[:, 13:(13 + tsSet.grid_cells)]).astype(int) + 1, args.out_length)).cuda()
    elif args.metric == 'sample':
        rmse_loss = torch.zeros(25).cuda()
        rmse_counts = torch.zeros(25).cuda()
        nll_loss = torch.zeros(25).cuda()
        nll_counts = torch.zeros(25).cuda()
    else:
        raise RuntimeError("Wrong type of evaluation metric is specified")
    avg_eva_time = 0
    ## Evaluation process
    with torch.no_grad():
        for i, data in enumerate(tsDataloader):
            st_time = time.time()
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, targsFut, targsFutMask, lat_enc, lon_enc, idxs, nbsVel, nbsAcc, nbsColl,vel_hist,acc_hist,coll_hist = data
            # Initialize Variables
            dest = targsFut[-1, :, :]
            if args.use_cuda:
                nbsHist = nbsHist.cuda()
                nbsMask = nbsMask.cuda()
                planFut = planFut.cuda()
                planMask = planMask.cuda()
                targsHist = targsHist.cuda()
                targsEncMask = targsEncMask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                targsFut = targsFut.cuda()
                targsFutMask = targsFutMask.cuda()
                nbsVel = nbsVel.cuda()
                nbsAcc = nbsAcc.cuda()
                nbsColl = nbsColl.cuda()
                vel_hist = vel_hist.cuda()
                acc_hist =acc_hist.cuda()
                coll_hist = coll_hist.cuda()

            # Inference
            enc = TGMF.soc_encode(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc, nbsVel, nbsAcc, nbsColl,vel_hist,acc_hist,coll_hist,None)
    
            if TGMF.multi_modal:
                
                # Generate N guess in order to select the best guess  
                best_of_n = TGMF.best_of_n
                all_l2_errors_dest = []
                
                all_guesses = torch.zeros((best_of_n,25,lat_enc.shape[0],2)).cuda()
                fut_pred_max = torch.zeros((25,lat_enc.shape[0],2)).cuda()
                for index in range(best_of_n):
                    
                    dest_recon,lat_pred, lon_pred = TGMF(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc, nbsVel, nbsAcc, nbsColl,vel_hist,acc_hist,coll_hist,None)
                    fut_pred_t = TGMF.predict(enc, dest_recon)

                    all_guesses[index]=fut_pred_t[:,:,:2]
                indices = []
                for k in range(lat_enc.shape[0]):
                    distances = []
                    dest_true = targsFut[:, k, 0:2]
                    for j in range(best_of_n):
                        dest0 = all_guesses[j][:, k, 0:2]
                        dist = ((dest0[:,0] - dest_true[:,0]) ** 2 + (dest0[:,1] - dest_true[:,1]) ** 2)**0.5
                        dist = sum(dist)
                        distances.append((dist, j))
                    min_dist, index = min(distances, key=lambda x: x[0])
                    indices.append(index)
                    fut_pred_max[:, k, :] = all_guesses[index][:, k, :]
                
                
            if args.metric == 'agent':
                dsIDs, targsIDs = tsSet.batchTargetVehsInfo(idxs)
                # Select the trajectory with the largest probability of maneuver label when evaluating by RMSE
                ll,ll_r, cc = maskedTest(fut_pred_max,targsFut, targsFutMask)
                ll = ll.detach()
                ll_r = ll_r.detach()
                cc = cc.detach()
                for j, targ in enumerate(targsIDs):
                    dsID = dsIDs[j]
                    rmse_loss_stat[dsID, targ, :]  += ll[:, j]
                    rmse_r_stat[dsID, targ, :]  += ll_r[:, j]
                    both_count_stat[dsID, targ, :]  += cc[:, j]

            # Time estimate
            batch_time = time.time() - st_time
            avg_eva_time += batch_time
            if i%100 == 99:
                eta = avg_eva_time / 100 * (len(tsSet) / args.batch_size - i)
                logging.info( "Evaluation progress(%):{:.2f}".format( i/(len(tsSet)/args.batch_size) * 100,) +
                              " | ETA(s):{}".format(int(eta)))
                avg_eva_time = 0
                
                # Result Summary
                if args.metric == 'agent':
                    # Loss averaged from all predicted vehicles.
                    rmse_r_stat1 = rmse_r_stat.cpu().numpy()
                    rmse_loss_stat1 = rmse_loss_stat.cpu().numpy()
                    both_count_stat1 = both_count_stat.cpu().numpy()
                    ds_ids, veh_ids = both_count_stat1[:,:,0].nonzero()
                    num_vehs = len(veh_ids)
                    rmse_loss_averaged = np.zeros((args.out_length, num_vehs))
                    rmse_r_averaged = np.zeros((args.out_length, num_vehs))
                    count_averaged = np.zeros((args.out_length, num_vehs))
                    for i in range(num_vehs):
                        count_averaged[:, i] = \
                            both_count_stat1[ds_ids[i], veh_ids[i], :].astype(bool)
                        rmse_loss_averaged[:,i] = rmse_loss_stat1[ds_ids[i], veh_ids[i], :] \
                                                * count_averaged[:, i] / (both_count_stat1[ds_ids[i], veh_ids[i], :] + 1e-9)
                        rmse_r_averaged[:,i]  = rmse_r_stat1[ds_ids[i], veh_ids[i], :] \
                                                * count_averaged[:, i] / (both_count_stat1[ds_ids[i], veh_ids[i], :] + 1e-9)
                    rmse_loss_sum = np.sum(rmse_loss_averaged, axis=1)
                    rmse_r_sum = np.sum(rmse_r_averaged, axis=1)
                    count_sum = np.sum(count_averaged, axis=1)
                    fde = (rmse_r_sum / count_sum) *0.3048
                    rmseOverall = np.power(rmse_loss_sum / count_sum, 0.5) * 0.3048  # Unit converted from feet to meter.
                    ade1 = (np.sum(rmse_r_sum[0:5]) / np.sum(count_sum[0:5])) *0.3048
                    ade2 = (np.sum(rmse_r_sum[0:10]) / np.sum(count_sum[0:10])) *0.3048
                    ade3 = (np.sum(rmse_r_sum[0:15]) / np.sum(count_sum[0:15])) *0.3048
                    ade4 = (np.sum(rmse_r_sum[0:20]) / np.sum(count_sum[0:20])) *0.3048
                    ade5 = (np.sum(rmse_r_sum[0:25]) / np.sum(count_sum[0:25])) *0.3048
                    adeaverage = (ade1+ade2+ade3+ade4+ade5)/5
                # Print the metrics every 5 time frame (1s)
                logging.info("RMSE (m)\t=> {}, Mean={:.3f}".format(rmseOverall[4::5], rmseOverall[4::5].mean()))
                logging.info("FDE (m)\t=> {}, Mean={:.3f}".format(fde[4::5], fde[4::5].mean()))
                logging.info("ADE (m)\t=> {} {} {} {} {}, Mean={:.3f}".format(ade1,ade2,ade3,ade4,ade5,adeaverage))
    # Result Summary
    if args.metric == 'agent':
            # Loss averaged from all predicted vehicles.
            rmse_r_stat = rmse_r_stat.cpu().numpy()
            rmse_loss_stat = rmse_loss_stat.cpu().numpy()
            both_count_stat = both_count_stat.cpu().numpy()
            ds_ids, veh_ids = both_count_stat[:,:,0].nonzero()
            num_vehs = len(veh_ids)
            rmse_loss_averaged = np.zeros((args.out_length, num_vehs))
            rmse_r_averaged = np.zeros((args.out_length, num_vehs))
            count_averaged = np.zeros((args.out_length, num_vehs))
            for i in range(num_vehs):
                count_averaged[:, i] = \
                    both_count_stat[ds_ids[i], veh_ids[i], :].astype(bool)
                rmse_loss_averaged[:,i] = rmse_loss_stat[ds_ids[i], veh_ids[i], :] \
                                        * count_averaged[:, i] / (both_count_stat[ds_ids[i], veh_ids[i], :] + 1e-9)
                rmse_r_averaged[:,i]  = rmse_r_stat[ds_ids[i], veh_ids[i], :] \
                                        * count_averaged[:, i] / (both_count_stat[ds_ids[i], veh_ids[i], :] + 1e-9)
            rmse_loss_sum = np.sum(rmse_loss_averaged, axis=1)
            rmse_r_sum = np.sum(rmse_r_averaged, axis=1)
            count_sum = np.sum(count_averaged, axis=1)
            fde = (rmse_r_sum / count_sum) *0.3048
            rmseOverall = np.power(rmse_loss_sum / count_sum, 0.5) * 0.3048  # Unit converted from feet to meter.
            ade1 = (np.sum(rmse_r_sum[0:5]) / np.sum(count_sum[0:5])) *0.3048
            ade2 = (np.sum(rmse_r_sum[0:10]) / np.sum(count_sum[0:10])) *0.3048
            ade3 = (np.sum(rmse_r_sum[0:15]) / np.sum(count_sum[0:15])) *0.3048
            ade4 = (np.sum(rmse_r_sum[0:20]) / np.sum(count_sum[0:20])) *0.3048
            ade5 = (np.sum(rmse_r_sum[0:25]) / np.sum(count_sum[0:25])) *0.3048
            adeaverage = (ade1+ade2+ade3+ade4+ade5)/5
        # Print the metrics every 5 time frame (1s)
    logging.info("RMSE (m)\t=> {}, Mean={:.3f}".format(rmseOverall[4::5], rmseOverall[4::5].mean()))
    logging.info("FDE (m)\t=> {}, Mean={:.3f}".format(fde[4::5], fde[4::5].mean()))
    logging.info("ADE (m)\t=> {} {} {} {} {}, Mean={:.3f}".format(ade1,ade2,ade3,ade4,ade5,adeaverage))

if __name__ == '__main__':
    model_evaluate()