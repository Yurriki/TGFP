import torch
from torch import nn
from utils import outputActivation
import torch.nn.functional as F

class TGMFNet(nn.Module):

    def __init__(self, args):
        super(TGMFNet, self).__init__()
        self.args = args
        self.use_cuda = args.use_cuda

        self.train_output_flag = args.train_output_flag
        self.use_planning = args.use_planning
        self.use_fusion = args.use_fusion

        # IO Setting
        self.grid_size = args.grid_size
        self.in_length = args.in_length
        self.out_length = args.out_length
        self.num_lat_classes = args.num_lat_classes
        self.num_lon_classes = args.num_lon_classes

        ## Sizes of network layers
        self.temporal_embedding_size = args.temporal_embedding_size
        self.encoder_size = args.encoder_size
        self.decoder_size = args.decoder_size
        self.soc_conv_depth = args.soc_conv_depth
        self.soc_conv2_depth = args.soc_conv2_depth
        self.dynamics_encoding_size = args.dynamics_encoding_size
        self.social_context_size = args.social_context_size
        self.targ_enc_size = self.social_context_size + self.dynamics_encoding_size
        self.fuse_enc_size = args.fuse_enc_size
        self.fuse_conv1_size = 2 * self.fuse_enc_size
        self.fuse_conv2_size = 4 * self.fuse_enc_size
        self.soc_embedding_size1 = (((args.grid_size[0] - 4) + 1) // 2) * self.soc_conv2_depth
        self.soc_embedding_size = args.social_context_size
        self.order = args.order
        self.use_attention = args.use_attention
        # Activations:
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.soc_conv = nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
        self.conv_3x1 = nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1))
        if self.use_planning:
            self.soc_maxpool = nn.MaxPool2d((2, 2), padding=(1, 0))
        else:
            self.soc_maxpool = nn.MaxPool2d((2, 1), padding=(1, 0))


        ## Define network parameters
        self.loc_emb = nn.Conv1d(in_channels=1, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)
        # temporal_em == input_embedding_size
        self.loc_lstm = nn.LSTM(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)
        
        if self.use_planning:
            self.plan_loc_lstm = nn.LSTM(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)

        self.temporalConv = nn.Conv1d(in_channels=2, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)

        self.nbh_lstm = nn.LSTM(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)
        if self.use_planning:
            self.plan_lstm = nn.LSTM(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)
        self.dyn_emb = nn.Linear(self.encoder_size, self.dynamics_encoding_size)
        self.loc_dyn = nn.Linear(self.encoder_size, self.dynamics_encoding_size)

        self.loc_op = nn.Linear(self.decoder_size, 5)

        self.dest_enc_size = [8,16]
        self.dest_latent_size = [8,64]
        self.dest_dec_size = [256,128,256]

        self.fdim = 16
        self.zdim = 16
        self.sigma = 1
        self.dest_enc = MLP(input_dim = 2, output_dim = self.fdim, hidden_size=self.dest_enc_size)
        self.latent_enc = MLP(input_dim = self.fdim + self.dynamics_encoding_size + self.soc_embedding_size, output_dim = 2*self.zdim, hidden_size=self.dest_latent_size)
        self.dest_dec = MLP(input_dim = self.dynamics_encoding_size + self.soc_embedding_size + self.zdim, output_dim = 2, hidden_size=self.dest_dec_size)
        self.offset_dec = MLP(input_dim = self.dynamics_encoding_size + self.soc_embedding_size + self.zdim, output_dim = 2, hidden_size=self.dest_dec_size)
        self.multi_modal = True
        self.best_of_n = 9


        self.nbrs_conv_social = nn.Sequential(
            nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3),
            self.leaky_relu,
            nn.MaxPool2d((3, 3), stride=2),
            nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1)),
            self.leaky_relu
        )
        self.nbrs_conv_social_1 = nn.Sequential(
            nn.Conv2d(self.encoder_size, self.soc_conv_depth, 5),
            self.leaky_relu,
            nn.MaxPool2d((5, 1), stride=2),
            self.leaky_relu
        )
        
        if self.use_planning:
            self.plan_conv_social = nn.Sequential(
                nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3),
                self.leaky_relu,
                nn.MaxPool2d((3, 3), stride=2),
                nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1)),
                self.leaky_relu
            )
            self.pool_after_merge = nn.MaxPool2d((2, 2), padding=(1, 0))
        else:
            self.pool_after_merge = nn.MaxPool2d((2, 1), padding=(1, 0))
        
        self.pool_after_merge1 = nn.MaxPool2d((2, 1), padding=(1, 0))

        
        if self.order == 0:
            self.loc_dec = torch.nn.LSTM(self.soc_embedding_size + self.dynamics_encoding_size+self.num_lat_classes + self.num_lon_classes,
                                             self.decoder_size, self.decoder_size)
        else:
            self.loc_dec = torch.nn.LSTM(self.soc_embedding_size + self.dynamics_encoding_size + self.decoder_size+ self.num_lat_classes + self.num_lon_classes,
                                             self.decoder_size)

            # Vel
        if not self.order == 0:

            self.vel_emb = torch.nn.Conv1d(in_channels=1, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)
            self.vel_lstm = torch.nn.LSTM(self.temporal_embedding_size, self.encoder_size, 1)
            self.vel_dyn = torch.nn.Linear(self.encoder_size, self.dynamics_encoding_size)
            self.vel_op = torch.nn.Linear(self.decoder_size, 5)

            if self.order == 1:
                self.trans1 = torch.nn.Linear(2 * self.dynamics_encoding_size, self.dynamics_encoding_size)
                self.trans = torch.nn.Linear(2 * self.encoder_size, self.encoder_size)
                self.vel_dec = torch.nn.LSTM(self.soc_embedding_size + self.dynamics_encoding_size, self.decoder_size)

            if self.order == 2:
                self.trans1 = torch.nn.Linear(3 * self.dynamics_encoding_size, self.dynamics_encoding_size)
                self.trans = torch.nn.Linear(3 * self.encoder_size, self.encoder_size)
                self.vel_dec = torch.nn.LSTM(self.soc_embedding_size + self.dynamics_encoding_size + self.decoder_size,
                                             self.decoder_size)

                # Acc
                self.acc_emb = torch.nn.Conv1d(in_channels=1, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)
                self.acc_lstm = torch.nn.LSTM(self.temporal_embedding_size, self.encoder_size, 1)
                self.acc_dyn = torch.nn.Linear(self.encoder_size, self.dynamics_encoding_size)
                self.acc_dec = torch.nn.LSTM(self.dynamics_encoding_size + self.soc_embedding_size, self.decoder_size)
                self.acc_op = torch.nn.Linear(self.decoder_size, 5)

            if self.order == 3:
                self.trans1 = torch.nn.Linear(4 * self.dynamics_encoding_size, self.dynamics_encoding_size)
                self.trans = torch.nn.Linear(4 * self.encoder_size, self.encoder_size)
                self.vel_dec = torch.nn.LSTM(self.soc_embedding_size + self.dynamics_encoding_size + self.decoder_size,
                                             self.decoder_size)

                # Acc
                self.acc_emb = torch.nn.Conv1d(in_channels=1, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)
                self.acc_lstm = torch.nn.LSTM(self.temporal_embedding_size, self.encoder_size, 1)
                self.acc_dyn = torch.nn.Linear(self.encoder_size, self.dynamics_encoding_size)
                self.acc_dec = torch.nn.LSTM(self.dynamics_encoding_size + self.soc_embedding_size+ self.decoder_size, self.decoder_size)
                self.acc_op = torch.nn.Linear(self.decoder_size, 5)

                self.coll_emb = torch.nn.Conv1d(in_channels=1, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)
                self.coll_lstm = torch.nn.LSTM(self.temporal_embedding_size, self.encoder_size, 1)
                self.coll_dyn = torch.nn.Linear(self.encoder_size, self.dynamics_encoding_size)
                self.coll_dec = torch.nn.LSTM(self.dynamics_encoding_size + self.soc_embedding_size, self.decoder_size)
                self.coll_op = torch.nn.Linear(self.decoder_size, 5)

        ''' Target Fusion Module'''
        if self.use_fusion:
            ''' Fused Structure'''
            self.fcn_conv1 = nn.Conv2d(self.targ_enc_size, self.fuse_conv1_size, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(self.fuse_conv1_size)
            self.fcn_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.fcn_conv2 = nn.Conv2d(self.fuse_conv1_size, self.fuse_conv2_size, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(self.fuse_conv2_size)
            self.fcn_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.fcn_convTrans1 = nn.ConvTranspose2d(self.fuse_conv2_size, self.fuse_conv1_size, kernel_size=3, stride=2, padding=1)
            self.back_bn1 = nn.BatchNorm2d(self.fuse_conv1_size)
            self.fcn_convTrans2 = nn.ConvTranspose2d(self.fuse_conv1_size, self.fuse_enc_size, kernel_size=3, stride=2, padding=1)
            self.back_bn2 = nn.BatchNorm2d(self.fuse_enc_size)
            self.op_lat = nn.Linear(self.targ_enc_size + self.fuse_enc_size,
                                self.num_lat_classes)  # output lateral maneuver.
            self.op_lon = nn.Linear(self.targ_enc_size + self.fuse_enc_size,
                                self.num_lon_classes)  # output longitudinal maneuver.
            self.dec_lstm = nn.LSTM(input_size=self.targ_enc_size + self.fuse_enc_size +self.fdim,
                                      hidden_size=self.decoder_size)

        else:
            self.fuse_enc_size = 0
            self.op_lat = nn.Linear(self.targ_enc_size,self.num_lat_classes)  # output lateral maneuver.
            self.op_lon = nn.Linear(self.targ_enc_size ,self.num_lon_classes)  # output longitudinal maneuver.
            self.dec_lstm = nn.LSTM(input_size=self.targ_enc_size + self.fdim,hidden_size=self.decoder_size)
        
        self.tanh = nn.Tanh()
        
        self.pre4att = nn.Sequential(
            nn.Linear(self.encoder_size, 1),
        )

        if self.use_attention:
                self.attention = nn.Sequential(
                    nn.Linear(self.encoder_size, 10),
                    self.tanh,
                    nn.Linear(10, 1),
                    self.leaky_relu,
                    nn.Softmax(dim=1)
                )
                self.attention_Out = nn.Sequential(
                    nn.Linear(117, 25),
                    nn.Softmax(dim=1)
                )



    def encode(self, emb, lstm, dyn, ip, nbrs):
        dyn_enc = self.leaky_relu(emb(ip.permute(1,2,0)))
        _, (dyn_enc, _) = lstm(dyn_enc.permute(2,0,1))
        hist_enc = self.leaky_relu( dyn(dyn_enc.view(dyn_enc.shape[1],dyn_enc.shape[2])) )

        dyn_enc1 = self.leaky_relu(emb(nbrs.permute(1,2,0)))
        _, (dyn_enc1, _) = lstm(dyn_enc1.permute(2,0,1))
        nbrs_enc = dyn_enc1.view(dyn_enc1.shape[1],dyn_enc1.shape[2])

        return hist_enc, nbrs_enc


    def soc_pooling(self, masks, ip):
        nbrs_grid = torch.zeros_like(masks).float()
        nbrs_grid = nbrs_grid.masked_scatter_(masks, ip)
        nbrs_grid = nbrs_grid.permute(0,3,2,1)

        soc_enc = self.nbrs_conv_social(nbrs_grid)
        soc_enc = self.pool_after_merge1(soc_enc)
        return soc_enc
         
    


    def soc_encode(self, nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc,vel_nbrs, acc_nbrs, coll_nbrs, vel_hist, acc_hist, coll_hist ,dest):
        dyn_enc = self.leaky_relu(self.temporalConv(targsHist.permute(1,2,0)))
        _, (dyn_enc, _) = self.nbh_lstm(dyn_enc.permute(2,0,1))
        dyn_enc = self.leaky_relu( self.dyn_emb(dyn_enc.view(dyn_enc.shape[1],dyn_enc.shape[2])) )

        ''' Forward neighbour vehicles'''
        nbrs_enc = self.leaky_relu(self.temporalConv(nbsHist.permute(1, 2, 0)))
        _, (nbrs_enc, _) = self.nbh_lstm(nbrs_enc.permute(2, 0, 1))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        if not self.order == 0:
            vel_enc, nbrs_vel_enc = self.encode(self.vel_emb, self.vel_lstm, self.vel_dyn, vel_hist, vel_nbrs)
            if self.order == 1:
                all_enc = torch.cat((nbrs_enc, nbrs_vel_enc), axis = 1)
                dyn_enc = torch.cat((dyn_enc, vel_enc), axis = 1)
                all_enc = self.trans(all_enc)
                dyn_enc = self.trans1(dyn_enc)

            if self.order == 2:
                acc_enc, nbrs_acc_enc = self.encode(self.acc_emb, self.acc_lstm, self.acc_dyn, acc_hist, acc_nbrs)
                all_enc = torch.cat((nbrs_enc, nbrs_vel_enc, nbrs_acc_enc), axis = 1)
                dyn_enc = torch.cat((dyn_enc, vel_enc,acc_enc), axis = 1)
                all_enc = self.trans(all_enc)
                dyn_enc = self.trans1(dyn_enc)
            
            if self.order == 3:
                acc_enc, nbrs_acc_enc = self.encode(self.acc_emb, self.acc_lstm, self.acc_dyn, acc_hist, acc_nbrs)

                coll_enc, nbrs_coll_enc = self.encode(self.coll_emb, self.coll_lstm, self.coll_dyn, coll_hist, coll_nbrs)

                all_enc = torch.cat((nbrs_enc, nbrs_vel_enc, nbrs_acc_enc, nbrs_coll_enc), axis = 1)
                dyn_enc = torch.cat((dyn_enc, vel_enc,acc_enc,coll_enc), axis = 1)
                all_enc = self.trans(all_enc)
                dyn_enc = self.trans1(dyn_enc)
        # merge_grid

        else:
            all_enc = social_context

        ''' Masked neighbour vehicles'''
        nbrs_grid = torch.zeros_like(nbsMask).float()
        nbrs_grid = nbrs_grid.masked_scatter_(nbsMask, all_enc)
        nbrs_grid = nbrs_grid.view(nbrs_grid.shape[0], nbrs_grid.shape[1]*nbrs_grid.shape[2],nbrs_grid.shape[3])
        nbrs_grid = nbrs_grid.view(nbrs_grid.shape[0], 5, 25 ,nbrs_grid.shape[2])
        nbrs_grid = nbrs_grid.permute(0,3,2,1)
        nbrs_grid_1 = self.nbrs_conv_social(nbrs_grid)
        nbrs_grid_2 = self.nbrs_conv_social(nbrs_grid)
        nbrs_grid = nbrs_grid_1+nbrs_grid_2

        if self.use_planning:
            ''' Forward planned vehicle'''
            plan_enc = self.leaky_relu(self.temporalConv(planFut.permute(1, 2, 0)))
            _, (plan_enc, _) = self.plan_lstm(plan_enc.permute(2, 0, 1))
            plan_enc = plan_enc.view(plan_enc.shape[1], plan_enc.shape[2])

            ''' Masked planned vehicle'''
            plan_grid = torch.zeros_like(planMask).float()
            plan_grid = plan_grid.masked_scatter_(planMask, plan_enc)
            plan_grid = plan_grid.view(plan_grid.shape[0], plan_grid.shape[1]*plan_grid.shape[2],plan_grid.shape[3])
            plan_grid = plan_grid.view(plan_grid.shape[0], 5, 25 ,plan_grid.shape[2])
            plan_grid = plan_grid.permute(0, 3, 2, 1)
            plan_grid = self.plan_conv_social(plan_grid)

            ''' Merge neighbour and planned vehicle'''
            merge_grid = torch.cat((nbrs_grid, plan_grid), dim=3)
            social_context = self.pool_after_merge(merge_grid)
        else:
            social_context = self.pool_after_merge(nbrs_grid)
        all_enc = social_context.view(-1, self.social_context_size)
        ## Concatenate encodings:
        enc = torch.cat((all_enc, dyn_enc), 1)
        return enc

    
    def forward(self, nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc,vel_nbrs, acc_nbrs, coll_nbrs, vel_hist, acc_hist, coll_hist ,dest):

        enc = self.soc_encode(nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc,vel_nbrs, acc_nbrs, coll_nbrs, vel_hist, acc_hist, coll_hist ,dest)

        if self.multi_modal:
            if self.train_output_flag:
                # CVAE code
                dest_features = self.dest_enc(dest)
                features = torch.cat((enc, dest_features), dim = 1)
                latent =  self.latent_enc(features)
    
                mu = latent[:, 0:self.zdim] # 2-d array
                logvar = latent[:, self.zdim:] # 2-d array
    
                var = logvar.mul(0.5).exp_()
                eps = torch.DoubleTensor(var.size()).normal_()
                eps = eps.cuda()
                z = eps.mul(var).add_(mu)
            
            else:
                z = torch.Tensor(enc.shape[0], self.zdim)  
                z.normal_(0, self.sigma)
            
            z = z.float().cuda()
            decoder_input = torch.cat((enc, z), dim = 1)
            generated_dest = self.dest_dec(decoder_input)
            generated_dest_features = self.dest_enc(generated_dest)
            enc_offset = torch.cat((enc, generated_dest_features), axis = 1)
            offset = self.offset_dec(enc_offset)
            generated_dest = offset+generated_dest

            if self.train_output_flag:
                loc_pred = self.predict(enc, generated_dest)

                return loc_pred, None,None, generated_dest, mu, logvar
            else:
                return generated_dest,lat_enc,lon_enc





    def decode(self,enc):
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.loc_op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred
    
    def decode_att(self,enc,W_M_O):
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(2, 1, 0)
        h_dec = h_dec * W_M_O
        h_dec = h_dec.permute(1, 2, 0)
        fut_pred = self.loc_op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred

    def predict(self,enc, generated_dest):
        if self.train_output_flag:
            if self.multi_modal:
                generated_dest_features = self.dest_enc(generated_dest)
                enc = torch.cat((enc, generated_dest_features), axis = 1)
                loc_enc = enc.repeat(self.out_length, 1, 1)


            fut_pred = self.decode(loc_enc)
            return fut_pred
        else:
            generated_dest_features = self.dest_enc(generated_dest)
            enc_tmp = torch.cat((enc, generated_dest_features), axis = 1)
            loc_enc = enc_tmp.repeat(self.out_length, 1, 1)
            # if not self.order == 0:
            #     loc_enc = torch.cat((loc_enc, h_vel), axis=2)
            fut_pred = self.decode(loc_enc)
            return fut_pred











class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(torch.nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()

        self.sigmoid = torch.nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = torch.nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x