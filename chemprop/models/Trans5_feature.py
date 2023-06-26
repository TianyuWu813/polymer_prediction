# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import math
import torch.nn as nn
from argparse import Namespace
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))
class RBFLayer(nn.Module):
    def __init__(self, K=64, cutoff=10, dtype=torch.float):
        super().__init__()
        self.cutoff = cutoff

        centers = torch.tensor(softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K)), dtype=dtype)
        self.centers = nn.Parameter(F.softplus(centers))

        widths = torch.tensor([softplus_inverse(0.5 / ((1.0 - np.exp(-cutoff) / K)) ** 2)] * K, dtype=dtype)
        self.widths = nn.Parameter(F.softplus(widths))
    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3, x4, x5 = torch.pow(x, 3.0), torch.pow(x, 4.0), torch.pow(x, 5.0)
        return torch.where(x < 1, 1-6*x5+15*x4-10*x3, torch.zeros_like(x))
    def forward(self, D):
        D = D.unsqueeze(-1)
        return self.cutoff_fn(D) * torch.exp(-self.widths*torch.pow((torch.exp(-D) - self.centers), 2))

class GraphFormer(nn.Module):
    def __init__(self,args,max_seq_count):
        super(GraphFormer, self).__init__()
        #self.save_hyperparameters()
        self.args = args
        self.n_layers = args.tr_n_layers
        self.head_size = args.tr_head_size

        # self.hidden_dim = args.tr_hidden_dim
        self.hidden_dim = args.tr_hidden_dim+args.features_dim

        self.dropout_rate = args.tr_dropout_rate
        self.intput_dropout_rate = args.tr_intput_dropout_rate
        self.weight_decay = args.tr_weight_decay
        self.ffn_dim = args.tr_ffn_dim
        self.attention_dropout_rate = args.tr_attention_dropout_rate
        self.max_seq_count = max_seq_count


        # if dataset_name == 'ZINC':
        #     self.atom_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        #     self.edge_encoder = nn.Embedding(64, head_size, padding_idx=0)
        #     self.edge_type = edge_type
        #     if self.edge_type == 'multi_hop':
        #         self.edge_dis_encoder = nn.Embedding(40 * head_size * head_size,1)
        #     self.rel_pos_encoder = nn.Embedding(40, head_size, padding_idx=0)
        #     self.in_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        #     self.out_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        # else:
        #     self.atom_encoder = nn.Embedding(128 * 37 + 1, hidden_dim, padding_idx=0)
        #     self.edge_encoder = nn.Embedding(128 * 6 + 1, head_size, padding_idx=0)
        #     self.edge_type = edge_type
        #     if self.edge_type == 'multi_hop':
        #         self.edge_dis_encoder = nn.Embedding(128 * head_size * head_size,1)
        #     self.rel_pos_encoder = nn.Embedding(512, head_size, padding_idx=0)
        #     self.in_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
        #     self.out_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
        self.atom_encoder = nn.Embedding(self.max_seq_count, self.hidden_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(self.intput_dropout_rate)

        self.encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.head_size)
                    for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(self.encoders)
        #self.final_ln = nn.LayerNorm(self.hidden_dim)
        # if dataset_name == 'PCQM4M-LSC':
        #     self.out_proj = nn.Linear(hidden_dim, 1)
        # else:
        self.downstream_out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # self.graph_token = nn.Embedding(1, self.hidden_dim)
        # self.graph_token_virtual_distance = nn.Embedding(1, self.head_size)
        #
        # self.graph_bia_enconder = nn.Embedding(self.hidden_dim,self.max_seq_count,padding_idx=0)

        self.feature_linear = nn.Linear(args.features_dim, self.hidden_dim)
        # self.feature_transfer = nn.Linear(self.dim,self.head_size)
        self.feature_transfer = nn.Linear(args.features_dim, self.head_size)

        # self.mask_linear = nn.Linear(self.hidden_dim, self.max_seq_count)
        # self.evaluator = get_dataset(dataset_name)['evaluator']
        # self.metric = get_dataset(dataset_name)['metric']
        # self.loss_fn = get_dataset(dataset_name)['loss_fn']
        # self.dataset_name = dataset_name
        #
        # self.warmup_updates = warmup_updates
        # self.tot_updates = tot_updates
        # self.peak_lr = peak_lr
        # self.end_lr = end_lr
        # self.weight_decay = weight_decay
        # self.multi_hop_max_dist = multi_hop_max_dist

        # self.flag = flag
        # self.flag_m = flag_m
        # self.flag_step_size = flag_step_size
        # self.flag_mag = flag_mag
        # self.hidden_dim = hidden_dim
        # self.automatic_optimization = not self.flag

        # K = 256
        # cutoff = 10
        # self.rbf = RBFLayer(K, cutoff)
        # self.rel_pos_3d_proj = nn.Linear(K, self.head_size)



    def forward(self, output,bias):#,perturb=None):
        # attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        # edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
        # all_rel_pos_3d_1 = batched_data.all_rel_pos_3d_1

        # graph_attn_bias
        # n_size = seq_output.size()[1]

         #+ mask_attn_bias
        # graph_attn_bias = graph_output.clone()
        # n_graph = graph_attn_bias.size()[0]
        # #print(n_graph)
        # # print(graph_attn_bias.size(), 1234)
        # graph_attn_bias = graph_attn_bias.unsqueeze(0).repeat(n_graph,1,1)  # [n_graph, n_head, n_node+1, n_node+1]
        # #print(graph_attn_bias.size())
        # graph_attn_bias = self.graph_linear(graph_attn_bias)
        # graph_attn_bias = graph_attn_bias.transpose(1, 2).transpose(0, 1)
        #print(graph_attn_bias.size(),1234)
        #print(graph_attn_bias.size(), 1234)
        # bias = graph_attn_bias
        # # rel pos




        # transfomrer encoder
        if bias != None:
            feature_bias = bias.clone()
            # print(feature_bias.size())
            feature_bias = self.feature_transfer(feature_bias)
            n_graph = feature_bias.size()[0]
            features_bias = feature_bias.unsqueeze(1)
            features_bias = features_bias.transpose(1, 2)
            features_bias = features_bias.unsqueeze(2)

            for enc_layer in self.layers:
                output = enc_layer(output, features_bias)

        else:
            features_bias=bias
        # output = self.input_dropout(output)
            for enc_layer in self.layers:
                output = enc_layer(output,features_bias)

        #output = self.final_ln(output)
        # output part
        # if self.dataset_name == 'PCQM4M-LSC':
        #     output = self.out_proj(output[:, 0, :])                        # get whole graph rep
        # else:
        #output = self.downstream_out_proj(output[:, :])
        return output

    # def training_step(self, batched_data, batch_idx):
    #     if self.dataset_name == 'ogbg-molpcba':
    #         if not self.flag:
    #             y_hat = self(batched_data).view(-1)
    #             y_gt = batched_data.y.view(-1).float()
    #             mask = ~torch.isnan(y_gt)
    #             loss = self.loss_fn(y_hat[mask], y_gt[mask])
    #         else:
    #             y_gt = batched_data.y.view(-1).float()
    #             mask = ~torch.isnan(y_gt)
    #
    #             forward = lambda perturb: self(batched_data, perturb)
    #             model_forward = (self, forward)
    #             n_graph, n_node = batched_data.x.size()[:2]
    #             perturb_shape = (n_graph, n_node, self.hidden_dim)
    #
    #             optimizer = self.optimizers()
    #             optimizer.zero_grad()
    #             loss, _ = flag(model_forward, perturb_shape, y_gt[mask], optimizer, batched_data.x.device, self.loss_fn,
    #                            m=self.flag_m, step_size=self.flag_step_size, mask=mask)
    #
    #     elif self.dataset_name == 'ogbg-molhiv':
    #         if not self.flag:
    #             y_hat = self(batched_data).view(-1)
    #             y_gt = batched_data.y.view(-1).float()
    #             loss = self.loss_fn(y_hat, y_gt)
    #         else:
    #             y_gt = batched_data.y.view(-1).float()
    #             forward = lambda perturb: self(batched_data, perturb)
    #             model_forward = (self, forward)
    #             n_graph, n_node = batched_data.x.size()[:2]
    #             perturb_shape = (n_graph, n_node, self.hidden_dim)
    #
    #             optimizer = self.optimizers()
    #             optimizer.zero_grad()
    #             loss, _ = flag_bounded(model_forward, perturb_shape, y_gt, optimizer, batched_data.x.device, self.loss_fn,
    #                            m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag)
    #             self.lr_schedulers().step()
    #     else:
    #         y_hat = self(batched_data).view(-1)
    #         y_gt = batched_data.y.view(-1)
    #         loss = self.loss_fn(y_hat, y_gt)
    #     self.log('train_loss', loss, sync_dist=True)
    #     return loss
    #
    # def validation_step(self, batched_data, batch_idx):
    #     if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
    #         y_pred = self(batched_data).view(-1)
    #         y_true = batched_data.y.view(-1)
    #     else:
    #         y_pred = self(batched_data)
    #         y_true = batched_data.y
    #     return {
    #         'y_pred': y_pred,
    #         'y_true': y_true,
    #     }
    #
    # def validation_epoch_end(self, outputs):
    #     y_pred = torch.cat([i['y_pred'] for i in outputs])
    #     y_true = torch.cat([i['y_true'] for i in outputs])
    #     if self.dataset_name == 'ogbg-molpcba':
    #         mask = ~torch.isnan(y_true)
    #         loss = self.loss_fn(y_pred[mask], y_true[mask])
    #         self.log('valid_ap', loss, sync_dist=True)
    #     else:
    #         mask = y_true >= 1
    #         input_dict = {"y_true": y_true[mask], "y_pred": y_pred[mask]}
    #         try:
    #             self.log('valid_' + self.metric, self.evaluator.eval(input_dict)[self.metric], sync_dist=True)
    #             print(self.evaluator.eval(input_dict)[self.metric])
    #         except:
    #             pass
    #
    # def test_step(self, batched_data, batch_idx):
    #     if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
    #         y_pred = self(batched_data).view(-1)
    #         y_true = batched_data.y.view(-1)
    #     else:
    #         y_pred = self(batched_data)
    #         y_true = batched_data.y
    #     return {
    #         'y_pred': y_pred,
    #         'y_true': y_true,
    #         'idx': batched_data.idx,
    #     }
    #
    # def test_epoch_end(self, outputs):
    #     y_pred = torch.cat([i['y_pred'] for i in outputs])
    #     y_true = torch.cat([i['y_true'] for i in outputs])
    #     if self.dataset_name == 'PCQM4M-LSC':
    #         result = y_pred.cpu().float().numpy()
    #         idx = torch.cat([i['idx'] for i in outputs])
    #         torch.save(result, 'y_pred.pt')
    #         print(result.shape)
    #         exit(0)
    #     input_dict = {"y_true": y_true, "y_pred": y_pred}
    #     self.log('test_' + self.metric, self.evaluator.eval(input_dict)[self.metric], sync_dist=True)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
    #     lr_scheduler = {
    #         'scheduler': PolynomialDecayLR(
    #             optimizer,
    #             warmup_updates=self.warmup_updates,
    #             tot_updates=self.tot_updates,
    #             lr=self.peak_lr,
    #             end_lr=self.end_lr,
    #             power=1.0,
    #         ),
    #         'name': 'learning_rate',
    #         'interval':'step',
    #         'frequency': 1,
    #     }
    #     return [optimizer], [lr_scheduler]

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("GraphFormer")
    #     parser.add_argument('--n_layers', type=int, default=12)
    #     parser.add_argument('--head_size', type=int, default=32)
    #     parser.add_argument('--hidden_dim', type=int, default=512)
    #     parser.add_argument('--ffn_dim', type=int, default=512)
    #     parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
    #     parser.add_argument('--dropout_rate', type=float, default=0.1)
    #     parser.add_argument('--weight_decay', type=float, default=0.01)
    #     parser.add_argument('--attention_dropout_rate', type=float, default=0.1)
    #     parser.add_argument('--checkpoint_path', type=str, default='')
    #     parser.add_argument('--warmup_updates', type=int, default=60000)
    #     parser.add_argument('--tot_updates', type=int, default=1000000)
    #     parser.add_argument('--peak_lr', type=float, default=2e-4)
    #     parser.add_argument('--end_lr', type=float, default=1e-9)
    #     parser.add_argument('--edge_type', type=str, default='multi_hop')
    #     parser.add_argument('--validate', action='store_true', default=False)
    #     parser.add_argument('--test', action='store_true', default=False)
    #     parser.add_argument('--flag', action='store_true')
    #     parser.add_argument('--flag_m', type=int, default=3)
    #     parser.add_argument('--flag_step_size', type=float, default=1e-3)
    #     parser.add_argument('--flag_mag', type=float, default=1e-3)
    #     return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        self.hidden_size =  hidden_size
        self.att_size = att_size = hidden_size // head_size
        # print(att_size)
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        # self.linear_atten = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.head_size, self.att_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)

        q = self.linear_q(q).view(batch_size,-1,self.head_size, d_k)
        k = self.linear_k(k).view(batch_size,-1,self.head_size, d_k)
        v = self.linear_v(v).view(batch_size,-1,self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]


        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, self.head_size * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x

class MultiHeadAttention_ProbSparse(nn.Module): # Full attention
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention_ProbSparse, self).__init__()
        self.factor = 5
        self.scale = None
        self.mask_flag = False
        self.output_attention = False


        self.head_size = head_size
        from .Trans3 import Transformer
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        # self.linear_atten = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        return Q_K, M_top
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, q, k, v, attn_bias=None):



        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size,-1,self.head_size, d_k)
        k = self.linear_k(k).view(batch_size,-1,self.head_size, d_k)
        v = self.linear_v(v).view(batch_size,-1,self.head_size, d_v)

        B, L_Q, H, D = q.shape
        _, L_K, _, _ = k.shape

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)#.transpose(2, 3)  # [b, h, d_k, k_len]

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        # U_part = U_part if U_part < L_K else L_K
        # u = u if u < L_Q else L_Q
        U_part =L_K
        u = L_Q
        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)
        if attn_bias is not None:
            scores_top = scores_top + attn_bias
        scale = self.scale or 1. / np.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(v, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, v, scores_top, index, L_Q, None)
        x=context.transpose(2, 1).contiguous()

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        # q = q * self.scale
        # x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        # # print(x.size(),222)
        # if attn_bias is not None:
        #     x = x + attn_bias
        #
        # x = torch.softmax(x, dim=3)
        # x = self.att_dropout(x)
        # x = x.matmul(v)  # [b, h, q_len, attn]
        # x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]


        x = x.view(batch_size, self.head_size * d_v)
        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

class MultiHeadAttention_LogSparse(nn.Module): # Full attention
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention_LogSparse, self).__init__()

        self.scale = None
        self.head_size = head_size
        from .Trans3 import Transformer
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5
        self.hidden_size=hidden_size

        self.split_size = hidden_size * head_size

        self.q_len = 1
        self.query_key = nn.Conv1d(hidden_size, hidden_size * head_size * 2, self.q_len)
        self.value = Conv1D(hidden_size * head_size, 1, hidden_size)
        self.c_proj = Conv1D(hidden_size, 1, hidden_size * head_size)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.head_size, x.size(-1) // self.head_size)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def attn(self, query: torch.Tensor, key, value: torch.Tensor,attn_bias, activation="Softmax"):
        activation = nn.Softmax(dim=-1)
        pre_att = torch.matmul(query, key)
        if attn_bias is not None:
            pre_att = pre_att + attn_bias
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        # mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        # pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def forward(self, x,y,z, attn_bias=None):
        # orig_q_size = x.size()
        d_k = self.att_size
        # d_v = self.att_size
        batch_size = x.size(0)
        x  = self.linear_q(x).view(batch_size,-1,self.hidden_size)
        # print(q.shape,111)
        value  = self.value(x)#.view(batch_size,-1,self.head_size, self.hidden_size)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))


        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn = self.attn(query, key, value,attn_bias)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)

        x = attn.view(batch_size, self.hidden_size)
        x = self.output_layer(x)
        # assert x.size() == orig_q_size
        return x

class MultiheadLinearAttention(nn.Module):
    """Based on "Linformer: Self-Attention with Linear Complexity" (https://arxiv.org/abs/2006.04768)"""
    def __init__(self, embed_dim, project_dim, num_heads, dropout=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim //num_heads
        # if self.head_dim * num_heads != self.embed_dim:
        #     raise ValueError("embed_dim must be divisible by num_heads")
        self.scale = 1 / math.sqrt(self.head_dim)
        self.query_embed_linear = nn.Linear(embed_dim, self.head_dim * self.num_heads)
        self.key_embed_linear = nn.Linear(embed_dim, self.head_dim * self.num_heads)
        self.value_embed_linear = nn.Linear(embed_dim, self.head_dim * self.num_heads)
        self.key_project_linear = nn.Linear(embed_dim, num_heads * project_dim)
        self.value_project_linear = nn.Linear(embed_dim, num_heads * project_dim)
        self.out_linear = nn.Linear(num_heads * self.head_dim, embed_dim)
        self.feature_expand =nn.Linear(num_heads,num_heads * project_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.constant_(p, 0.)

    def forward(self, query, key, value,attn_bias, key_padding_mask=None, need_weights=False, attn_mask=None):
        tgt_len = 1
        src_len = 1
        bs = query.size(0)

        q = self.query_embed_linear(query).view(tgt_len, bs * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.key_embed_linear(key).view(src_len, bs * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.value_embed_linear(value).view(src_len, bs * self.num_heads, self.head_dim).transpose(0, 1)
        e = self.key_project_linear(key).view(src_len, bs * self.num_heads, self.project_dim).permute(1, 2, 0)
        f = self.value_project_linear(value).view(src_len, bs * self.num_heads, self.project_dim).permute(1, 2, 0)
        attn = self.scale * q @ (e @ k).transpose(1, 2)
        attn_bias = attn_bias.view(attn_bias.size(0),attn_bias.size(1))
        attn_bias=self.feature_expand(attn_bias).view(src_len, bs * self.num_heads, self.project_dim).transpose(0, 1)
        attn =attn+attn_bias

        # masking code from PyTorch MultiheadAttention source code
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float('-inf'))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.view(bs, self.num_heads, tgt_len, self.project_dim)
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn = attn.view(bs * self.num_heads, tgt_len, self.project_dim)
        attn = F.dropout(F.softmax(attn, dim=-1), p=self.dropout, training=self.training)
        out = attn @ (f @ v)
        # out = self.out_linear(out.transpose(0, 1).contiguous().view(tgt_len, bs, self.embed_dim))
        out = self.out_linear(out.transpose(0, 1).contiguous().view(bs, self.num_heads * self.head_dim))

        if need_weights:
            attn = attn.view(bs, self.num_heads, tgt_len, self.project_dim).sum(dim=1) / self.num_heads
            return out#, attn
        else:
            return out

class SparseMultiheadAttention(nn.Module):
    """Simple sparse multihead attention using a limited attention span"""
    def __init__(self, embed_dim, num_heads, dropout=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_span = 50
        self.head_dim = embed_dim // num_heads
        # if self.head_dim * num_heads != self.embed_dim:
        #     raise ValueError("embed_dim must be divisible by num_heads")
        self.query_ff = nn.Linear(embed_dim, self.head_dim * self.num_heads)
        self.key_ff = nn.Linear(embed_dim, self.head_dim * self.num_heads)
        self.value_ff = nn.Linear(embed_dim, self.head_dim * self.num_heads)
        self.out_ff = nn.Linear(self.head_dim * self.num_heads, embed_dim)
        self.feature_expand = nn.Linear(num_heads, num_heads * self.head_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, value,attn_bias, **kwargs):
        # pytorch sparse tensors still under active development, so expect changes soon
        # for example, sparse batch matrix multiplication is not currently supported
        # TODO add support for masks
        m = query.size(0)
        n = key.size(0)
        if key.size(0) != value.size(0):
            raise RuntimeError("key and value must have same length")
        query = self.query_ff(query).view(m, -1, self.head_dim).transpose(0, 1)
        key = self.key_ff(key).view(n, -1, self.head_dim).transpose(0, 1)
        value = self.value_ff(value).view(n, -1, self.head_dim).transpose(0, 1)
        attn_bias = attn_bias.view(attn_bias.size(0), attn_bias.size(1))
        attn_bias = self.feature_expand(attn_bias).view(n, -1, self.head_dim).transpose(0, 1)

        rows = torch.arange(m, device=query.device).repeat(2 * self.attn_span + 1, 1).transpose(0, 1).flatten()

        cols = torch.cat([torch.arange(i - self.attn_span, i + self.attn_span + 1, device=query.device) for i in range(n)])

        bounds = (cols >= 0) & (cols < n)
        cols[~bounds] = 0
        idxs = torch.stack([rows, cols])
        vals = (query[:, rows, :] * key[:, cols, :] * bounds.view(1, -1, 1)).sum(-1) / math.sqrt(n)
        attn_bias = (attn_bias[:, rows, :] * attn_bias[:, cols, :]).sum(-1)
        vals= vals + attn_bias
        vals[:, ~bounds] = -float("inf")
        vals = torch.dropout(torch.softmax(vals.view(-1, n, 2 * self.attn_span + 1), dim=-1), self.dropout, self.training).view(-1, idxs.size(1))
        attn_matrix = [torch.sparse.FloatTensor(idxs[:, bounds], val[bounds], (m, n)) for val in vals]
        out = self.out_ff(torch.stack([torch.sparse.mm(attn, val) for attn, val in zip(attn_matrix, value)]).transpose(0, 1).contiguous().view(n, self.head_dim * self.num_heads))
        return out

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        # self.self_attention = MultiHeadAttention_ProbSparse(hidden_size, attention_dropout_rate, head_size)
        # self.self_attention = MultiHeadAttention_LogSparse(hidden_size, attention_dropout_rate, head_size)
        # self.self_attention = MultiheadLinearAttention(hidden_size,hidden_size,head_size, attention_dropout_rate )
        # self.self_attention = SparseMultiheadAttention(hidden_size,head_size, attention_dropout_rate )

        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias):
        y = self.self_attention_norm(x)
        y = self.self_attention(y,y,y,attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class Transformer_fuse(nn.Module):
    def __init__(self,
                 args: Namespace, max_seq_count):
        super(Transformer_fuse, self).__init__()
        self.args = args
        self.max_seq_count = max_seq_count

        self.encoder = GraphFormer(self.args,self.max_seq_count)

    def forward(self, output,bias) -> torch.FloatTensor:
        output = self.encoder.forward(output,bias)

        return output

class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x