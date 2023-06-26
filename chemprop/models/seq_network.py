import subprocess
import platform
import os
import re
import argparse
import torch
import torch.nn as nn
from torch.autograd import Function
from collections import namedtuple
import torch.nn.functional as F
from argparse import Namespace
from typing import List, Union
import numpy as np
from torch.autograd import Variable
from chemprop.features import BatchMolGraph, mol2graph
from chemprop.features import BatchSmilesSquence,smile2smile
from chemprop.features import construct_seq_index,get_smiles_feature

def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        torch.nn.init.xavier_uniform_(m.weight,gain=0.5)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def pad_2d_unsqueeze(x, padlen):
    #x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

# class Easy_MLP(torch.nn.Module):
#
#     def __init__(self, latent_size, device):
#         super(Easy_MLP,self).__init__()
#
#         self.latent_size = latent_size
#
#         self.MLP = torch.nn.Sequential(torch.nn.Linear(self.latent_size, 64),  ##latent 64
#                                         torch.nn.ReLU(),
#                                         torch.nn.Linear(64, 32),
#                                         torch.nn.ReLU(),
#                                         torch.nn.Linear(32, 1)
#                                         )
#
#         # self.MLP = torch.nn.Sequential(torch.nn.Linear(self.latent_size, 32),  ##latent 64
#         #                                  torch.nn.ReLU(),
#         #                                  torch.nn.Linear(32, 1),
#         #                                 )
#
#         #self.classifier = nn.Sequential(nn.Linear(self.latent_size, 32),
#          #                               nn.ReLU(),
#           #                              nn.Linear(32, 16),
#            #                             nn.ReLU(),
#             #                            nn.Linear(16, 1),
#              #                           )
#
#         self.apply(weights_init)
#         self.to(device)
#
#     def forward(self, x):
#
#         out = self.MLP(x)
#         return out
#
# class Classifier(torch.nn.Module):
#
#     def __init__(self, latent_size, num_task, device):
#         super(Classifier,self).__init__()
#
#         self.latent_size = latent_size
#         self.num_task = num_task
#
#
#         self.classifier = nn.Sequential(nn.Linear(self.latent_size, 16),
#                                         nn.ReLU(),
#                                         nn.Linear(16, num_task),
#                                         )
#
#         self.apply(weights_init)
#         self.to(device)
#
#     def forward(self, x):
#
#         out = self.classifier(x)
#         return out


class SEQencoder(torch.nn.Module):

    def __init__(self, args):
        super(SEQencoder, self).__init__()

        self.args = args
        self.input_dim = args.seq_input_dim
        self.hidden_size = args.seq_hidden_size
        self.latent_size = args.seq_latent_size
        self.dropout = args.seq_dropout
        self.layer = args.seq_layer
        self.node_index, self.seq_index,self.max_seq_count,self.seq_index_list = construct_seq_index(args.data_path)
        #self.encoder = nn.GRU(self.input_dim, int(self.hidden_size / 2), batch_first=True, bidirectional=True)  ###
        self.encoder = torch.nn.GRU(self.input_dim, int(self.hidden_size / 2), self.layer, batch_first=True, bidirectional=True, dropout=self.dropout)

        self.AtomEmbedding = torch.nn.Embedding(len(self.seq_index),
                                                 self.hidden_size)
        # self.mask_AtomEmbedding = torch.nn.Embedding(len(self.seq_index),
        #                                         self.hidden_size)
        self.AtomEmbedding.weight.requires_grad = True
        #self.encoder = nn.GRU(self.input_dim, self.hidden_size, self.layer, batch_first=True,
         #                     bidirectional=False, dropout=self.dropout)
        #尝试改非双向RNN bid False

        # self.attention = SelfAttention(self.hidden_size)

        self.apply(weights_init)

    def forward(self, smile_list: BatchSmilesSquence, features_batch=None) -> torch.FloatTensor:
        smile_batch = smile_list.get_components()
        smile_feature,smile_sequence = get_smiles_feature(smile_batch,self.seq_index)
        #print(smile_sequence)
        batch_size = len(smile_batch)

        # print(last_hidden.size())

        #sequence_vector = torch.zeros(0, len(self.seq_index))
        seq_vecs = []

        for sequence in smile_sequence:

            smile_emb = self.AtomEmbedding(sequence)
            smile_emb = smile_emb.reshape(1, -1, self.input_dim)
            smile_embbeding, last_hidden = self.encoder(smile_emb)

            smile_embbeding = smile_embbeding.squeeze(0)
            seq_vecs.append(smile_embbeding.mean(0))

        seq_vecs = torch.stack(seq_vecs, dim=0)

        # for sequence in smile_sequence:
        #  last_hidden = self._initialize_hidden_state(batch_size)
        #  for item in sequence:
        #    smile_emb = self.AtomEmbedding(item)
        #    smile_emb = smile_emb.reshape(1, -1, self.input_dim)
        #    smile_embbeding, last_hidden = self.encoder(smile_emb,last_hidden)
        #
        #
        #  smile_embbeding = smile_embbeding.squeeze(0)
        #  seq_vecs.append(smile_embbeding.mean(0))
        #
        # seq_vecs = torch.stack(seq_vecs, dim=0)


        # if self.args.cuda or next(self.parameters()).is_cuda:
        #     smile_feature,smile_sequence = (
        #             smile_feature.cuda(),smile_sequence.cuda())

        # smile_vector = torch.zeros(0, len(self.seq_index))
        # for feature in smile_feature:
        #     smile_vector = torch.cat((smile_vector,feature),0)  #？？？？不确定
        #     print(smile_vector.size())
        # smile_vector = torch.stack(smile_feature, dim=0)
        # smile_vector=smile_vector.long()  #tips smile_vector.long 不对
        #
        # seq_nodes_emb = self.AtomEmbedding(smile_vector)
        # print(seq_nodes_emb.size())
        # seq_nodes_emb = seq_nodes_emb.reshape(1, -1, self.input_dim)
        # print(seq_nodes_emb.size())
        # seq_embbeding, last_hidden = self.encoder(seq_nodes_emb)
        # print(seq_embbeding.size(), 12344)
        # seq_embbeding = seq_embbeding.squeeze(0)
        # print(seq_embbeding.size(),12344)
        # seq_embbeding = seq_embbeding[:, -1, :]
        # print(seq_embbeding.size(), 12344)
        # seq_embbeding = torch.flatten(seq_embbeding, 1)
        # print(seq_embbeding.size(), 12344)



        # print(seq_embbeding)
        #encoder_outputs = seq_embbeding.squeeze(0)
        #x = x.reshape(1, -1, self.input_dim)
        # outputs, last_hidden = self.encoder(x)
        #
        # # attn_output, attn_weights = self.attention(output)
        #
        # output = torch.mean(outputs, dim=1, keepdim=True)
        #
        # encoder_outputs = outputs.squeeze(0)
        #
        # # print(encoder_outputs.shape, mu.shape, logvar.shape)
        #
        return seq_vecs,0,0,0

    def _initialize_hidden_state(self, batch_size):
        if torch.cuda.is_available():
            return torch.zeros(self.layer * 2, 1, int(self.hidden_size/2)).cuda()
        else:
            return torch.zeros(self.layer * 2, 1, int(self.hidden_size/2))

    # def forward(self, smile_list: BatchSmilesSquence, features_batch=None) -> torch.FloatTensor:
    #     smile_batch = smile_list.get_components()
    #     #print(smile_batch)
    #     smile_feature,smile_sequence = get_smiles_feature(smile_batch,self.seq_index)
    #     max_seq_count = self.max_seq_count
    #     #print(smile_sequence)
    #     #sequence_vector = torch.zeros(0, len(self.seq_index))
    #     # if self.args.cuda or next(self.parameters()).is_cuda:
    #     #     smile_sequence = (
    #     #         smile_sequence.cuda())
    #     seq_vecs = []
    #     i = 0
    #     seq = torch.zeros((len(smile_sequence), self.max_seq_count)).long()
    #     mask = torch.zeros((len(smile_sequence), self.max_seq_count)).long()
    #     for sequence in smile_sequence:
    #        #print(sequence.size(),12345)
    #        #print(seq.size(),12345)
    #        seq[i,:len(sequence)] = torch.LongTensor(sequence)
    #        mask[i, :] = torch.LongTensor(([1] * len(sequence)) + ([0] * (self.max_seq_count - len(sequence))))
    #        i+=1
    #        #sequence =sequence.cuda()
    #        #print(smile_embbeding.size())
    #        #smile_embbeding = smile_embbeding.reshape(1, -1, self.input_dim) #[n_graph(1), n_smiles, hidden]
    #        # print(smile_emb.size())
    #
    #        #smile_embbeding, last_hidden = self.encoder(smile_emb)
    #
    #        #smile_embbeding = smile_embbeding.squeeze(0)
    #
    #        #smile_embbeding = torch.mean(smile_embbeding,dim=0,keepdim=True)
    #        # seq_vecs.append(smile_embbeding)
    #        seq_vecs.append(seq)
    #        #print(smile_embbeding.mean(0).size())
    #     #print(seq.size())
    #     #print(mask.size())
    #     smile_embbeding = self.AtomEmbedding(seq)
    #
    #     #print(smile_embbeding.size())
    #
    #     #ex_e_mask = mask.unsqueeze(1).unsqueeze(2)
    #     #print(ex_e_mask.size())
    #     #ex_e_mask = (1.0 - ex_e_mask) * -10000.0
    #     #print(ex_e_mask.size())
    #     mask_embbeding = self.mask_AtomEmbedding(mask)
    #
    #     smile_features = []
    #     for features in smile_feature:
    #         #mask_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    #         #print(features.size)
    #         # mask_tensor= torch.LongTensor(([1] * features.size(1)) + ([0] * (self.max_seq_count - features.size(1))))
    #         # print(mask_tensor.size())
    #         features = pad_2d_unsqueeze(features, self.max_seq_count)
    #         #print(features)
    #         smile_features.append(features)
    #     x = torch.cat(smile_features)
    #
    #
    #     seq_vecs = torch.stack(seq_vecs, dim=0)  #no need
    #     #print(seq_vecs, 12345)
    #
    #
    #     return smile_embbeding,mask_embbeding,x,max_seq_count#[bachsize*hidden]


class Seq_enconder(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(Seq_enconder, self).__init__()
        self.args = args

        #self.encoder = encoder()
        # self.atom_fdim = atom_fdim or get_atom_fdim(args)
        # self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
        #                     (not args.atom_messages) * self.atom_fdim # * 2
        self.graph_input = graph_input
        self.encoder = SEQencoder(self.args)
        # self.max_seq_count = self.encoder.max_seq_count
        self.max_seq_count = 200

    def forward(self, batch: Union[List[str], BatchSmilesSquence],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            batch = smile2smile(batch, self.args)
        output,mask_embbeding,x,max_seq_count = self.encoder.forward(batch, features_batch)

        return output,mask_embbeding,x,max_seq_count

# class SEQencoder(torch.nn.Module):
#
#     def __init__(self, args):
#         super(SEQencoder, self).__init__()
#
#         self.args = args
#         self.input_dim = args.seq_input_dim
#         self.hidden_size = args.seq_hidden_size
#         self.latent_size = args.seq_latent_size
#         self.dropout = args.seq_dropout
#         self.layer = args.seq_layer
#         self.node_index, self.seq_index = construct_seq_index(args.data_path)
#         #self.encoder = nn.GRU(self.input_dim, int(self.hidden_size / 2), batch_first=True, bidirectional=True)  ###
#         self.encoder = torch.nn.GRU(self.input_dim, int(self.hidden_size / 2), self.layer, batch_first=True, bidirectional=True)
#         self.AtomEmbedding = torch.nn.Embedding(len(self.seq_index),
#                                                  self.hidden_size)
#         self.AtomEmbedding.weight.requires_grad = True
#         #self.encoder = nn.GRU(self.input_dim, self.hidden_size, self.layer, batch_first=True,
#          #                     bidirectional=False, dropout=self.dropout)
#         #尝试改非双向RNN bid False
#
#         # self.attention = SelfAttention(self.hidden_size)
#
#         self.apply(weights_init)
#
#
#     def forward(self, smile_list: BatchSmilesSquence, features_batch=None) -> torch.FloatTensor:
#         smile_batch = smile_list.get_components()
#         #print(smile_batch)
#         smile_feature,smile_sequence = get_smiles_feature(smile_batch,self.seq_index)
#         #print(smile_sequence)
#
#         #sequence_vector = torch.zeros(0, len(self.seq_index))
#         seq_vecs = []
#         for sequence in smile_sequence:
#            smile_emb = self.AtomEmbedding(sequence)
#            #print(smile_emb.size())
#            smile_emb = smile_emb.reshape(1, -1, self.input_dim)
#            #print(smile_emb.size())
#            smile_embbeding, last_hidden = self.encoder(smile_emb)
#            smile_embbeding = smile_embbeding.squeeze(0)
#            seq_vecs.append(smile_embbeding.mean(0))
#            #print(sequence)
#            #print(smile_emb.size())
#            #print(smile_embbeding.size(),1111)
#         seq_vecs = torch.stack(seq_vecs, dim=0)
#         #print(seq_vecs.size(), 12345)
#
#         # if self.args.cuda or next(self.parameters()).is_cuda:
#         #     smile_feature,smile_sequence = (
#         #             smile_feature.cuda(),smile_sequence.cuda())
#
#         # smile_vector = torch.zeros(0, len(self.seq_index))
#         # for feature in smile_feature:
#         #     smile_vector = torch.cat((smile_vector,feature),0)  #？？？？不确定
#         #     print(smile_vector.size())
#         # smile_vector = torch.stack(smile_feature, dim=0)
#         # smile_vector=smile_vector.long()  #tips smile_vector.long 不对
#         #
#         # seq_nodes_emb = self.AtomEmbedding(smile_vector)
#         # print(seq_nodes_emb.size())
#         # seq_nodes_emb = seq_nodes_emb.reshape(1, -1, self.input_dim)
#         # print(seq_nodes_emb.size())
#         # seq_embbeding, last_hidden = self.encoder(seq_nodes_emb)
#         # print(seq_embbeding.size(), 12344)
#         # seq_embbeding = seq_embbeding.squeeze(0)
#         # print(seq_embbeding.size(),12344)
#         # seq_embbeding = seq_embbeding[:, -1, :]
#         # print(seq_embbeding.size(), 12344)
#         # seq_embbeding = torch.flatten(seq_embbeding, 1)
#         # print(seq_embbeding.size(), 12344)
#
#
#
#         # print(seq_embbeding)
#         #encoder_outputs = seq_embbeding.squeeze(0)
#         #x = x.reshape(1, -1, self.input_dim)
#         # outputs, last_hidden = self.encoder(x)
#         #
#         # # attn_output, attn_weights = self.attention(output)
#         #
#         # output = torch.mean(outputs, dim=1, keepdim=True)
#         #
#         # encoder_outputs = outputs.squeeze(0)
#         #
#         # # print(encoder_outputs.shape, mu.shape, logvar.shape)
#         #
#         return seq_vecs


# class Seq_enconder(nn.Module):
#     def __init__(self,
#                  args: Namespace,
#                  atom_fdim: int = None,
#                  bond_fdim: int = None,
#                  graph_input: bool = False):
#         super(Seq_enconder, self).__init__()
#         self.args = args
#
#         #self.encoder = encoder()
#         # self.atom_fdim = atom_fdim or get_atom_fdim(args)
#         # self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
#         #                     (not args.atom_messages) * self.atom_fdim # * 2
#         self.graph_input = graph_input
#         self.encoder = SEQencoder(self.args)
#
#     def forward(self, batch: Union[List[str], BatchSmilesSquence],
#                 features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
#         if not self.graph_input:  # if features only, batch won't even be used
#             batch = smile2smile(batch, self.args)
#         output = self.encoder.forward(batch, features_batch)
#
#         return output