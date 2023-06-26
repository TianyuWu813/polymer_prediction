from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mpn import MPN
from .seq_network import Seq_enconder
# from .Trans3 import Transformer
# from .Trans4 import Transformer_fuse
from .Trans_gra_seq import Transformer
# from .Trans5_feature_nobias import Transformer_fuse
from .Trans5_feature import Transformer_fuse
from chemprop.nn_utils import get_activation_function, initialize_weights
# from chemprop.utils import load_args, load_checkpoint, load_scalers
import numpy as np

def feature_normalize(features,scaler):
    list = []
    for i in range(len(features)):
        feature = scaler.transform(np.array(features[i]).reshape(1, -1))[0]
        x=torch.from_numpy(feature)
        list.append(x)
    tensor=torch.stack(list)
    return tensor

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.hidden_size = 300
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_sequence_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.seq_encoder = Seq_enconder(args)

    def create_Trans(self, args: Namespace,max_seq_count):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.trans = Transformer(args,max_seq_count)

    def create_Trans_fuse(self, args: Namespace,max_seq_count):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.trans_fuse = Transformer_fuse(args,max_seq_count)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            # first_linear_dim =  args.features_size
            first_linear_dim = args.hidden_size * 1
            # if args.use_input_features:
            #     first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, int(args.ffn_hidden_size*0.5)),
                    # nn.Linear(args.ffn_hidden_size, int(args.ffn_hidden_size)),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(int(args.ffn_hidden_size*0.5), args.output_size),
                # nn.Linear(int(args.ffn_hidden_size), args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """

        # seq_output,mask,x,max_seq_count = self.seq_encoder(*input)
        # graph_output,graph_bond_output,features_batch = self.encoder(*input)
        #
        # # molecule_emb = F.normalize(seq_output) # SMILES+FEATURE
        # # molecule_emb = F.normalize(graph_output)  # GRAPH+FEATURE
        # molecule_emb = F.normalize(seq_output)+F.normalize(graph_output)  # SMILES+GRAPH+FEATURE
        # # molecule_emb = self.trans(graph_output,seq_output, mask)  # Tranfrom fuse
        #
        #
        # if features_batch != None:
        #
        #   output_test = torch.cat([molecule_emb, features_batch], dim=1)
        #   trans_fuse_output = self.trans_fuse(output_test, features_batch)
        #   output = self.ffn(trans_fuse_output)
        #
        #
        #
        #   # 不加Trans注意力
        #   # output_test = torch.cat([molecule_emb, features_batch], dim=1)
        #   # output = self.ffn(output_test)
        #   #
        #   # trans_fuse_output = self.trans_fuse(molecule_emb, features_batch)
        #   # output_test = torch.cat([trans_fuse_output, features_batch], dim=1)
        #   # output = self.ffn(output_test)
        #
        # else:
        #   output = self.ffn(molecule_emb)

        # SMILES only
        seq_output, mask, x, max_seq_count = self.seq_encoder(*input)
        # graph_output, graph_bond_output, features_batch = self.encoder(*input)
        molecule_emb = F.normalize(seq_output)
        # molecule_emb = self.trans(molecule_emb, graph_output)
        output = self.ffn(molecule_emb)

        #Graph+ratio
        # graph_output,graph_bond_output,features_batch = self.encoder(*input)
        # molecule_emb = F.normalize(graph_output)
        # molecule_emb = torch.cat([molecule_emb, features_batch], dim=1)
        # output = self.ffn(molecule_emb)

        # feature
        # graph_output, graph_bond_output, features_batch = self.encoder(*input)
        # # print(features_batch.size(),111)
        # output = self.ffn(features_batch)

        # output = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        # feature = graph_output
        # return output, feature # for predict

        return output  # for train

    def predict(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        # """

        # seq_output,mask,x,max_seq_count = self.seq_encoder(*input)
        # graph_output,graph_bond_output,features_batch = self.encoder(*input)
        # #
        # # molecule_emb = F.normalize(seq_output) # SMILES+FEATURE
        # molecule_emb = F.normalize(graph_output)  # GRAPH+FEATURE
        # # molecule_emb = F.normalize(seq_output)+F.normalize(graph_output)  # SMILES+GRAPH+FEATURE
        # # molecule_emb = self.trans(graph_output,seq_output,mask) # Tranfrom fuse
        #
        # # print(features_batch)
        # if features_batch != None:
        #
        #   output_test = torch.cat([molecule_emb, features_batch], dim=1)
        #   trans_fuse_output = self.trans_fuse(output_test, features_batch)
        #   output = self.ffn(trans_fuse_output)
        #
        #
        #
        #   # 不加Trans注意力
        #   # output_test = torch.cat([molecule_emb, features_batch], dim=1)
        #   # output = self.ffn(output_test)
        #   #
        #   # trans_fuse_output = self.trans_fuse(molecule_emb, features_batch)
        #   # output_test = torch.cat([trans_fuse_output, features_batch], dim=1)
        #   # output = self.ffn(output_test)
        #
        # else:
        #   output = self.ffn(molecule_emb)

        # SMILES only
        seq_output, mask, x, max_seq_count = self.seq_encoder(*input)
        # graph_output, graph_bond_output, features_batch = self.encoder(*input)
        molecule_emb = F.normalize(seq_output)
        # molecule_emb = self.trans(molecule_emb, graph_output)
        output = self.ffn(molecule_emb)

        # Graph+ratio
        # graph_output,graph_bond_output,features_batch = self.encoder(*input)
        # molecule_emb = F.normalize(graph_output)
        # molecule_emb = torch.cat([molecule_emb, features_batch], dim=1)
        # output = self.ffn(molecule_emb)

        # # feature only
        # graph_output, graph_bond_output, features_batch = self.encoder(*input)
        # # print(features_batch.size(),111)
        # output = self.ffn(features_batch)

        # output = F.normalize(torch.mean(graph_emb, dim=0, keepdim=True), p=2, dim=1) + F.normalize(torch.mean(seq_emb, dim=0, keepdim=True), p=2, dim=1)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:

                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        feature =  molecule_emb
        # feature = trans_fuse_output
        return output, feature # for predict

        # return output  # for train


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)
    model.create_sequence_encoder(args)
    model.create_Trans(args,model.seq_encoder.max_seq_count)
    model.create_Trans_fuse(args, model.seq_encoder.max_seq_count)
    initialize_weights(model)

    return model
