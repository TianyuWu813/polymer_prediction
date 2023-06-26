# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:42:36 2019

@author: SY
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from chemprop.parsing import parse_train_args, modify_train_args,parse_predict_args,modify_predict_args
from chemprop.train import make_predictions
from chemprop.features import load_features
from chemprop.utils import rmse
import torch
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
# from seaborn import sns
from rdkit.Chem import AllChem as Chem
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, precision_score, recall_score
def get_fp(list_of_smi):
    """ Function to get fingerprint from a list of SMILES"""
    fingerprints = []
    mols = [Chem.MolFromSmiles(x) for x in list_of_smi]
    # if rdkit can't compute the fingerprint on a SMILES
    # we remove that SMILES
    idx_to_remove = []
    for idx, mol in enumerate(mols):
        try:
            fprint = Chem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False)
            fingerprints.append(fprint)
        except:
            idx_to_remove.append(idx)

    smi_to_keep = [smi for i, smi in enumerate(list_of_smi) if i not in idx_to_remove]
    return fingerprints, smi_to_keep



if __name__ == '__main__':

    # args = parse_predict_args()
    args = parse_train_args()

    modify_train_args(args)
    args.test_path= args.data_path
    df = pd.read_csv(args.test_path)

    property_value= df['property'].values.tolist()


    if args.features_path is not None:
        features_data = []
        for feat_path in args.features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
        # print(features_data, 777777)
    else:
        features_data = None

    pred,smiles,feature = make_predictions(args=args,features=features_data)



    feature_vis_pos = []
    pred_vis_pos = []
    pred_values = []
    for m,n in zip(pred,feature):
        # print(m,n)
        # if m[0] > 7:
            feature_vis_pos.append(n)
            pred_vis_pos.append([m[0]])
            pred_values.append(m[0])
    feature_vis_stack = torch.stack(feature_vis_pos)
    print(pred_values)
    print(rmse(property_value,pred_values))
    print(r2_score(property_value, pred_values))


    activity = []
    for item in pred:
        # activity.append(1.573* np.exp(0.6923 * abs(item))) #1.573  6.2732
        activity.append(6.2732* np.exp(0.6923 * item))  # 1.573  6.2732


    df = pd.DataFrame({'smiles':smiles})
    for i in range(len(pred[0])):
        df[f'pred_{i}'] = [item[i] for item in pred]
        df[f'value_{i}'] = [item[i] for item in activity]
    df.to_csv(f'./predict.csv', index=False)