## Multi-Modal Polymer Property Prediction

## Overview
This project includes the training process of multi-modal polymer property prediction.


## Environment
- python 
- pytroch 
- RDKit
- numpy
- pandas

Tips: Using code `conda install -c rdkit rdkit` can help you install package RDKit quickly.

The install time is near 10-15 minutes.

## Usage
For multi-modal polymer representations:

1.Descriptors

We use Mordred to calculate various of monomer descriptors (https://github.com/mordred-descriptor/mordred). Then we apply a feature downselection process to filter the optimized descriptors (https://github.com/PV-Lab/MLforCOE).

2.Sequence and Graph Representations

The construction of sequence and graph representations can be found in `./chemreprop/features/seq_featurization.py` and `./chemreprop/features/featurization.py`

Demo training and predicting codes can be seen in ./run.sh

## Acknowledgements
We thank all cited codes in github, which are all quite helpful for realizing our work.

## Contact
tianyuwu813@gmail.com

