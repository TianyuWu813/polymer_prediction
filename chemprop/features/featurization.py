#important
from argparse import Namespace
from typing import List, Tuple, Union

from rdkit import Chem
import torch
import numpy as np



mole_dict = {'He': "2", 'Li': "3", 'Be': "4", 'B': "5", 'C': "6",'c': "6", 'N': "7",'n': "7", 'O': "8", 'o': "8",'F': "9", 'Ne': " 10",
             'Na': "11", 'Mg': "12", 'Al': "13", 'Si': "14", 'P': "15", 'S': "16",'s': "16", 'Cl': "17", 'Ar': "18", 'K': "19", 'Ca': "20", 'Ti': "22",
             'Cr': "24", 'Fe': "26", 'Ni': "28",
             29: "Cu", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 'Br': "35", 40: "Zr", 44: "Ru", 45: "Rh", 46: "Pd",
             47: "Ag", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 65: "Tb", 75: "Re", 77: "Ir", 78: "Pt", 79: "Au",
             80: "Hg",
             81: "Tl", 82: "Pb", 'Bi': "83"}

bonding_dict ={'$':0,'<':1,'>':2,'<<':3,'>>':4,'<<<':5,'>>>':6 }
#atom pair dict
pair_list = ["Br", "Cl", "Si", "Na", "Ca", "Ge", "Cu", "Au", "Sn", "Tb", "Pt", "Re", "Ru", "Bi", "Li", "Fe", "Sb", "Hg","Ar","nH","NH",
             "Pb", "Se", 'se', "Ag", "Cr", "Pd", "Ga", "Mg", "Ni", "Ir", "Rh", "Te", "Ti", "Al", "Zr", "Tl", "As"]
# bond_dict = {'SINGLE': 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}

# Atom feature sizes
MAX_ATOMIC_NUM = 100
# ATOM_FEATURES = {
#     'atomic_num': list(range(MAX_ATOMIC_NUM)),
#     'degree': [0, 1, 2, 3, 4, 5],
#     'formal_charge': [-1, -2, 1, 2, 0],
#     'chiral_tag': [0, 1, 2, 3],
#     'num_Hs': [0, 1, 2, 3, 4],
#     'hybridization': [
#         Chem.rdchem.HybridizationType.SP,
#         Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3,
#         Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2
#     ],
# }

Big_ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'bonding_num': [0 ,1 ,2 ,3 ,4 ,5 ,6],
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in Big_ATOM_FEATURES.values()) + 2
BOND_FDIM = 18  #14

# Memoization
SMILES_TO_GRAPH = {}


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding



def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, Big_ATOM_FEATURES['atomic_num']) + \
           [0]* (len(Big_ATOM_FEATURES['bonding_num'])+1)+\
           onek_encoding_unk(atom.GetTotalDegree(), Big_ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), Big_ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), Big_ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), Big_ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), Big_ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def atom_bond_features(bonding_index, functional_groups: List[int] = None):
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """

    features = [0] * (len(Big_ATOM_FEATURES['atomic_num'])+1) + onek_encoding_unk(bonding_index, Big_ATOM_FEATURES['bonding_num'])+ \
           [0] * (len(Big_ATOM_FEATURES['degree'])+1) + [0] * (len(Big_ATOM_FEATURES['formal_charge'])+1) + \
               [0] * (len(Big_ATOM_FEATURES['chiral_tag'])+1) +\
               [0] * (len(Big_ATOM_FEATURES['num_Hs'])+1) + \
               [0] * (len(Big_ATOM_FEATURES['hybridization'])+1)+ \
               [0]*2
    # scaled to about the same range as other features
    # features = onek_encoding_unk(atom.GetAtomicNum() - 1, Big_ATOM_FEATURES['atomic_num']) + \
    #        onek_encoding_unk(atom.GetTotalDegree(), Big_ATOM_FEATURES['degree']) + \
    #        onek_encoding_unk(atom.GetFormalCharge(), Big_ATOM_FEATURES['formal_charge']) + \
    #        onek_encoding_unk(int(atom.GetChiralTag()), Big_ATOM_FEATURES['chiral_tag']) + \
    #        onek_encoding_unk(int(atom.GetTotalNumHs()), Big_ATOM_FEATURES['num_Hs']) + \
    #        onek_encoding_unk(int(atom.GetHybridization()), Big_ATOM_FEATURES['hybridization']) + \
    #        [1 if atom.GetIsAromatic() else 0] + \
    #        [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
            False,
            False,
            False,
            False
        ]

        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    # print(fbond)
    return fbond

def bonding_features(bonding_index) -> List[Union[bool, int, float]]:  #maybe 要改
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    # if bond is None:
    #     fbond = [1] + [0] * (BOND_FDIM - 1)
    # else:
    #     bt = bond.GetBondType()
    if bonding_index == 0:
        fbond = [
                0,  # bond is not None
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False
            ]
        fbond += [0] + [0] * 6
    elif bonding_index == 1 or bonding_index == 2:
        fbond = [
                0,  # bond is not None
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False
            ]
        fbond += [0] + [0] * 6
    elif bonding_index == 3 or bonding_index == 4:
        fbond = [
                0,  # bond is not None
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False
            ]
        fbond += [0] + [0] * 6
    elif bonding_index == 5 or bonding_index == 6:
        fbond = [
                0,  # bond is not None
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True
            ]
        fbond += [0] + [0] * 6

    # print(fbond)
    return fbond

def bond_clear(bigsmiles):
   bigsmiles = bigsmiles.replace('$','')
   bigsmiles = bigsmiles.replace('<','')
   bigsmiles = bigsmiles.replace('>','')
   bigsmiles = bigsmiles.replace('()','')
   bigsmiles = bigsmiles.replace('[]', '')
   return bigsmiles


def bigsmilestosmiles(bigsmiles):
    f = bigsmiles.split('{')
    forward_smile = f[0]
    b = f[1].split('}')
    #backward_smile = b[1]
    polymer = b[0]
    end_scaffold = []
    polymer_unit = polymer.split(';')
    repeat_scaffold_origin = polymer_unit[0].split(',')
    repeat_scaffold=[]
    for i in range(len(repeat_scaffold_origin)):
        repeat_scaffold.append((repeat_scaffold_origin[i].split('.'))[0])
    if len(polymer_unit)>1 :
     end_scaffold = polymer_unit[1].split(',')
    polymer_len = len(repeat_scaffold)
    #print(repeat_scaffold)
    #if len(polymer_unit) > 1:
      #print(end_scaffold)
    #print(bigsmiles)
    #print(f)
    #print(b)
    #print('forward_smile:'+ str(forward_smile) )
    #print('backward_smile:'+ str(backward_smile) )
    #[print('repeat_scaffold: '+ str(i)) for i in repeat_scaffold]
    # if len(polymer_unit) > 1:
    #   [print('end_scaffold: ' + str(i)) for i in end_scaffold]
    return forward_smile, repeat_scaffold, end_scaffold #, backward_smile

def bond_judge(bond_dict1,bond_dict2):
   if bond_dict1==bond_dict2 and bond_dict1== 0:
     return True
   elif bond_dict1 == 1 and bond_dict2 == 2:
     return True
   elif bond_dict1 == 2 and bond_dict2 == 1:
     return True
   elif bond_dict1 == 3 and bond_dict2 == 4:
     return True
   elif bond_dict1 == 4 and bond_dict2 == 3:
     return True
   elif bond_dict1 == 5 and bond_dict2 == 6:
     return True
   elif bond_dict1 == 6 and bond_dict2 == 5:
     return True
   else:
     return None



class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.bond_atom =[]
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.bonds = []

        # Convert smiles to molecule
        #mol = Chem.MolFromSmiles(smiles)
        # fake the number of "atoms" if we are collapsing substructures
        #self.n_atoms = mol.GetNumAtoms()  #原子数量
        #print(self.n_atoms)

        #bigsmiles
        # print(self.smiles)
        self.forward_smile, self.repeat_scaffold, self.end_scaffold = bigsmilestosmiles(self.smiles)

        # print(self.forward_smile, self.repeat_scaffold, self.end_scaffold)
        self.repeat_mol_list =[]  #重复单元的 smile形式
        for i in range(len(self.repeat_scaffold)):
            self.repeat_mol_list.append(bond_clear(self.repeat_scaffold[i]))
        # print(self.repeat_mol_list,1123)

        self.num_atoms = 0
        self.molculars_index = []
        self.smiles_index=[]
        for x in range(len(self.repeat_scaffold)):
            #print(self.repeat_scaffold[x])
            bigsmiles_list = list(self.repeat_scaffold[x])
            smiles_list = list(self.repeat_mol_list[x])
            # print(smiles_list,123)
            molcular_index=[]
            smile_index=[]
            i = 0
            while i <= len(smiles_list)-1:
              if i< len(smiles_list)-1 and (smiles_list[i]+smiles_list[i+1]) in pair_list:
                 self.num_atoms += 1
                 molcular_index.append(smiles_list[i] + smiles_list[i + 1])
                 smile_index.append(i)
                 i = i + 2
                 #print(smiles_list[i]+smiles_list[i+1])
              # elif smiles_list[i] in bonding_dict:
              #     self.num_atoms += 1
              #     smile_index.append(i)
              #     i = i + 1
              #     #print(bonding_dict[smiles_list[i]])
              elif smiles_list[i] in mole_dict:
                 self.num_atoms += 1
                 molcular_index.append(smiles_list[i])
                 smile_index.append(i)
                 i = i + 1
                 #print(mole_dict[smiles_list[i]])
              # print(i)
              else:
                  i = i+1
            self.molculars_index.append(molcular_index)
            self.smiles_index.append(smile_index)
        # print(self.molculars_index)
        # print(self.smiles_index)

        self.bonds_index = []
        self.bonds_dict = []
        self.bigmolculars_index = []
        self.bigsmiles_index=[]
        for x in range(len(self.repeat_scaffold)):
            #print(self.repeat_scaffold[x])
        # smiles list
            bigsmiles_list = list(self.repeat_scaffold[x])
            smiles_list = list(self.repeat_mol_list[x])
            # print(bigsmiles_list, 123)
            # print(smiles_list, 123)
            molcular_index=[]
            smile_index=[]
            bond_index = []
            bondings_dict=[]
            i = 0
            while i <= len(bigsmiles_list)-1:
              if i< len(bigsmiles_list)-1 and (bigsmiles_list[i]+bigsmiles_list[i+1]) in pair_list:
                 self.n_atoms += 1
                 molcular_index.append(bigsmiles_list[i] + bigsmiles_list[i + 1])
                 smile_index.append(i)
                 #print(bigsmiles_list[i] + bigsmiles_list[i + 1])
                 i = i + 2
              elif bigsmiles_list[i] in bonding_dict:
                  if i< len(bigsmiles_list)-2 and (bigsmiles_list[i]+bigsmiles_list[i+1]+bigsmiles_list[i+2]=='<<<' or bigsmiles_list[i]+bigsmiles_list[i+1]+bigsmiles_list[i+2]=='>>>'):
                      self.n_atoms += 1
                      bond_index.append(i)
                      bondings_dict.append(bonding_dict[bigsmiles_list[i]+bigsmiles_list[i+1]+bigsmiles_list[i+2]])
                      i = i + 3
                  elif i< len(bigsmiles_list)-1 and (bigsmiles_list[i]+bigsmiles_list[i+1]=='<<' or bigsmiles_list[i]+bigsmiles_list[i+1]=='>>'):
                      self.n_atoms += 1
                      bond_index.append(i)
                      bondings_dict.append(bonding_dict[bigsmiles_list[i] + bigsmiles_list[i + 1]])
                      i = i + 2
                  else:
                      self.n_atoms += 1
                      bond_index.append(i)
                      bondings_dict.append(bonding_dict[bigsmiles_list[i]])
                      i = i + 1
              elif bigsmiles_list[i] in mole_dict:
                 self.n_atoms += 1
                 molcular_index.append(bigsmiles_list[i])
                 smile_index.append(i)
                 i = i + 1
                 #print(mole_dict[smiles_list[i]])
              # print(i)
              else:
                  i = i+1
            self.bigmolculars_index.append(molcular_index)
            self.bigsmiles_index.append(smile_index)
            self.bonds_index.append(bond_index)
            self.bonds_dict.append(bondings_dict)
        # print(self.bigmolculars_index)
        # print(self.bigsmiles_index)  # 原子 符号 index
        # print(self.bonds_index)   # bond 符号 index
        # print(self.bonds_dict)
        # print(self.num_atoms)
        # print(self.n_atoms)

        bonds_dict= []
        # Get atom features
        for m in range(len(self.repeat_mol_list)):
            # print(self.repeat_mol_list[m])
            mol =  Chem.MolFromSmiles(self.repeat_mol_list[m])
            n_atoms = mol.GetNumAtoms()
            #print(n_atoms)
            for i, atom in enumerate(mol.GetAtoms()):  # i 原子索引 atom rdkit 原子形式
                #print(atom_features(atom))
                self.f_atoms.append(atom_features(atom))
            # print(len(self.f_atoms), 1234)
            for q in range(len(self.bonds_dict[m])):
                #print(atom_bond_features(self.bonds_dict[m][q]))
                self.f_atoms.append(atom_bond_features(self.bonds_dict[m][q]))
                bonds_dict.append(self.bonds_dict[m][q])
            # print(len(self.f_atoms), 1234)
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]  #分子特征 所有原子特征组成的list
        # print(self.f_atoms,1234)
        for _ in range(self.n_atoms):
            self.a2b.append([])
        #print(self.a2b)

        # Get bond features
        atom_count = 0
        atom_count_index = []
        bonds_indexs = []

        for m in range(len(self.repeat_mol_list)):
            mol =  Chem.MolFromSmiles(self.repeat_mol_list[m])
            # smile 的 边
            bond_count = 0
            for a1 in range(mol.GetNumAtoms()):  #atom 边 特征
                for a2 in range(a1 + 1, mol.GetNumAtoms()):
                    bond = mol.GetBondBetweenAtoms(a1, a2)  #rdkit bond 形式
                    #print(bond)
                    if bond is None:
                        continue

                    f_bond = bond_features(bond)  #边的特征
                    #print(f_bond)

                    if args.atom_messages:
                        self.f_bonds.append(f_bond)
                        self.f_bonds.append(f_bond)
                    else:
                        self.f_bonds.append(self.f_atoms[a1+atom_count] + f_bond)
                        self.f_bonds.append(self.f_atoms[a2+atom_count] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2+atom_count].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1+atom_count)
                    self.a2b[a1+atom_count].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2+atom_count)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2  #化学键看作两条有向边，即左到右，右到左
                    self.bonds.append(np.array([a1+atom_count, a2+atom_count]))

            for n in range(len(self.bonds_dict[m])):  ####写的好像不对  bond 符号 边特征
                # print(self.bonds_index[m][n],123)
                if self.bonds_index[m][n] == 0 :
                    f_bond = bonding_features(self.bonds_dict[m][n])
                    if args.atom_messages:
                        self.f_bonds.append(f_bond)
                        self.f_bonds.append(f_bond)
                    else:
                        self.f_bonds.append(self.f_atoms[atom_count+mol.GetNumAtoms()+bond_count] + f_bond)   ### 边的连接不大对？
                        bonds_indexs.append(atom_count+mol.GetNumAtoms()+bond_count)
                        self.f_bonds.append(self.f_atoms[atom_count] + f_bond)

                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[atom_count].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(atom_count+mol.GetNumAtoms()+bond_count)
                    self.a2b[atom_count+mol.GetNumAtoms()+bond_count].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(atom_count)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2  #化学键看作两条有向边，即左到右，右到左
                    self.bonds.append(np.array([atom_count+mol.GetNumAtoms()+bond_count, atom_count]))
                    bond_count+=1
                else:
                    for p in range(1,len(self.bigsmiles_index[m])+1):   ### 连在前一个原子上 可能会导致连载支链等问题 好像不大对
                      if  (p<len(self.bigsmiles_index[m]) and self.bonds_index[m][n]> self.bigsmiles_index[m][p-1] and self.bonds_index[m][n]< self.bigsmiles_index[m][p]) or (p ==len(self.bigsmiles_index[m]) and self.bonds_index[m][n]> self.bigsmiles_index[m][p-1] ):
                          # print(p)
                          # print(self.bigsmiles_index[m][p-1],111)
                          # print(self.bonds_index[m][n],222)
                          # print(self.bigsmiles_index[m][p],333)
                          f_bond = bonding_features(self.bonds_dict[m][n])
                          if args.atom_messages:
                              self.f_bonds.append(f_bond)
                              self.f_bonds.append(f_bond)
                          else:
                              self.f_bonds.append(self.f_atoms[atom_count+mol.GetNumAtoms()+bond_count] + f_bond)
                              bonds_indexs.append(atom_count + mol.GetNumAtoms() + bond_count)
                              self.f_bonds.append(self.f_atoms[atom_count+p-1] + f_bond)
                              # print(atom_count+p-1,12345)
                              b1 = self.n_bonds
                              b2 = b1 + 1
                              self.a2b[atom_count+p-1].append(b1)  # b1 = a1 --> a2
                              self.b2a.append(atom_count+mol.GetNumAtoms()+bond_count)
                              self.a2b[atom_count+mol.GetNumAtoms()+bond_count].append(b2)  # b2 = a2 --> a1
                              self.b2a.append(atom_count+p-1)
                              self.b2revb.append(b2)
                              self.b2revb.append(b1)
                              self.n_bonds += 2  #化学键看作两条有向边，即左到右，右到左
                              self.bonds.append(np.array([atom_count+mol.GetNumAtoms()+bond_count, atom_count+p-1]))
                              bond_count+=1
                              # break


            atom_count =atom_count+mol.GetNumAtoms()+bond_count
            atom_count_index.append(atom_count)


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim # * 2

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features  多的0？
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features  多的0？
        a2b = [[]]  # mapping from atom index to incoming bond indices   真真正正多的0？？？？
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from  真真正正多的0？？？？
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        bonds = [[0,0]]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]]) #  if b!=-1 else 0

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1], 
                              self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
        
        bonds = np.array(bonds).transpose(1,0)
        
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        
        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.bonds

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(smiles_batch: List[str],
              args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, args)
            if not args.no_cache:
                SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)
    
    return BatchMolGraph(mol_graphs, args)
