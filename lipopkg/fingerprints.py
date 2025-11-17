import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs

DEFAULT_SMILES_COL = "smiles"

def smiles_to_mols(smiles_list):
    """Convert a list of SMILES strings to RDKit Mol objects."""
    mols = []
    for smi in smiles_list:
        try:
            m = Chem.MolFromSmiles(smi)
        except Exception:
            m = None
        mols.append(m)
    return mols

def mols_to_morgan(mols, n_bits=2048, radius=2):
    """Convert a list of Mol objects to Morgan fingerprints."""
    X_list = []
    for m in mols:
        if m is None:
            arr = np.zeros((n_bits,), dtype=np.int8)
        else:
            bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
        X_list.append(arr)
    X = np.asarray(X_list, dtype=np.float32)
    return X

def mols_to_maccs(mols):
    """Convert a list of Mol objects to MACCS keys fingerprints."""
    X_list = []
    # get bit length from a dummy molecule
    bv_dummy = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles("CC"))
    n_bits = bv_dummy.GetNumBits()

    for m in mols:
        if m is None:
            arr = np.zeros((n_bits,), dtype=np.int8)
        else:
            bv = MACCSkeys.GenMACCSKeys(m)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
        X_list.append(arr)

    X = np.asarray(X_list, dtype=np.float32)
    return X
