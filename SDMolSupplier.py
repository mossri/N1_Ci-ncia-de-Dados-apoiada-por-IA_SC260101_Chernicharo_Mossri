from rdkit import Chem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import sys

def sdf_to_mols(sdf_filename):
    """
    Reads an SDF file and returns a list of RDKit molecule objects.
    """
    # Use a context manager for good practice
    with Chem.SDMolSupplier(sdf_filename) as suppl:
        mols = []
        for mol in suppl:
            # Check if the molecule was read successfully
            if mol is not None:
                mols.append(mol)
        return mols

# Usage example:
filename = 'phase_3_training_docked.sdf'
molecules_list = sdf_to_mols(filename)
 
    
    
nbits=512
Radius=2
fmgen=rdFingerprintGenerator.GetMorganGenerator(radius=Radius,fpSize=nbits,includeChirality=True)


#header_zeros=np.zeros(nbits, dtype=int)
#header_str=header_zeros.astype('U')
#header_score=np.insert(header_str,0,"score")
#header_line="".join(header_str)

#header_score="score"

#with open("destiny_bit.txt", "a") as f:      
#            f.write(header_line+"\n")

#with open("destiny_score.txt", "a") as f:      
#            f.write(header_score+"\n")

for mol in molecules_list:
        arg = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fmgen.GetFingerprint(mol),arg)
        #array_line=[mol.GetProp('FRED Chemgauss4 score')]+arg
        bit_array=arg
        score=mol.GetProp('FRED Chemgauss4 score')
        with np.printoptions(threshold=sys.maxsize):
            bit_int=bit_array.astype('int')
            bit_str=bit_int.astype('U')
            bit_line="".join(bit_str)
        
        with open("training_bit.txt", "a") as f:      
            f.write(bit_line+"\n")
            
        with open("training_score.txt", "a") as f:      
            f.write(score+"\n")
