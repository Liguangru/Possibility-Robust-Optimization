#Test patient: Anonymized_RT260745
import numpy as np
from pathlib import Path

output_folder = Path(__file__).resolve().parent

# 1. read .npy
npy_path = os.path.join(output_folder, "test_fr_dose.npy")
#np.save(npy_path, dose_array)
arr_npdose = np.load(npy_path)
#print(f"for eval fr dose, type is {type(arr_npdose)}, dtype is {arr_npdose.dtype}, size is {arr_npdose.shape}, max is {arr_npdose.max()/100}, min is {arr_npdose.min()/100}, median is {np.median(arr_npdose[arr_npdose != 0])/100}")

# 2. read and reshape .txt
txt_path = os.path.join(output_folder, f"test_fr_dose.txt")
#dose_array.tofile(txt_path, sep = '\r\n')
arr_flat = np.fromfile(txt_path, dtype = np.float32, sep = '\r\n')
arr_reshape3D = arr_flat.reshape(arr_npdose.shape)
print(f"for plan fr dose, type is {type(arr_reshape3D)}, dtype is {arr_reshape3D.dtype}, size is {arr_reshape3D.shape}, max is {arr_reshape3D.max()/100}, min is {arr_reshape3D.min()/100}, median is {np.median(arr_reshape3D[arr_reshape3D != 0])/100}")