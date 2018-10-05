import os
import pickle

def reading_file(pkl_path):
    if not os.path.exists(pkl_path):
        raise NameError('DatabasesCacheMgr: Pickle file does NOT exit at path {}'.format(pkl_path))
            
    fileObject = open(pkl_path,'r')
    data_inputs = pickle.load(fileObject)
    data_targets = pickle.load(fileObject)
    fileObject.close()
    
    return data_inputs, data_targets

def reading_pkl(pkl_path):
    if not os.path.isdir(pkl_path):
        return reading_file(pkl_path)
        
    data_inputs = []
    data_targets = []
    
    for idx in range(1, 999):
        path = pkl_path + "/" + str(idx).zfill(3)
        
        if not os.path.exists(path):
            break
            
        data_inputs1, data_targets1 = reading_file(path)
        data_inputs.extend(data_inputs1)
        data_targets.extend(data_targets1)
    
    return data_inputs, data_targets
    