from pickle_helper import reading_pkl

from sets import Set
import numpy as np
import threading
import time
import os

from relational_network_utilities import print_debug
from relational_network_utilities import time_string

class DatabasesCacheMgr(object):
    instance = None
    
    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance
    
    def __init__(self):        
        self.reset()
        
    def internal_add_path_key(self, path, key):
        if path in self.path_keys_map:
            self.path_keys_map[path].append(key)
        else:
            self.path_keys_map[path] = [key]
        
    def remove_path_key(self, path):
        path = path.rstrip('/')
        
        if not path in self.path_keys_map:
            return
        
        print_debug('DatabasesCacheMgr: Clearing cache for path {}'.format(path))
        
        for key in self.path_keys_map[path]:
            if key in self.data_inputs_map:
                del self.data_inputs_map[key]
                del self.data_targets_map[key]
                
        del self.path_keys_map[path]
        
    def reset(self):
        self.data_inputs_map = {}
        self.data_targets_map = {}
        self.async_databases = Set([])
        self.path_keys_map = {}
        
    def get_key(self, path, batch_size=1, objects_cnt=1, window=1, is_ignore_zero_inputs=False):
        return 'Database path: {} - batch: {} - objs: {} - window: {} - zeros {}'.format(path, batch_size, objects_cnt, window, is_ignore_zero_inputs)

    def get_data(self, path, batch_size=1, objects_cnt=1, window=1, is_ignore_zero_inputs=False):
        path = path.rstrip('/')
        
        key_view = self.get_key(path, batch_size, objects_cnt, window, is_ignore_zero_inputs)
        key_base = self.get_key(path, 1, 1)
        
        if key_view in self.data_inputs_map:
            return self.data_inputs_map[key_view], self.data_targets_map[key_view]
        
        if key_base in self.async_databases:
            print_debug('\n\n\nDatabasesCacheMgr Warning: Another thread wanna access while loading async, Lets sleep at path\n\n\n' + path)
            
            while key_base in self.async_databases:
                time.sleep(3 * 60)
                
            while key_base in self.async_databases:   # For safety as we should protect this variable
                time.sleep(3 * 60)
        
        data_inputs = None
        data_targets = None
        
        if key_base in self.data_inputs_map:
            data_inputs = self.data_inputs_map[key_base]
            data_targets = self.data_targets_map[key_base]
        else:
            data_inputs, data_targets = self.internal_load_db_sync(path, key_base)
        
        data_inputs_view, data_targets_view = self.get_data_view(data_inputs, data_targets, batch_size, objects_cnt, window, is_ignore_zero_inputs)
        self.data_inputs_map[key_view] = data_inputs_view
        self.data_targets_map[key_view] = data_targets_view
        
        self.internal_add_path_key(path, key_base)
        self.internal_add_path_key(path, key_view)
        
        return data_inputs_view, data_targets_view
            
    def load_data_async(self, path, batch_size=1, objects_cnt=1, window=1, is_ignore_zero_inputs=False):
        path = path.rstrip('/')
        
        key_view = self.get_key(path, batch_size, objects_cnt, window, is_ignore_zero_inputs)
        key_base = self.get_key(path, 1, 1)
        
        if key_base in self.data_inputs_map:
            return
        
        self.async_databases.add(key_base)
        
        t = threading.Thread(target=self.internal_load_db_sync, args = (path, key_base))
        t.daemon = True
        t.start()
        
        return key_view
    
    def get_data_view(self, data_inputs, data_targets, batch_size, objects_cnt, window, is_ignore_zero_inputs):
        if len(data_inputs) == 0:
            raise NameError('DatabasesCacheMgr: Empty data!')
        
        if len(data_inputs) % objects_cnt != 0:
            raise NameError('DatabasesCacheMgr: can not reshape width {} to {}'.format(len(data_inputs), objects_cnt))
            
        data_inputs_view = []
        data_targets_view = []
        
        pos = 0;
        while pos < len(data_inputs):
            batch_inputs = np.array([])
            batch_targets = np.array([])
        
            for batch_idx in range(batch_size):
                if pos >= len(data_inputs):
                    break
                    
                group_inputs = np.array([])
                    
                # read persons * window (each person t steps consective) and rearrange
                # build a group of consecutive vectors, e.g. 12 feature vector of players as scene representation
                for obj_idx in range(objects_cnt):
                    person_inputs = np.array([])
                    
                    for w in range(window):
                        if pos >= len(data_inputs):
                            raise NameError('DatabasesCacheMgr: Incomplete data, cur lenth={}'.format(len(data_inputs)))
                        input = data_inputs[pos]
                        target = data_targets[pos][0]
                        pos = pos+1
                        
                        if person_inputs.size == 0:
                            person_inputs = input
                        else:
                            person_inputs = np.concatenate((person_inputs, input), axis=0)                        
                        
                    if group_inputs.size == 0:
                        group_inputs = person_inputs
                    else:
                        group_inputs = np.concatenate((group_inputs, person_inputs), axis=2)

                if group_inputs.size > 0:
                    if batch_inputs.size == 0:
                        batch_inputs = group_inputs
                    else:
                        batch_inputs = np.concatenate((batch_inputs, group_inputs), axis=0)
                        
                    for w in range(window):
                        batch_targets = np.concatenate((batch_targets, np.array([target])))
                                
            data_inputs_view.append(batch_inputs)
            data_targets_view.append(batch_targets)
        
        return data_inputs_view, data_targets_view
        
    def internal_load_db_sync(self, path, key_base):
        start_time = time.time()
        
        print_debug('Reading from pickle file {}'.format(path))
        
        if not os.path.exists(path):
            raise NameError('DatabasesCacheMgr: Pickle file does NOT exit at path {}'.format(path))
        
        data_inputs, data_targets = reading_pkl(path)
            
        self.data_inputs_map[key_base] = data_inputs
        self.data_targets_map[key_base] = data_targets
        
        self.internal_add_path_key(path, key_base)
        
        if key_base in self.async_databases:
            self.async_databases.remove(key_base)

        print_debug("Total reading time took {} for {}".format(time_string(time.time() - start_time), path))
            
        return data_inputs, data_targets        
        