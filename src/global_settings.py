import numpy as np
import theano

class GlobalSettings(object):
    instance = None
    
    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance
    
    def __init__(self):
        self.is_sample_test = True
        
        self.network_feature_name = 'vgg19-tuned'
        self.input_person_feat_sz = 4096
        
        self.lr = theano.shared(np.cast['float32'](0.1))
        
        self.dataset_name = 'volleyball'
        self.data_dir = 'data'
            
        self.test_factor=50
        self.decay_times=1      # decay the rate cnt? 1 e.g. from 1e-4 to 1e-5 no more
        self.lr_inital=1e-4
        self.num_epochs=200        # 200
        self.lr_decay_factor=100    # 100
        
        self.logger = open('logs/log.txt', 'w')

        if self.is_sample_test:
            print 'NOTICE: is_sample_test = TRUE, using fewer iterations. For real test, switch to false\n'
            self.num_epochs=6
            self.lr_decay_factor=6
            self.test_factor=3
        
    
    def finalize(self):
        self.logger.flush()
        self.logger.close() 