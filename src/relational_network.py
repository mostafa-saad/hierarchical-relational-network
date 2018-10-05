import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import SliceLayer 

from relational_clique_layer import RelationalCliqueLayer
from databases_cache_mgr import DatabasesCacheMgr

from relational_network_utilities import utility_train
from relational_network_utilities import print_debug
from global_settings import GlobalSettings

SIZE1=256
SIZE2=128
dropout_val=0.5

W_init_func = lasagne.init.GlorotNormal()
db_cache_mgr = DatabasesCacheMgr.get_instance()
global_settings = GlobalSettings.get_instance()
    
class RelationalNetwork(object):
    def __init__(self,):
        global global_settings
        
        self.network = None        
        self.train_fn = None
        self.val_fn = None
        self.input_var = None
        self.target_var = None
        
        self.batch_size = 250                
        self.trainval_path = None
        self.test_path = None
        self.model_name = None
        self.dataset_name = '/volleyball'
        self.persons_cnt = 12
        self.labels_cnt = 8     # of scene labels
        
    def init_paths(self):
        global global_settings
        
        self.trainval_path = global_settings.data_dir + '/trainval.pkl'
        self.test_path = global_settings.data_dir + '/test.pkl'
        
    ############################################################################
    # Some utilities
                            
    def util_add_dense_layer(self, layer, dense_size, nonlinearity=lasagne.nonlinearities.rectify):
        return lasagne.layers.DenseLayer(
            lasagne.layers.DropoutLayer(layer, p=dropout_val), num_units=dense_size,
            W=W_init_func, nonlinearity=nonlinearity)
                
    def util_add_dense(self, dense_size, nonlinearity=lasagne.nonlinearities.rectify):
        self.network = self.util_add_dense_layer(self.network, dense_size, nonlinearity)     
    
    def util_add_relational_clique_layer(self, input_layer_or_layers, size_from, size_to, ppl_cnt):
        inp = lasagne.layers.InputLayer((None, 2*size_from, ppl_cnt, ppl_cnt))
        nin = lasagne.layers.NINLayer(inp, num_units = size_to, W=W_init_func)
        nin = lasagne.layers.NINLayer(nin, num_units = size_to, W=W_init_func, nonlinearity=lasagne.nonlinearities.rectify)
        
        if not isinstance(input_layer_or_layers, list):
            out_layer = RelationalCliqueLayer(input_layer_or_layers, subnet = nin)
            return out_layer
        
        ret_layers = []        
        for input_layer in input_layer_or_layers:
            out_layer = RelationalCliqueLayer(input_layer, subnet = nin)           
            ret_layers.append(out_layer)
            
        return ret_layers
        
    
    def util_slice_layer(self, layer, persons_cnt, factor):
        g_sz = persons_cnt//factor
        
        layers = []
        
        for i in range(factor):
            layer_i = SliceLayer(layer, indices=slice(i*g_sz, (i+1)*g_sz), axis=2)
            layers.append(layer_i)
              
        return layers     
      
    ############################################################################  
    def network_inp_prepare(self):
        self.input_var = T.tensor3('inputs')
        self.target_var = T.ivector('targets')
        
        input_layer = lasagne.layers.InputLayer((None, global_settings.input_person_feat_sz, self.persons_cnt), input_var = self.input_var)
        self.network = lasagne.layers.DropoutLayer(input_layer, p=dropout_val)
                
    # Pairwsie model (RCRG-2R-11C) - Fully connected graph
    def network_build_layers_v1(self, r1 = SIZE1, r2 = SIZE2, d1 = SIZE1):    #ID10091
        self.hidden1_sz, self.hidden2_sz, self.dense1_sz = r1, r2, d1
        
        print_debug('Method Called: '+sys._getframe().f_code.co_name)
        print_debug('Info: Relational layer Sizes: {} {} - Dense {} - RCRG-2R-11C'.format(self.hidden1_sz, self.hidden2_sz, self.dense1_sz))
        
        self.model_name = 'RCRG-2R-11C'
        self.init_paths()        
        self.network_inp_prepare()
        
        self.network = self.util_add_relational_clique_layer(self.network, global_settings.input_person_feat_sz, self.hidden1_sz, self.persons_cnt)
        self.network = self.util_add_relational_clique_layer(self.network, self.hidden1_sz, self.hidden2_sz, self.persons_cnt)
        
        self.util_add_dense(self.dense1_sz)
        self.util_add_dense(self.labels_cnt, lasagne.nonlinearities.softmax)     
           
        self.network_build_classification_loss_scene()
        
    # Hierarchical model (RCRG-3R-421C) - 3 relational layers, first is 4 cliques, 2nd is 2 cliques and 3rd is fully connected graph
    # Input is assumed: 12 players ordered such that first 6 is team 1 and the remaining is team 2
    # The first 6 assumed to be first 3 clique and 2nd 3 people are another clique
    # An approximation for that, is sort all players based on (x, y) of each bounding box (null for missing persons)
    def network_build_layers_v2(self, r1 = 2*SIZE1, r2 = SIZE2, r3 = SIZE1, d1 = SIZE1):        #ID100111
        self.hidden1_sz, self.hidden2_sz, self.hidden3_sz, self.dense1_sz = r1, r2, r3, d1
        
        print_debug('Method Called: '+sys._getframe().f_code.co_name)
        print_debug('Info: Relational layer Sizes: {} {} {} - Dense {} - RCRG-3R-421C'.format(self.hidden1_sz, self.hidden2_sz, self.hidden3_sz, self.dense1_sz))
        
        self.model_name = 'RCRG-3R-421C'
        self.init_paths()
        self.network_inp_prepare()
                
        g_sz = self.persons_cnt//4  # 4 cliques
        cliques4g = self.util_slice_layer(self.network, self.persons_cnt, 4)
        cliques4g = self.util_add_relational_clique_layer(cliques4g, global_settings.input_person_feat_sz, self.hidden1_sz, g_sz)
        
        g_sz = 2*g_sz   # merge the first 2 cliques and last 2 cliques => overall 2 cliques, one per team
        cliques2g = [lasagne.layers.ConcatLayer([cliques4g[0], cliques4g[1]], axis=2), 
                    lasagne.layers.ConcatLayer([cliques4g[2], cliques4g[3]], axis=2) ]
        cliques2g = self.util_add_relational_clique_layer(cliques2g, self.hidden1_sz, self.hidden2_sz, g_sz)
        
        g_sz = 2*g_sz   # merge the last 2 cliques => 1 clique, fully connected layer
        cliques1g = lasagne.layers.ConcatLayer([cliques2g[0], cliques2g[1]], axis=2)
        cliques1g = self.util_add_relational_clique_layer(cliques1g, self.hidden2_sz, self.hidden3_sz, g_sz)
        self.network = cliques1g

        # For temporal w > 1, replace dense with RNN layer
        self.util_add_dense(self.dense1_sz)
        self.util_add_dense(self.labels_cnt, lasagne.nonlinearities.softmax)    
        
        self.network_build_classification_loss_scene()
        
    def network_build_classification_loss_scene(self):
        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, self.target_var)
        loss = loss.mean()
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        
        self.lr = theano.shared( np.cast['float32'](1e-4))  # actual value later
        updates = lasagne.updates.adam(loss,params,learning_rate = self.lr)
        
        self.test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(self.test_prediction, self.target_var)
        test_loss = test_loss.mean()
        
        self.classification_results=T.argmax(self.test_prediction, axis=1)
        self.test_acc = T.mean(T.eq(self.classification_results, self.target_var), dtype=theano.config.floatX)

        self.train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates, allow_input_downcast=True) #TODO
        self.val_fn = theano.function([self.input_var, self.target_var], [test_loss, self.test_acc, self.test_prediction], allow_input_downcast=True)
                
    def network_train(self):        
        self.test_inputs, self.test_targets = db_cache_mgr.get_data(self.test_path, 1, self.persons_cnt, 1)
        self.training_inputs, self.training_targets = db_cache_mgr.get_data(self.trainval_path, self.batch_size, self.persons_cnt, 1)
        
        utility_train(self, self.training_inputs, self.training_targets, self.test_inputs, self.test_targets)

if __name__ == '__main__':    
    lasagne.random.set_rng(np.random.RandomState(10170))
    
    semantic_ae = RelationalNetwork()   
    semantic_ae.network_build_layers_v1()    
    semantic_ae.network_train()
    
    global_settings.finalize()

