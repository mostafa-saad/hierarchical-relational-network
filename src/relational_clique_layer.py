# Moustafa Ibrhaim Comments
# Code Original Source: https://github.com/arayabrain/PermutationalNetworks, layer is renamed for convenience 
# If we have K players and their connectivity graph is K cliques, 
# then each clique is pairwise connections, which what this layer simply does
# Our proposed layer is generalization of this work, 
# though implementation wise we follow simple clique style which can directly utilize this code  

import lasagne
import theano.tensor as T
from lasagne.layers import helper

class RelationalCliqueLayer(lasagne.layers.Layer):
    def __init__(self,incoming,subnet,pooling='mean',**kwargs):
        super(RelationalCliqueLayer, self).__init__(incoming, **kwargs)
        self.subnet = subnet
        self.pooling = pooling
        
    # Moustafa Ibrhaim
    # MostChange: FIX BUG: get_output_for() got an unexpected keyword argument 'deterministic'
    # Adding **kwargs: https://groups.google.com/forum/#!topic/lasagne-users/PTJy8Tut3WM
    def get_output_for(self, input, **kwargs):
		rs = input.reshape((input.shape[0], input.shape[1], input.shape[2], 1)) # B,V,S,1
		z1 = T.tile( rs, (1,1,1,input.shape[2]))
		z2 = z1.transpose((0,1,3,2))
		Z = T.concatenate([z1,z2],axis=1)
		Y = helper.get_output(self.subnet, Z)
		if self.pooling == 'mean':
			return T.mean(Y,axis=3)
		elif self.pooling == 'max':
			return T.max(Y,axis=3)
		else: return self.pooling(Y)

    def get_params(self, **tags):
		# Get all parameters from this layer, the master layer
		params = super(RelationalCliqueLayer, self).get_params(**tags)
		# Combine with all parameters from the child layers
		params += helper.get_all_params(self.subnet, **tags)
		return params

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.subnet.output_shape[1], input_shape[2])
