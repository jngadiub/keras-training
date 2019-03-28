import setGPU
import sys
import h5py
import glob
import numpy as np

# keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from keras.layers import Concatenate, Reshape, BatchNormalization
from keras.layers import MaxPooling2D, MaxPooling3D
from keras.optimizers import Adam, Nadam, Adadelta
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.regularizers import l1
import models
from models import *
from train import *
from optparse import OptionParser

# hyperparameters
import GPy, GPyOpt

####################################################

from keras.activations import relu
# myModel class
class myModel():
    def __init__(self, x_train, x_test, y_train, y_test, optmizer_index=0, DNN_neurons=40, 
                 DNN_layers=2, batch_size=100, learning_rate_index=0, epochs=50):
        #self.optimizer = ['adam', 'nadam','adadelta']
	self.learning_rate = [0.01,0.001,0.0001,0.00001]
        self.optimizer = [Adam(lr=self.learning_rate[learning_rate_index]), Nadam(lr=self.learning_rate[learning_rate_index]), Adadelta(lr=self.learning_rate[learning_rate_index])]
        self.optimizer_index = optmizer_index
        self.DNN_neurons = DNN_neurons
        self.DNN_layers = DNN_layers
        self.batch_size = batch_size
        self.learning_rate_index = learning_rate_index
        self.epochs = epochs
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = x_train, x_test, y_train, y_test
        self.__model = self.build()
    
    #  model
    def build(self):

        model = Sequential()

        model.add(BinaryDense(self.DNN_neurons, H=1, use_bias=False, name='fc0', input_shape=(16,)))
        model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn0'))
        model.add(Activation(binary_tanh, name='act{}'.format(0)))

        for i in range(1,self.DNN_layers):
         model.add(BinaryDense(self.DNN_neurons, H=1, use_bias=False, name='fc%i'%i))
         model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn%i'%i))
         model.add(Activation(binary_tanh, name='act{}'.format(i)))

        model.add(BinaryDense(5, H=1, use_bias=False, name='output'))
        model.add(BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn'))

        model.compile(optimizer=self.optimizer[self.optimizer_index],loss='squared_hinge', metrics=['acc'])

        return model
    
    # fit model
    def model_fit(self):

       #outdir = 'hpscan_opt_'+str(int(self.optimizer_index))+'_lyrs_'+str(int(self.DNN_layers))+'_neurs_'+str(int(self.DNN_neurons))+'_lr_'+str(float(self.learning_rate))+'_batch_'+str(int(self.batch_size))
       outdir = 'hpscan_opt_%i_lyrs_%i_neurs_%i_lr_%.7f_batch_%i'%(self.optimizer_index,self.DNN_layers,self.DNN_neurons,self.learning_rate[self.learning_rate_index],self.batch_size)

       callbacks=all_callbacks(stop_patience=1000, 
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001, 
                            lr_cooldown=2, 
                            lr_minimum=0.0000001,
                            outputDir=outdir)

       self.__model.fit(self.__x_train, self.__y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        verbose=0,
                        validation_data=[self.__x_test, self.__y_test],
                        callbacks = callbacks.callbacks)       
    
    # evaluate  model
    def model_evaluate(self):
        self.model_fit()
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, 
                                           batch_size=self.batch_size, verbose=0)
        return evaluation


####################################################

# Runner function for model
# function to run  class

def run_model(x_train, x_test, y_train, y_test, optmizer_index=0, DNN_neurons=40, 
              DNN_layers=2, batch_size=100, learning_rate_index=0, epochs=50):
    
    _model = myModel(x_train, x_test, y_train, y_test, optmizer_index, 
                     DNN_neurons, DNN_layers, batch_size, learning_rate_index, epochs)
    model_evaluation = _model.model_evaluate()
    return model_evaluation

####################################################

parser = OptionParser()
parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='../data_raw/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z', help='input file')
parser.add_option('-t','--tree'   ,action='store',type='string',dest='tree'   ,default='t_allpar_new', help='tree name')
parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
parser.add_option('-c','--config'   ,action='store',type='string', dest='config', default='train_config_threelayer.yml', help='configuration file')
(options,args) = parser.parse_args()
     
yamlConfig = parse_config(options.config)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test, labels  = get_features(options, yamlConfig)

n_epochs = 50
# Bayesian Optimization

# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'optmizer_index',        'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'DNN_neurons',           'type': 'discrete',   'domain': (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)},
          {'name': 'DNN_layers',            'type': 'discrete',   'domain': (1, 2, 3, 4, 5)},
          {'name': 'batch_size',            'type': 'discrete',   'domain': (500,1000,2000,3000,4000)},
          {'name': 'learning_rate_index',   'type': 'discrete',   'domain': (0.01,0.001,0.0001,0.00001)}]
#bounds = [{'name': 'optmizer_index',        'type': 'discrete',   'domain': (0, 1, 2)},
#          {'name': 'DNN_neurons',           'type': 'discrete',   'domain': (10)},
#          {'name': 'DNN_layers',            'type': 'discrete',   'domain': (1)},
#          {'name': 'batch_size',            'type': 'discrete',   'domain': (500)},
#          {'name': 'learning_rate',         'type': 'discrete',   'domain': (0.01, 0.001, 0.0001)}]

# function to optimize model
def f(x):
    print(x)
    evaluation = run_model(X_train, X_test, Y_train, Y_test,
                           optmizer_index = int(x[:,0]), 
                           DNN_neurons = int(x[:,1]), 
                           DNN_layers = int(x[:,2]),
                           batch_size = int(x[:,3]),
                           learning_rate_index = int(x[:,4]),
                           epochs = n_epochs)
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]

# run optimization
opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
opt_model.run_optimization(max_iter=10000)

print("x:",opt_model.x_opt)
print("y:",opt_model.fx_opt)

# print optimized model
print("""
Optimized Parameters:
\t{0}:\t{1}
\t{2}:\t{3}
\t{4}:\t{5}
\t{6}:\t{7}
\t{8}:\t{9}
\t{10}:\t{11}
""".format(bounds[0]["name"],opt_model.x_opt[0],
           bounds[1]["name"],opt_model.x_opt[1],
           bounds[2]["name"],opt_model.x_opt[2],
           bounds[3]["name"],opt_model.x_opt[3],
           bounds[4]["name"],opt_model.x_opt[4]))
print("optimized loss: {0}".format(opt_model.fx_opt))

print(opt_model.x_opt)
