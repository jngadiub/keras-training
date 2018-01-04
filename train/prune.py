import sys
import os
from optparse import OptionParser
from keras.models import load_model, Model
from argparse import ArgumentParser
from keras import backend as K
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf
from constraints import ZeroSomeWeights
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"ZeroSomeWeights": ZeroSomeWeights})

# To turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m','--model'   ,action='store',type='string',dest='inputModel'   ,default='train_simple/KERAS_check_best_model.h5', help='input model')
    parser.add_option('--relative-weight-max'   ,action='store',type='float',dest='relative_weight_max'   ,default=None, help='max relative weight')
    parser.add_option('--absolute-weight-max'   ,action='store',type='float',dest='absolute_weight_max'   ,default=None, help='max absolute weight')
    parser.add_option('-o','--outputModel'   ,action='store',type='string',dest='outputModel'   ,default='prune_simple/pruned_model.h5', help='output directory')
    (options,args) = parser.parse_args()

    model = load_model(options.inputModel, custom_objects={'ZeroSomeWeights':ZeroSomeWeights})
    
    weightsPerLayer = {}
    droppedPerLayer = {}
    binaryTensorPerLayer = {}
    allWeights = []
    for layer in model.layers:     
        droppedPerLayer[layer.name] = []
        if layer.__class__.__name__ in ['Dense', 'Convolution1D', 'Convolution2D']:
            original_w = layer.get_weights()
            weightsPerLayer[layer.name] = original_w
            for my_weights in original_w:
                if len(my_weights.shape) < 2: #bias term
                    continue
                #l1norm = tf.norm(my_weights,ord=1)
                elif len(my_weights.shape) == 2: # Dense or Conv1D (?)
                    tensor_abs = tf.abs(my_weights)
                    tensor_reduce_max_1 = tf.reduce_max(tensor_abs,axis=-1)
                    tensor_reduce_max_2 = tf.reduce_max(tensor_reduce_max_1,axis=-1)
                elif len(my_weights.shape) == 3: # Conv2D
                    tensor_abs = tf.abs(my_weights)
                    tensor_reduce_max_0 = tf.reduce_max(tensor_abs,axis=-1)
                    tensor_reduce_max_1 = tf.reduce_max(tensor_reduce_max_0,axis=-1)
                    tensor_reduce_max_2 = tf.reduce_max(tensor_reduce_max_1,axis=-1)
                with tf.Session():
                    #l1norm_val = float(l1norm.eval())
                    tensor_max = float(tensor_reduce_max_2.eval())
                it = np.nditer(my_weights, flags=['multi_index'], op_flags=['readwrite'])                
                binaryTensorPerLayer[layer.name] = np.ones(my_weights.shape)
                while not it.finished:
                    w = it[0]
                    if options.relative_weight_max is not None:
                        allWeights.append(abs(w)/tensor_max)
                    if options.absolute_weight_max is not None:
                        allWeights.append(abs(w))                        
                    if options.relative_weight_max is not None and abs(w)/tensor_max < options.relative_weight_max:
                        #print "small relative weight %e/%e = %e -> 0"%(abs(w), tensor_max, abs(w)/tensor_max)
                        w[...] = 0
                        droppedPerLayer[layer.name].append((it.multi_index, abs(w)))
                        binaryTensorPerLayer[layer.name][it.multi_index] = 0
                    if options.absolute_weight_max is not None and abs(w) < options.absolute_weight_max:
                        #print "small absolute weight %e -> 0"% abs(w)
                        w[...] = 0
                        droppedPerLayer[layer.name].append((it.multi_index, abs(w)))
                        binaryTensorPerLayer[layer.name][it.multi_index] = 0
                    it.iternext()
            #print '%i weights dropped from %s out of %i weights'%(len(droppedPerLayer[layer.name]),layer.name,layer.count_params())
            #converted_w = convert_kernel(original_w)
            converted_w = original_w
            layer.set_weights(converted_w)


    print 'Summary:'
    totalDropped = sum([len(droppedPerLayer[layer.name]) for layer in model.layers])
    for layer in model.layers:
        print '%i weights dropped from %s out of %i weights'%(len(droppedPerLayer[layer.name]),layer.name, layer.count_params())
    print '%i total weights dropped out of %i total weights'%(totalDropped,model.count_params())
    print '%.1f%% compression'%(100.*totalDropped/model.count_params())
    model.save(options.outputModel)
    model.save_weights(options.outputModel.replace('.h5','_weights.h5'))

    # save binary tensor in h5 file 
    h5f = h5py.File(options.outputModel.replace('.h5','_drop_weights.h5'))
    for layer, binary_tensor in binaryTensorPerLayer.iteritems():
        h5f.create_dataset('%s'%layer, data = binaryTensorPerLayer[layer])
    h5f.close()

    # plot the distribution of weights
    allWeightsArray = np.array(allWeights)
    percentiles = [5,32,50,68,95]
    vlines = np.percentile(allWeightsArray,percentiles,axis=-1)
    xmin = np.amin(allWeightsArray)
    xmax = np.amax(allWeightsArray)
    bins = np.linspace(xmin, xmax, 100)
    logbins = np.geomspace(xmin, xmax, 100)
    
    plt.figure()
    plt.hist(allWeightsArray,bins=bins)
    axis = plt.gca()
    ymin, ymax = axis.get_ylim()
    for vline, percentile in zip(vlines, percentiles):
        plt.axvline(vline, 0, 1, color='r', linestyle='dashed', linewidth=1, label = '%s%%'%percentile)
        plt.text(vline, ymax+0.01*(ymax-ymin), '%s%%'%percentile, color='r', horizontalalignment='center')
    plt.ylabel('Number of Weights')
    if options.absolute_weight_max:
        plt.xlabel('Absolute Weights')
    elif options.relative_weight_max:
        plt.xlabel('Absolute Relative Weights')
    plt.savefig(options.outputModel.replace('.h5','_weight_histogram.pdf'))

        
    plt.figure()
    plt.hist(allWeightsArray,bins=logbins)
    plt.semilogx()
    for vline, percentile in zip(vlines, percentiles):
        plt.axvline(vline, 0, 1, color='r', linestyle='dashed', linewidth=1, label = '%s%%'%percentile)
        plt.text(vline, ymax+0.01*(ymax-ymin), '%s%%'%percentile, color='r', horizontalalignment='center')
    plt.ylabel('Number of Weights')
    if options.absolute_weight_max:
        plt.xlabel('Absolute Weights')
    elif options.relative_weight_max:
        plt.xlabel('Absolute Relative Weights')
    plt.savefig(options.outputModel.replace('.h5','_weight_histogram_logx.pdf'))

    
