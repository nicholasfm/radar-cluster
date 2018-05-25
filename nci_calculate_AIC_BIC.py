import numpy as np
import os
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
from sklearn.mixture import GaussianMixture
import pickle
import traceback
import sys

#models_fn = './models/Mt_Bolton_altitude_4_12.gmm'
#savename = 'Mt_Bolton_altitude_4_12'
#outdirr = './'
#datafile = './training_data/Mt_Bolton_height.h5'
test = False

def calc_AIC(n):
    training_data = np.memmap('./training_data/memmap.dat', dtype='float32', mode='r',shape=memmap_shape)

    models = pickle.load(open(models_fn, 'rb'))
    models = sorted(models, key=lambda k: k.n_components)

    m = models[n]

    aic = m.aic(training_data)
    return [aic, m.n_components]

def calc_BIC(n):
    training_data = np.memmap('./training_data/memmap.dat', dtype='float32', mode='r',shape=memmap_shape)

    models = pickle.load(open(models_fn, 'rb'))
    models = sorted(models, key=lambda k: k.n_components)

    m = models[n]

    bic = m.bic(training_data)
    return [bic, m.n_components]

if __name__ == '__main__':

    models_fn = sys.argv[1]
    datafile = sys.argv[2]
    #Load pandas HDF
    df = pd.read_hdf(datafile)

    if test:
        df = df.iloc[:10,:]

    #Save a numpy memmap for use inside the pool
    memmap_shape = df.shape
    training_data = np.memmap('./training_data/memmap.dat', dtype='float32', mode='w+', shape=memmap_shape)
    training_data[:] = df.values[:]
    del df
    del training_data

    #Allocate pool CPUs (this is memory intensive, so only use 4 CPUs but 60Gb of RAM)
    p = Pool(4)

    #Load the models before running the pool map
    models = pickle.load(open(models_fn, 'rb'))
    models = sorted(models, key=lambda k: k.n_components)

    # Calculate AIC score
    try:
        AIC = p.map(calc_AIC, range(len(models)))
    except Exception as e:
        print(e)
        traceback.print_exc()
        p.terminate()

    # Calculate BIC score
    try:
        BIC = p.map(calc_BIC, range(len(models)))
    except Exception as e:
        print(e)
        traceback.print_exc()
        p.terminate()

    pickle.dump(AIC, open(models_fn[:-4]+'.aic', 'wb'))
    pickle.dump(BIC, open(models_fn[:-4]+'.bic', 'wb'))
