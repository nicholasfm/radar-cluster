import numpy as np
import os
#from sklearn.mixture import GaussianMixture
#from netCDF4 import num2date, date2num
#import math
#from multiprocessing import Pool
#from multiprocessing import cpu_count
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory

import pandas as pd
from sklearn.mixture import GaussianMixture
import pickle

def fit_GMM(n):
    model = GaussianMixture(n, covariance_type='full', random_state=0).fit(training_data)
    pickle.dump(model, open(outdirr+savename+str(n)+'.gmm', 'wb'))
    return
savename = 'Mt_Bolton_altitude'
outdirr = './models/'
datafile = './training_data/Mt_Bolton_height.h5'

if __name__ == '__main__':

    df = pd.read_hdf(datafile)
    training_data = df.values
    #print(np.shape(training_data))

    n_components = np.arange(4, 18)
    models = []


    Parallel(n_jobs=16, max_nbytes=1e6)(
        delayed(has_shareable_memory)(
            fit_GMM(n_components[i]))
        for i in range(len(n_components)))
