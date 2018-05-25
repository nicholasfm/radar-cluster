import numpy as np
import os
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
from sklearn.mixture import GaussianMixture
import pickle
import traceback
import sys

#Enter paths
#savename = 'Mt_Bolton_altitude_13_20'
outdirr = './models/'
#datafile = './training_data/Mt_Bolton_height.h5'
test = True

nproc = 4
#walltime =

def fit_GMM(n):
    '''
    Function to fit a (sklearn) GaussianMixture model (GMM) using pandas hdf5 file,
    to be used inside multiprocessing pool
    '''
    df = pd.read_hdf(datafile)

    if test:
        df = df.iloc[:10,:]

    #training_data = np.memmap('./training_data/memmap.dat', dtype='float32', mode='r',shape=memmap_shape)
    model = GaussianMixture(n, covariance_type='full', random_state=0).fit(df.values[:])
    out_path = outdirr + os.path.basename(datafile)[:-3] + '_n' + str(n) + '.gmm'
    pickle.dump(model, open(out_path, 'wb'))
    return out_path

if __name__ == '__main__':

    #if len(sys.argv) > 1:

    datafile = sys.argv[1]
    #Set the range of n values to be used in the GMM
    n_components = np.arange(int(sys.argv[2]), int(sys.argv[3]))

    #Load pandas HDF
    #df = pd.read_hdf(datafile)

    #Test flag for subsetting a smaller dataset to confirm everything's working
    #if test:
    #    df = df.iloc[:10,:]
        #n_components = np.arange(4, 6)

    #Save a numpy memmap for use inside the pool

    #Allocate pool CPUs (this is memory intensive, so only use 4 CPUs but 60Gb of RAM)
    p = Pool(nproc)
    models = []
    model_fns = []

    #Run pool map to fit the GMMs
    try:
        model_fns = p.map(fit_GMM, n_components)
    except Exception as e:
        print(e)
        traceback.print_exc()
        p.terminate()

    #Save output GMMs
    #pickle.dump(models, open(outdirr+savename+'.gmm', 'wb'))

    for model_fn in model_fns:
        print(model_fn)
        models.append(pickle.load(open(model_fn, 'rb')))

    models = sorted(models, key=lambda k: k.n_components)
    first_k = str(models[0].n_components)
    last_k = str(models[-1].n_components)

    final_output_fn = outdirr + os.path.basename(datafile)[:-3] + '_' + first_k + '_' + last_k + '_joined.gmm'
    pickle.dump(models, open(final_output_fn, 'wb'))

    print('Success')

    #Enter any cleanup below here (todo: remove memmap)
