# radar-cluster
GMM clustering method, initially developed for use with the BCPE Mt Bolton Data.


## Building training datasets

Build h5 training datasets with `Generate Training Data for Clustering Models Mt Bolton` notebook. This iterates through the raw scans/volumes with PyArt, and builds a pandas DataFrame, which is saved using pandas .h5 routines (for speedup). Uses multiprocessing with a `handle output` function to join then save.

## Model Training

Models were trained on NCI using scripts, not notebooks. See `nci_fit_GMM.py` script. This is designed to be used standalone with 1) input pandas dataframe in h5, 2) minimum k and 3) maximum k as command line arguments. Number of processors is currently specific determined in the top of the script with `nproc` but memory increases with threading so be careful.

**Note:** In the head of the file is a flag for testing, set to `True` by default. It only takes a subset of 10 rows to fit the model to test everything is working.

The models are pickled individually and then again as a list if the script runs to completion using the dataframe h5 name as a template.

## Post processing

Calculating the standard AIC and BIC measures can be compute intensive on large datasets. Use the `nci_calculate_AIC_BIC.py` script to evaluate these.

Output is pickled from a list ready to be plotted coming in the form `[ k, AIC or BIC value]`

## Plotting

A number of plotting notebooks are also present,the main being `Mt Bolton Clustering - Analysis` and `Analyse Clusterings - Spiders and Animations`. Animations are currently done as html5 movies as they are imbedded in the notebooks, but can easy be changed to output mp4.
