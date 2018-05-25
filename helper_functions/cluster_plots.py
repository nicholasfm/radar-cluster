# For making a spider plot (not radar, to avoid confusion).
# Adapted from: https://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def prep_spider(means,ranges,offsets):

    label_ranges = {}
    for key, values in ranges.iteritems():
        #ZDR and DBZH should be the only values with negative values in their ranges.

        label_ranges[key] = values
        if key == 'ZDR':

            ranges[key] = [x + offsets['ZDR'] for x in values]
            values = ranges[key]

        elif key == 'DBZH':

            ranges[key] = [x + offsets['DBZH'] for x in values]
            values = ranges[key]

        # The radial grid won't accept non-positive numbers, so we also need to
        # add a pertubation to where 0 is the lower end on of the range

        if values[0] == 0.:
            ranges[key] = [x + offsets['zero'] for x in values]

    means_offset = means
    means_offset['ZDR'] = means['ZDR'] + offsets['ZDR']
    means_offset['DBZH'] = means['DBZH'] + offsets['DBZH']
    means_offset['KDP'] = means['KDP'] + offsets['zero']

    return means_offset, ranges, label_ranges

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        try:
            assert (y1 <= d <= y2) or (y2 <= d <= y1)
        except Exception as e:
            print('Value ' + str(d) + ' is  out of range ' + str(y1) + ' ' + str(y2))
            print(e)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1)
                     * (x2 - x1) + x1)
    return sdata

class ComplexSpider():
    def __init__(self, fig, variables, ranges,label_ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):

            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)

            grid_for_labels = np.linspace(*label_ranges[i],
                               num=n_ordinate_levels)

            gridlabel = ["{}".format(round(x,1))
                         for x in grid_for_labels]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)




from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""

    ax = ax or plt.gca()
    #print(covariance)


    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    #print(width)
    #print(height)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm_scatter(gmm, df, n_samples=1000, ax=None, x_var='DBZH', y_var='ZDR',cmap='Dark2'):
    X = df.sample(n_samples).values
    x_index = [i for i, s in enumerate(list(df)) if x_var in s]
    y_index = [i for i, s in enumerate(list(df)) if y_var in s]

    print(x_index)
    print(y_index)

    ax = ax or plt.gca()
    labels = gmm.predict(X)
    #if label:
    ax = ax.scatter(X[:,x_index], X[:,y_index], c=labels, s=40, cmap=cmap, zorder=2)
    #else:
    #    ax = ax.scatter(X[:, x_var], X[:, y_var], s=40, zorder=2)
    #ax.axis('equal')

    #plt.legend(loc='lower right')

    #This assumes that df columns are in the same order as in the model

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_[:,[x_index,y_index]], gmm.covariances_[:,[x_index,y_index]], gmm.weights_):
        draw_ellipse(pos, covar[:,[x_index,y_index]], alpha=w * w_factor)

    #ax.set_ylim(-10., 10.)
    plt.ylabel(y_var)
    plt.xlabel(x_var)


    #legend = ax.get_legend()
    #plt.legend()
    #plt.legend(colors,loc='lower right')
