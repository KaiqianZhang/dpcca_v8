"""=============================================================================
Utility functions for plotting experimental results.
============================================================================="""

import csv
import os

if os.path.dirname(os.path.realpath(__file__)) == '/scratch/gpfs/gwg3/dmcm':
    import matplotlib
    matplotlib.use('agg')

import matplotlib.pyplot as plt
from   matplotlib.offsetbox import OffsetImage, AnnotationBbox
from   matplotlib.patches import Ellipse
from   PIL import Image
import seaborn

import numpy as np
from   scipy import linalg
from   sklearn import decomposition, manifold

# ------------------------------------------------------------------------------

# Colors from: http://colorbrewer2.org/.
COLORS = [
    '#a6cee3', '#b15928', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
    '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#99995b', '#1f78b4'
]
C_IDX = 0

# ------------------------------------------------------------------------------

class DummyCompressor():
    def fit_transform(self, x): return x

# ------------------------------------------------------------------------------

def imscatter(points, images, ax, zoom=1, frameon=True):
    for (x, y), image in zip(points, images):
        try:
            image = np.asarray(Image.fromarray(image).convert('RGB'))
        except:
            pass
        im = OffsetImage(image.T, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=frameon)
        ax.add_artist(ab)

# ------------------------------------------------------------------------------

def plot_both_embeddings(Zs, subtitles=['Images', 'Genes'], suptitle=None,
                         dataset=None, images=None, comp_mode='pca',
                         images_both=False, zoom=0.25, frameon=True):

    fig, axes = plt.subplots(1, len(Zs), dpi=72)
    fig.set_size_inches(20, 10)  # Width, height

    if comp_mode == 'pca':
        compressor = decomposition.PCA(n_components=2)
    elif comp_mode == 'tsne':
        compressor = manifold.TSNE(n_components=2)
    elif comp_mode == None:
        compressor = DummyCompressor()

    if len(Zs) == 1:
        points = compressor.fit_transform(Zs[0])
        x_label, y_label = _get_labels(compressor)
        _plot_both_embeddings(points, axes, None, dataset, images, True,
                              x_label, y_label, zoom, frameon)
    elif dataset is not None and images is not None:
        Z1, Z2 = Zs
        subt1, subt2 = subtitles
        # Print images with images and genes with classes.

        points = compressor.fit_transform(Z1)
        x_label, y_label = _get_labels(compressor)
        _plot_both_embeddings(points,  axes[0], subt1, None, images, False,
                              x_label, y_label, zoom, frameon)

        points = compressor.fit_transform(Z2)
        x_label, y_label = _get_labels(compressor)
        if images_both:
            _plot_both_embeddings(points, axes[1], subt2, None, images, False,
                                  x_label, y_label, zoom, frameon)
        else:
            _plot_both_embeddings(points, axes[1], subt2, dataset, None, True,
                                  x_label, y_label, zoom, frameon)
    else:
        Z1, Z2 = Zs
        subt1, subt2 = subtitles
        ax1, ax2 = axes

        points = compressor.fit_transform(Z1)
        x_label, y_label = _get_labels(compressor)
        _plot_both_embeddings(points, ax1, subt1, dataset, images, False,
                              x_label, y_label, zoom, frameon)

        points = compressor.fit_transform(Z2)
        x_label, y_label = _get_labels(compressor)
        _plot_both_embeddings(points, ax2, subt2, dataset, images, True,
                              x_label, y_label, zoom, frameon)

    if suptitle:
        plt.suptitle(suptitle, fontsize=36)

# ------------------------------------------------------------------------------

def _plot_both_embeddings(points, ax, subtitle, dataset, images, use_legend,
                          x_label, y_label, zoom, frameon):
    Xp = points[:, 0]
    Yp = points[:, 1]

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.set_xlim([Xp.min(), Xp.max()])
    ax.set_ylim([Yp.min(), Yp.max()])

    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)

    seaborn.reset_orig()

    if dataset and images is None:
        clrs = seaborn.color_palette('hls', n_colors=len(dataset.classes))
        LINE_STYLES = ['o', 'v', 's', '*']
        NUM_STYLES = len(LINE_STYLES)
        for i in range(len(dataset.classes)):
            indices = dataset.labels == i
            x = Xp[indices]
            y = Yp[indices]
            label = dataset.labelEncoder.inverse_transform([i])[0]
            marker = LINE_STYLES[i % NUM_STYLES]
            ax.scatter(x, y, c=clrs[i], label=label, marker=marker, zorder=10)
        if use_legend:
            plt.legend(loc=1, prop={'size': 12})
            # pass
    elif images is not None:
        imscatter(points, images, ax, zoom, frameon)
    else:
        clrs = seaborn.color_palette('hls', n_colors=len(points))
        for i in range(len(points)):
            x = Xp[i]
            y = Yp[i]
            ax.scatter(x, y, c=clrs[i])
            ax.annotate(str(i), xy=(x, y))

    if subtitle:
        ax.set_title(subtitle, fontsize=24)

# ------------------------------------------------------------------------------

def _get_labels(compressor):
    try:
        var_expl = compressor.explained_variance_ratio_
        x_label  = 'PCA1: %s%% variance' % round(100 * var_expl[0], 2)
        y_label  = 'PCA2: %s%% variance' % round(100 * var_expl[1], 2)
        return x_label, y_label
    except AttributeError:
        return '', ''

# ------------------------------------------------------------------------------

def plot_learning_curve(directory):
    """Plot figures based on experimenetal directory.
    """
    fpath = _full_path(directory)
    logfile = os.path.join(fpath, 'out.txt')
    res = _parse_logfile(logfile)

    try:
        x, y1 = res
        plt.plot(x, y1,     color='r')
    except:
        pass

    try:
        x, y1, y2 = res
        plt.plot(x, y1,     color='r')
        if (np.array(y2) != 0).all():
            plt.plot(x, y2, '', color='b')
    except:
        pass

    try:
        x, y1, y2, y3 = res

        plt.plot(x, y1,     color='r')
        if (np.array(y2) != 0).all():
            plt.plot(x, y2, '', color='b')
        if (np.array(y3) != 0).all():
            plt.plot(x, y3, '', color='g')
    except:
        pass

    plt.xlabel('Batch')
    plt.ylabel('Train error')

    plt.show()

# ------------------------------------------------------------------------------

def next_color():
    """Return new, unique color to use in plot.
    """
    global C_IDX
    if len(COLORS) == C_IDX:
        C_IDX = 0
    c = COLORS[C_IDX]
    C_IDX += 1
    return c

# ------------------------------------------------------------------------------

def _full_path(subdir):
    """Return full path to output files based on experimental subdirectory.
    """
    if subdir[0] == '/':
        subdir = subdir[1:]
    BASE = '/Users/gwg/local/dmcm/experiments/'
    return os.path.join(BASE, subdir)

# ------------------------------------------------------------------------------

def _parse_logfile(fpath):
    """Parse logfile from DMCM experiments.
    """
    with open(fpath) as f:
        reader = csv.reader(f, delimiter='\t')
        lines = [line for line in reader]

    idx = 0
    for i, line in enumerate(lines):
        if line[0] == 'Training model.':
            # i  : The line with 'Training model.'
            # i+1: The line with '===...'
            # i+2: The line with the desired output.
            idx = i + 2
            break

    data = lines[idx:]

    if 'slurmstepd: error:' in data[-1][0]:
        data = data[:-1]

    xs  = []
    y1s = []
    try:
        for i, (x, y1) in enumerate(data):
            xs.append(int(x))
            y1s.append(float(y1))
        return xs, y1s
    except:
        if '=' in data[i][0]:
            return xs, y1s

    y2s = []
    try:
        for i, (x, y1, y2) in enumerate(data):
            xs.append(int(x))
            y1s.append(float(y1))
            y2s.append(float(y2))
        return xs, y1s, y2s
    except:
        if '=' in data[i][0]:
            return xs, y1s, y2s

    y3s = []
    try:
        for i, (x, y1, y2, y3) in enumerate(data):
            xs.append(int(x))
            y1s.append(float(y1))
            y2s.append(float(y2))
            y3s.append(float(y3))
        return xs, y1s, y2s
    except:
        if '=' in data[i][0]:
            return xs, y1s, y2s, y3s

# ------------------------------------------------------------------------------

def parse_vae_file(fname):

    with open(fname) as f:
        reader = csv.reader(f, delimiter='\t')
        lines = [line for line in reader]

    idx = 0
    for i, line in enumerate(lines):
        if line[0] == 'Training model.':
            idx = i + 2
            break

    data = lines[idx:]
    new_data = []
    try:
        for i, (epoch, rec_tr, kld_tr, rec_te, kld_te) in enumerate(data):
            line = [int(epoch), float(rec_tr), float(kld_tr), float(rec_te),
                    float(kld_te)]
            new_data.append(line)
        return new_data
    except:
        return new_data

# ------------------------------------------------------------------------------

def parse_jmvae_file(fname):
    with open(fname) as f:
        reader = csv.reader(f, delimiter='\t')
        lines = [line for line in reader]

    idx = 0
    for i, line in enumerate(lines):
        if line[0] == 'Training model.':
            idx = i + 2
            break

    data = lines[idx:]
    new_data = []
    try:
        for i, line in enumerate(data):
            epoch, rec_tr1, rec_tr2, kld_tr, rec_te1, rec_te2, kld_te = line
            line = [int(epoch),
                    float(rec_tr1), float(rec_tr2), float(kld_tr),
                    float(rec_te1), float(rec_te2), float(kld_te)]
            new_data.append(line)
        return new_data
    except:
        return new_data

# ------------------------------------------------------------------------------

def plot_gmm(X, gmm):
    Y = gmm.predict(X)
    _plot_gmm(X, Y, gmm.means_, gmm.covariances_)

# ------------------------------------------------------------------------------

def _plot_gmm(X, Y_, means, covariances):

    fig, ax = plt.subplots(1, 1, dpi=72)
    fig.set_size_inches(20, 10)

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    colors = seaborn.color_palette('hls', n_colors=len(means))

    for i, (mean, covar) in enumerate(zip(means, covariances)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # As the DP will not use every component it has access to unless it
        # needs it, we shouldn't plot the redundant components.
        if not np.any(Y_ == i):
            continue
        if u[0] == 0:
            continue

        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], color=colors[i])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = Ellipse(mean, v[0], v[1], 180. + angle, color=colors[i])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

# ------------------------------------------------------------------------------

def corrmat(X, compute_corrmat=True, fname=None):
    """Save correlation matrix for a batch of samples.
    """
    if compute_corrmat:
        X = np.corrcoef(X)
    plt.imshow(X)
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
    plt.clf()

# ------------------------------------------------------------------------------

def hinton(matrix, fname=None):
    """Draw Hinton diagram for visualizing a weight matrix.
    """
    plt.figure(figsize=(20, 100))
    ax = plt.gca()
    max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    is_square = len(matrix.shape) > 1 and matrix.shape[0] == matrix.shape[1]
    if is_square:
        ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # ax.autoscale_view()
    # ax.invert_yaxis()

    if fname:
        plt.savefig(fname)
    else:
        plt.show()
    plt.clf()