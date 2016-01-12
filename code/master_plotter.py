import numpy as np
from statsmodels.tsa.stattools import acf
import itertools
import matplotlib.pyplot as plt
import wave, struct
import numpy as np
import pylab
from scipy.fftpack import fft
from scipy.io import wavfile  # get the api
from sklearn.cluster import KMeans
import pandas as pd
from Globals import Globals
from os import listdir
from os.path import isfile, join
from collections import Counter
from wave_read_advanced import WaveReadAdvanced as WaveRead
from preprocessor import PreProcessor
from Modeler import Modeler as Modeler
import sys
from sklearn.metrics.pairwise import pairwise_distances
from pyexpat import features
from sklearn.metrics import confusion_matrix
from sklearn.decomposition.nmf import NMF
from bokeh.embed import components
from _sqlite3 import Row
import matplotlib.patches as mpatches
from matplotlib import colors
from sklearn.preprocessing import normalize, scale
# cd C:\Users\carls\Google Drive\PythonStocks\src\root\nested
# prepro = PreProcessor()
# prepro.write_pre_labels()
def get_majors(components, labels):
    foo = zip(components, labels)
    struct_array = np.array([np.array([element[0], element[1]]) for element in foo])
    outputs = []
    output_names = []
    for name in set(struct_array[:, 1]):
        mask = struct_array[:, 1] == name
        outputs.append(Counter(struct_array[mask][:, 0]).most_common())
        output_names.append(name)
    return outputs, output_names
    
def plot_data(X, index_mask, frequency_bins, plot_label, songs, labels):
    plt.figure()
    plt.axes([.125, .175, .75, .75])
    plt.ylabel(plot_label, fontsize=20)
    plt.xlabel("Frequency (hz)", fontsize=20)
    x = np.log(frequency_bins)
    print labels
    print labels.shape
    print X.shape
    for lab in set(labels):
        label_mask = labels == lab
        print index_mask
        X_d = X.copy()
        X_d = X_d.T[index_mask].T[label_mask]
        plt.plot(x, np.mean(X_d, axis=0), linewidth=5, label = lab)
    plt.legend()
    modded_bins = (np.array(frequency_bins)).astype(int)
    plt.xticks(x, modded_bins, rotation='vertical')
    plt.xlim([np.min(x), np.max(x)])
    
    
def plot_supervised_confusion_matrix(actuals, predicteds):
    n_labels = len(set(list(actuals[0])))
    master_conf = np.zeros((n_labels, n_labels))
    col_labels = ['hip', 'roc', 'pop', 'cla', 'tec']
    row_labels = ['hip', 'roc', 'pop', 'cla', 'tec']
    for i in xrange(actuals.shape[0]):
        actual = actuals[i]
        predicted = predicteds[i]
        conf_matrix = confusion_matrix(predicted, actual, labels=col_labels)
        master_conf += conf_matrix
    table_vals = master_conf
    plt.figure()
    cellcolours = np.empty_like(table_vals, dtype='object')
    cellcolours[:, 1] = 'Blues'
    input_map = [str(col) for col in np.linspace(.1, 1.0, 50).tolist()]
    cmap1 = colors.ListedColormap(input_map[::-1])
    bounds = [i * np.round(np.max(table_vals) / 50) for i in xrange(50)]
    norm = colors.BoundaryNorm(bounds, cmap1.N)
    foo = plt.table(cellText=table_vals, rowLabels=row_labels,
                    colLabels=col_labels,
                    loc='center', bbox=[0, 0, 1, 1], cellColours=cmap1(norm(table_vals)))
    foo.set_fontsize(20)
    plt.xticks([])
    plt.yticks([])
    
def plot_feature_importance(importances, all_masks, all_labels, ind_labels):#, major_indeces, major_names, all_names):
    plt.figure()
    plt.axes([.125, .22, .75, .75])
    agg_importances = []
    agg_mins = []
    agg_maxs = []
    ccount = 0
    agg_mask = np.zeros(all_masks[0].shape[0])
    for mask in all_masks:
        ccount += 1
        print ccount
        print mask.shape
        print importances.shape
        print 'asdfffffff'
        agg_mask += mask
        agg_importances.append(np.mean(importances[mask[:-2]]))
        agg_mins.append(np.min(importances[mask[:-2]]))
        agg_maxs.append(np.max(importances[mask[:-2]]))
        stds = [np.array(agg_importances) - np.array(agg_mins), np.array(agg_maxs) - np.array(agg_importances)]
    plt.bar(np.arange(len(all_masks)),agg_importances, yerr=stds, ecolor = 'k', capsize = 10)
    print all_labels
    plt.xticks(np.arange(0,len(all_masks)) + .4, all_labels, rotation=45,  fontsize = 14)
    plt.ylabel('Feature Importance', fontsize = 20)
    plt.figure()
    plt.axes([.125, .26, .75, .71])
    importances = np.array(importances)
    print agg_mask.shape
    agg_mask = np.array([bool(num) for num in agg_mask.tolist()])
    print importances[agg_mask].shape[0]
    print "BLAARGGGGGGG"
    print importances[agg_mask].shape[0]
    print ind_labels.shape[0]
    print agg_mask.shape
    sub_importances = importances[agg_mask]
    plt.bar(np.arange(sub_importances.shape[0]), sub_importances)
    plt.xticks(np.arange(0,ind_labels.shape[0]) + .5, ind_labels, rotation='vertical',  fontsize =6)
    plt.ylabel('Feature Importance', fontsize = 20)
    print ind_labels[np.argsort(sub_importances)[::-1]][0:8]
   
    
    
def make_scatter_plots():
    pass
    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    # USE PANDAS PLOTTING HERE
#     ax1.plot(x, y)
#     ax1.set_title('Sharing x per column, y per row')
#     ax2.scatter(x, y)
#     ax3.scatter(x, 2 * y ** 2 - 1, color='r')
#     ax4.plot(x, 2 * y ** 2 - 1, color='r')

def make_cluster_purity_plots(components, labels):
    cluster_groupings, assoc_names = get_majors(components, labels)
    group_associations = []
    data = np.zeros(5)
    color_dict = {'0':'r', '1':'b', '2':'g', '3':'y', '4':'c'}
    all_colors = np.array(['r', 'r', 'r', 'r', 'r'])
    for grouping in cluster_groupings:
        # group_associations.append((name, grouping[0][0]))
        current_info = np.zeros(5)
        ccount = 0
        print 
        colors = np.array(['r', 'r', 'r', 'r', 'r'])
        for tup in grouping:                
            current_info[ccount] = tup[1]
            colors[ccount] = color_dict[str(tup[0])]
            ccount += 1
        data = np.vstack((data, current_info))  
        all_colors = np.vstack((all_colors, colors))     
    cell_text = []
    n_rows = 5
    n_columns = n_rows
    index = np.arange(n_columns)
    bar_width = 0.4
    y_offset = np.array([0.0] * n_columns)
    patches = []
    for row in range(0, n_rows):
        plt.bar(index, data[1:, row], bar_width, bottom=y_offset, color=all_colors[1:, row])
        y_offset = y_offset + data[1:, row]
        patches.append(mpatches.Patch(color=all_colors[row + 1, 0], label=assoc_names[row] + ' cluster'))
        # cell_text.append(['%1.1f' % (x/1000.0) for x in y_offset])
    purities = data[1:, 0] / np.sum(data[1:, :], axis=1)
    assoc_names = [assoc_names[i] + ", p=" + str(np.round(purities[i] * 1000) / 10) for i in xrange(len(purities))]
    plt.legend(handles=patches)
    plt.xticks(np.arange(n_rows) + .2, assoc_names, fontsize=16, rotation=20)
    plt.xlabel('Genres', fontsize=18)
    plt.ylabel('Contents', fontsize=18)
    plt.ylim([0, 130])
    
def get_prec_rec_f1(all_reports, classes):
    # cla,0.86,0.81,0.83,31
    precisions = np.zeros(5)
    recalls = np.zeros(5)
    f1s = np.zeros(5)
    supports = np.zeros(5)
    av_count = 0.
    for report in all_reports:
        modded = report.replace("       ", ",").replace("      ", ",").replace(" ", "").replace("\n", "").replace("avg/total", "")
        av_count += 1
        for i in xrange(len(classes)):
            scores = [float(nms) for nms in modded.split(classes[i])[1].split(',')[1:4]]
            # precision    recall  f1-score   suppor
            counts = float(modded.split(classes[i])[1].split(',')[4])
            precisions[i] += scores[0] * counts
            recalls[i] += scores[1] * counts
            f1s[i] += scores[2] * counts
            supports[i] += counts
    
    return np.array(precisions) / np.array(supports), np.array(recalls) / np.array(supports), np.array(f1s) / np.array(supports)  
            
            
    
def make_acc_prec_rec(all_reports, classes):
    precision, recall, f1 = get_prec_rec_f1(all_reports, classes)
    fig, ax = plt.subplots()
    n_groups = 5
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, precision, bar_width, alpha=opacity, color='b', label='Precision')
    rects2 = plt.bar(index + bar_width, recall, bar_width, alpha=opacity, color='r', label='Recall')
    rects3 = plt.bar(index + bar_width * 2, f1, bar_width, alpha=opacity, color='g', label='F1')
    plt.xlabel('Genre')
    plt.ylabel('Scores')
    plt.title('Scores by precision, recall, and F1')
    plt.xticks(index + bar_width, classes)
    plt.legend()

all_reports = np.load('all_reports.npy')    
components = np.load('components.npy')  # ## this is the saved normalized data
labels = np.load('labels2.npy')
all_tests = np.load('all_tests.npy')
all_predicts = np.load('all_predicts.npy')
X_unscaled = np.load('X_unscaled.npy')
all_importances = np.load('all_importances.npy')
all_importances = np.sum(all_importances, axis = 0) / all_importances.shape[0]
#X_unscaled  = normalize(X_unscaled, axis = 0)
make_cluster_purity_plots(components, labels)
make_acc_prec_rec(all_reports, list(set(labels)))
plot_supervised_confusion_matrix(all_tests, all_predicts)
modeler = Modeler()
df = modeler.df
# components, song_names, labels, components2 = modeler.run_all_models()
all_masks = modeler.extract_frequencies_and_indeces()
all_labels = ['Average Power', 'Beat Strength', '<Beat sep>', 'Med<Beat sep>', 'Std<Beat sep>', 'zcr data', 'Maxs norm']
plot_feature_importance(all_importances, all_masks[:-2], all_labels, all_masks[-1])
good_mask = np.load('good_mask.npy')
df = df[good_mask]
df_trunc = df[['Beat strength in 6426.17 hz range', 'Mean _maxs', 'med(Beat sep) (seconds) in 100.0 hz range']]
color_dict = dict()
color_dict['tec'] = 'b'
color_dict['hip'] = 'r'
color_dict['cla'] = 'g'
color_dict['roc'] = 'k'
color_dict['pop'] = 'c'

INVERT PURITIES HERE SO THAT WHEN YOU HAVE MORE DATA YOU WILL HAVE GROUP 0, group 1, and purities of classes

color_set = np.array([color_dict[name] for name in df['Label']])
pd.scatter_matrix(df_trunc, color = color_set)
#plot_data(scale(X_unscaled, axis = 0), power_column_mask[:-2], frequency_bins, "Song-averaged power", [0, 1], labels)
#plot_data(scale(X_unscaled, axis = 0), beat_column_mask[:-2], frequency_bins, "Beat strength", [0, 1], labels)
#plot_data(1./(X_unscaled * .5 + .0000000001), bpm_column_mask[:-2], frequency_bins, "Average beat rate", [0, 1], labels)