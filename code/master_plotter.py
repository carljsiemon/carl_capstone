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

'''
The purpose of this code is to load saved feature data 
and .npy outputs from the models and then
visualize the results.
OUTPUTS are none if OUTPUTS are not specified in function documentation
'''

def get_majors(components, labels):
    '''
    Function purpose is to get the genre composition of each cluster
    RETURNS: outputs = the genre composition of each cluster as a list
    INPUTS: components = list of cluster names
    for each song; labels = list of corresponding genre label for each song
    '''
    print components
    print labels
    foo = zip(components, labels)
    struct_array = np.array([np.array([element[1], element[0]]) for element in foo])
    outputs = []
    output_names = []
    print struct_array
    for name in set(struct_array[:, 1]):
        mask = struct_array[:, 1] == name
        outputs.append(Counter(struct_array[mask][:, 0]).most_common())
        output_names.append(name)
    '''
    Outputs is list where each element is a 
    dictionary containing the sorted compositions
    (by genre) of each cluster
    '''
    return outputs

def plot_feature_vs_frequency_data(X,\
        index_mask, frequency_bins, plot_label, song_labels):
    '''
    Plot aggregated (averaged) feature data for each 
    genre for a given type of feature (e.g. beat separation)
    INPUTS: X = 2d numpy array feature matrix; index_mask = 1d Boolean numpy array
    with column locations of feature of interest; plot_label = plot label
    as string; song_labels = list of strings (genre labels for each song)
    '''
    plt.figure()
    plt.axes([.125, .175, .75, .75])
    plt.ylabel(plot_label, fontsize=20)
    plt.xlabel("Frequency (hz)", fontsize=20)
    x = np.log(frequency_bins)
    '''plot aggregate curves for each label here'''
    for lab in set(song_labels):
        label_mask = song_labels == lab
        X_d = X.copy()
        X_d = X_d.T[index_mask].T[label_mask]
        plt.plot(x, np.mean(X_d, axis=0), linewidth=5, label = lab)
    plt.legend()
    modded_bins = (np.array(frequency_bins)).astype(int)
    plt.xticks(x, modded_bins, rotation='vertical')
    plt.xlim([np.min(x), np.max(x)])
    
def make_feature_vs_frequency_plots(all_masks, mask_labels, X, frequency_bins, labels):
    '''
    This makes the aggregate feature vs. frequency plots 
    for each type of feature.  
    INPUTS: all_masks= list of 1d boolean numpy arrays
    containing all masks that point to each feature type;
    mask_labels= list of strings containing all feature labels;
    X = 2d numpy float array of Feature data; 
    frequency bins = 1d float numpy array of frequency data;
    labels = list of strings, each string is a genre  
    '''
    for i in xrange(len(mask_labels)-2):
        plot_feature_vs_frequency_data(X, all_masks[i][:-2],\
            frequency_bins, mask_labels[i], labels) 
   
def plot_supervised_confusion_matrix(actuals, predicteds):
    '''
    This visualizes the confusion matrix for supervised learning.
    INPUTS: actuals/predicteds= list of 1d numpy arrays containing actual/predicted
    outputs from each KFold test;
    RETURNS: master_conf = aggregated confusion matrix (from all kfolds) 
    as 2d numpy int array 
    '''
    n_labels = len(set(list(actuals[0])))
    master_conf = np.zeros((n_labels, n_labels))
    col_labels = ['hip', 'roc', 'pop', 'cla', 'tec']
    row_labels = ['hip', 'roc', 'pop', 'cla', 'tec']
    '''
    We need to aggregate the confusion 
    matrix results from each of the KFolds first
    '''
    for i in xrange(actuals.shape[0]):
        actual = actuals[i]
        predicted = predicteds[i]
        conf_matrix = confusion_matrix(predicted, actual, labels=col_labels)
        master_conf += conf_matrix
    
    '''Visualize our aggregate confusion matrix (master_conf)'''
    table_vals = master_conf
    plt.figure()
    cellcolours = np.empty_like(table_vals, dtype='object')
    cellcolours[:, 1] = 'Blues'
    '''Use greyscale coloring to visualize amount in each cell'''
    input_map = [str(col) for col in np.linspace(.7, 1.0, 100).tolist()]
    cmap1 = colors.ListedColormap(input_map[::-1])
    bounds = [i * np.round(np.max(table_vals) / 100) for i in xrange(100)]
    norm = colors.BoundaryNorm(bounds, cmap1.N)
    foo = plt.table(cellText=table_vals, rowLabels=row_labels,\
                    colLabels=col_labels,\
                    loc='center', bbox=[0, 0, 1, 1],\
                    cellColours=cmap1(norm(table_vals)))
    foo.set_fontsize(20)
    plt.xticks([])
    plt.yticks([])
    return master_conf 

def plot_feature_importance(importances, all_masks, agg_labels, ind_labels):
    '''
    This plots both feature importances and averaged feature importances for each
    type of feature.
    INPUTS: importances= 1d numpy array containing floats of feature importances; 
    all_masks= list of 1d boolean numpy arrays pointing to each type of feature;
    agg_labels= list of strings for each feature group type; ind_labels= list of
    strings labeling all features
    RETURNS: top 8 most important features as 1d numpy string array from ind_labels
    '''
    plt.figure()
    plt.axes([.125, .22, .75, .75])
    agg_importances = []
    agg_mins = []
    agg_maxs = []
    ccount = 0
    agg_mask = np.zeros(all_masks[0].shape[0])
    '''
    This gets the data for the averaged feature importances, 
    including min and max for each feature type
    '''
    for mask in all_masks:
        ccount += 1
        agg_mask += mask
        agg_importances.append(np.mean(importances[mask[:-2]]))
        agg_mins.append(np.min(importances[mask[:-2]]))
        agg_maxs.append(np.max(importances[mask[:-2]]))
        stds = [np.array(agg_importances)\
        - np.array(agg_mins), np.array(agg_maxs) - np.array(agg_importances)]
    '''plot the aggregate feature importances'''
    plt.bar(np.arange(len(all_masks)),\
    agg_importances, yerr=stds, ecolor = 'k', capsize = 10)
    plt.xticks(np.arange(0,len(all_masks)) + .4, agg_labels, rotation=45,  fontsize = 14)
    plt.ylabel('Feature Importance', fontsize = 20)
    
    '''plot all the individual features'''
    plt.figure()
    plt.axes([.125, .26, .75, .71])
    importances = np.array(importances)
    agg_mask = np.array([bool(num) for num in agg_mask.tolist()])
    sub_importances = importances[agg_mask]
    plt.bar(np.arange(sub_importances.shape[0]), sub_importances)
    plt.xticks(np.arange(0,ind_labels.shape[0]) + .5, ind_labels, rotation='vertical',  fontsize =6)
    plt.ylabel('Feature Importance', fontsize = 20)
    ''' return top 8 most important features here'''
    return ind_labels[np.argsort(sub_importances)[::-1]][0:8]
   
def make_scatter_plots(features_of_interest, df):
    '''
    This function makes bivariate scatter matrix plot for the
    inputed features of interest, which are typically the 
    individual features of the greatest importance in our 
    supervised learning classification model
    INPUTS: features_of_interest = list of strings; df =  pandas
    data frame containing song feature data
    '''
    plt.figure
    '''get mask containing songs used in our model'''
    good_mask = np.load('good_mask.npy')
    df = df[good_mask]
    contains_outliers = 'B- Var(c.t.)'
    '''
    remove outliers in the 'B- Var(c.t.)' feature to better see plots
    '''
    df = df[np.abs(df[contains_outliers]\
    - df[contains_outliers].mean()) / df[contains_outliers].std() <= 2.3 ]
    df_trunc = df[features_of_interest]
    color_dict = dict()
    '''label data points by color'''
    color_dict['tec'] = 'b'
    color_dict['hip'] = 'r'
    color_dict['cla'] = 'g'
    color_dict['roc'] = 'k'
    color_dict['pop'] = 'c'
    color_set = np.array([color_dict[name] for name in df['Label']])
    ax = pd.scatter_matrix(df_trunc, color = color_set)
    plt.xlabel([])
    plt.ylabel([])
    
def make_cluster_purity_plots(components, labels):
    '''
    This function gets the total amount of each
    genre contained within each cluster and the makes 
    purity plots.  Should try to shorten this function
    in future
    INPUTS: components = list of cluster names
    for each song; labels = list of corresponding label for each song
    '''
    cluster_groupings = get_majors(components, labels)
    data = []
    color_dict = {'cla':'r', 'pop':'b', 'roc':'g', 'hip':'y', 'tec':'c'}
    all_colors = []
    majorities = []
    ass_keys = []
    
    '''
    loop through each cluster and find what is majority genre 
    in the cluster, also keep track of non-majority composition
    '''
    for grouping in cluster_groupings:
        current_info = np.zeros(len(grouping))
        ccount = 0
        colors = ["" for x in xrange(len(grouping))]
        for tup in grouping:           
            current_info[ccount] = tup[1]
            colors[ccount] = color_dict[str(tup[0])]
            ccount += 1
        data.append(current_info)  
        all_colors.append(colors)
        majorities.append(colors[0])
        ass_keys.append(grouping[0][0])
    
    ass_keys = np.array(ass_keys)
    ccount = 0
    y_ticks = []
    '''
    compute purity percentage of each cluster, that is,
    divide plurality leader by total count in entire cluster
    '''
    purity = np.array([row[0] / float(np.sum(row)) for row in data])
    '''compute non_purity = 1 - non_purity'''
    non_purity = np.array([1. - row[0] / float(np.sum(row)) for row in data])
    
    '''
    plot purity then non-purity on top of it, 
    giving a nice stacked bar plot
    '''
    p1 = plt.barh(np.arange(5),purity, color = np.array(majorities))
    p2 = plt.barh(np.arange(5), non_purity, left = purity, color = 'grey')
    '''print purity percentage of each cluster'''
    [plt.text(.01,row + .3, 'purity='+str(trunc(purity[row])) +'%', \
            color = 'w', weight= 'bold', fontsize=16) for row in xrange(5)]
    
    '''make this legend more elegant and more general in the future'''
    plt.legend((p1[0],p1[1],p1[2],p1[3],p1[4], p2[0]),\
    (ass_keys[0],ass_keys[1], ass_keys[2],ass_keys[3],ass_keys[4], 'non-majority'))
    plt.yticks(y_ticks, ['0','1','2','3','4','5'],\
            fontsize=16, rotation='horizontal')
    plt.xlim(0,1.5)
    foo = np.arange(5)
    vals = np.array([str(elm + 1) for elm in foo])
    plt.yticks(np.arange(5)+.4, vals, fontsize = 20)
    plt.xticks(fontsize = 16)
    plt.ylabel('Clusters', fontsize=22)
    plt.xlabel('Composition', fontsize=22)
    
def get_tf(conf_matrix):
    '''
    This function gets true positive, true negative, etc. data from confusion matrix.
    RETURNS: list of true negatives, true positives, etc. for each
    genre class 
    INPUTS: conf_matrix as 2d numpy int array
    '''
    total = np.sum(conf_matrix)
    tps, tns, fps, fns = [],[],[],[]
    for j in xrange(conf_matrix.shape[0]):
        tps.append(conf_matrix[j,j])
        fns.append(np.sum(conf_matrix[j]) - conf_matrix[j,j])
        fps.append(np.sum(conf_matrix[:,j]) - conf_matrix[j,j])
        index = len(tps) -1
        tns.append(total - tps[index] - fns[index] - fps[index])
    return tps, tns, fps, fns

def get_metric(kernel, tps, tns, fps, fns, conf_matrix):
    '''
    This function gets one scoring metric (depending on the 
    inputted kernel) for each of the classes
    RETURNS: worker = computed scoring metric for each genre
    as 1d numpy array.  Also returns a singleton containing 
    the average value of the score across all genres
    INPUTS: kernel = function to compute type of scoring metric (e.g. recall);
    tps, tns, fps, fns = list of true positives, true negatives, etc for
    each genre class; conf_matrix = confusion matrix as 2d int numpy array
    '''
    worker = []
    avg_sum = 0.
    for i in xrange(len(tps)):
        worker.append(kernel(tps[i], tns[i], fps[i], fns[i]))
        avg_sum += worker[len(worker) - 1] * np.sum(conf_matrix[i])
    worker = np.array(worker)
    return worker, avg_sum / float(np.sum(conf_matrix))

def recall(tp, tn, fp, fn):
    return float(tp) / float(tp + fn)
def precision(tp, tn, fp, fn):
    return float(tp) / float(tp + fp)
def f1(tp, tn, fp, fn):
    return float(2. * tp) / float(2. * tp + fp + fn)
def accc(tp, tn, fp, fn):
    return float(tp + tn) / float(tp + tn + fp + fn)    
    
        
def get_prec_rec_f1_acc_from_conf_mat(conf_mat):
    '''
    RETURNS recall, f1, accuracy, and precision
    from the confusion matrix for each of the genres
    as lists.  Also it returns the
    average-across-genres for each of these metrics as
    floats
    INPUTS: conf_mat as 2d numpy array
    '''
    tps, tns, fps, fns = get_tf(conf_mat)
    recalls, avg_recall = get_metric(recall, tps, tns, fps, fns, conf_mat)
    precisions, avg_precision = get_metric(precision, tps, tns, fps, fns, conf_mat)
    f1s, avg_f1 = get_metric(f1, tps, tns, fps, fns, conf_mat)
    '''
    We use the more classical accuracy of right/(wrong+right)
    here otherwise we can get very high accuracies (since this is
    a multi-class problem) that are misleading
    '''
    avg_acc = float(np.trace(conf_mat)) / float(np.sum(conf_mat))
    '''Scoring metrics by class along with the average-across-genres scores'''
    return recalls, precisions, f1s, avg_recall, avg_precision, avg_f1, avg_acc

def trunc(input):
    return np.round(input * 1000) / 10

def make_acc_prec_rec_plots(classes, conf_mat):
    '''
    This function plots accuracy, precision, recall, and 
    F1 score for each genre as bar plots.
    INPUTS: classes = 1d string numpy array of class names; conf_mat = confusion
    matrix as 2d numpy int array
    '''
    recall, precision, f1,\
    avg_recall, avg_precision,avg_f1, acc_av = get_prec_rec_f1_acc_from_conf_mat(conf_mat)
    fig, ax = plt.subplots()
    n_groups = 5
    index = np.arange(n_groups)
    bar_width = 0.23
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, precision, bar_width, alpha=opacity,\
                    color='b', label='Precision')
    rects2 = plt.bar(index + bar_width, recall, bar_width,\
                    alpha=opacity, color='r', label='Recall')
    rects3 = plt.bar(index + bar_width * 2, f1, bar_width,\
                    alpha=opacity, color='g', label='F1')
    plt.text(.1,1.03,'Recall=' + str(trunc(avg_recall))\
             + ', Precision=' + str(trunc(avg_precision))\
             + '\n F1=' + str(trunc(avg_f1))+ ', Accuracy='\
             + str(trunc(acc_av)),size = 18)
    plt.xlabel('Genre', fontsize = 20)
    plt.ylabel('Scores', fontsize = 20)
    plt.title('Supervised learning: GBC scores by precision,\
    recall, F1, and accuracy', fontsize = 20)
    plt.xticks(index + bar_width + .1, classes, fontsize = 16)
    plt.ylim([0, 1.2])
    plt.legend()


'''*********** THE CODE BELOW ACTUALLY GENERATES THE PLOTS*************'''
'''Load appropriate data from npy's'''
all_reports = np.load('all_reports.npy')    
components = np.load('components.npy')
labels = np.load('labels2.npy')
all_tests = np.load('all_tests.npy')
all_predicts = np.load('all_predicts.npy')
all_importances = np.load('all_importances.npy')
X_unscaled = np.load('X_unscaled.npy')
X_unscaled  = normalize(X_unscaled, axis = 0)    
'''aggregate the importances from each of the KFolds'''
all_importances = np.sum(all_importances, axis = 0) / all_importances.shape[0]
'''
It is convenient to work with
the pandas data frame for some plots
'''
modeler = Modeler()
df = modeler.df
'''
Get feature mask data and labels so we know what to plot
'''
all_masks = modeler.extract_frequencies_and_indeces()
mask_labels = ['Energy by frequency', 'Beat Strength',\
'$<Beat\ \ Separation>$', '$Med(Beat\ \ Separation)$',\
'$Std(Beat\ \ Separation)$', 'ZCR data', 'Total energy']
frequency_bins = all_masks[-2]

'''Start plotting'''
make_cluster_purity_plots(components, labels)
conf_mat = plot_supervised_confusion_matrix(all_tests, all_predicts)
make_acc_prec_rec_plots(list(set(labels)), conf_mat)
features_of_interest =\
plot_feature_importance(all_importances, all_masks[:-2], mask_labels, all_masks[-1])
'''We plot the top 8 features of interest in our scatter plots'''
make_scatter_plots(features_of_interest, df)
make_feature_vs_frequency_plots(all_masks, mask_labels, X_unscaled, frequency_bins, labels)