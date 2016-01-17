import numpy as np
from statsmodels.tsa.stattools import acf
import itertools
import matplotlib.pyplot as plt
import wave, struct
import numpy as np
import pylab
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
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

'''
As the name suggests, this is the main work horse 
where I conduct both file conversion and modeling
'''

def print_majors(components, labels):
    '''This prints the composition of each song'''
    '''
    INPUTS: components = list of cluster names (strings)
    for each song; labels = list of corresponding genre labels (strings) for each song
    '''    
    foo = zip(components, labels)
    struct_array = np.array([np.array([element[0], element[1]]) for element in foo])
    for name in set(struct_array[:,1]):
        mask = struct_array[:,1] == name
        print name
        print Counter(struct_array[mask][:,0])
    
'''
conduct feature generation using waveread, 
this takes several hours    
'''
file_convert = False    
if file_convert:        
    reader = WaveRead(60000, .05)
    reader.convert_all()
    
    
    
'''this creates the supervised and unsupervised models'''
conduct_model = True
if conduct_model:
    modeler = Modeler()
    '''run supervised/unsupervised models here'''
    '''
    OUTPUTS BELOW: components = 1-d numpy array of kmeans cluster labels; 
    song_names = 1-d numpy array of song names; 
        song_labels =1-d numpy array of genre labels ; 
        components2 = 1-d numpy array of kmeans/nmf cluster labels; 
        all_reports = list of confusion matrices (2-d numpy arrays) from each KFold output; 
        all_tests = list of actual labels (1-d numpy arrays) from each KFOLD test, 
        all_predicts= list of predicted labels (1-d numpy arrays) from each KFOLD test; 
        all_importances= list of feature importances for each KFOLD test
    '''
    components, song_names, song_labels,\
    components2, all_reports,\
    all_tests, all_predicts, all_importances = modeler.run_all_models()
    print "KMEANS"
    print_majors(components, song_labels)
    print "--------------"
    print "NMF"
    print_majors(components2, song_labels)
    print "--------------"
    '''save outputs for visualization'''
    np.save('all_reports.npy', all_reports)
    np.save('components.npy', components)
    np.save('components_nmf.npy', components2)
    np.save('labels2.npy', song_labels)
    np.save('all_tests.npy', all_tests)
    np.save('all_predicts.npy', all_predicts)
    np.save('song_names_post.npy', song_names)
    np.save('all_importances.npy', all_importances)
    df = modeler.df
    name_dict = dict()
    for name in song_names:
        name_dict[name] = name
    '''
    this gets the location of the rows in our original data frame that we 
    used in our analysis (in modeler we sub sample and 
    also remove outliers, so we need to keep track of the data we have used
    '''
    good_mask = np.array([True if name in name_dict else False for name in df['Song Name']])
    np.save('good_mask.npy', good_mask)