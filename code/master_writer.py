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
#cd C:\Users\carls\Google Drive\PythonStocks\src\root\nested
#prepro = PreProcessor()
#prepro.write_pre_labels()
def print_majors(components, labels):
    foo = zip(components, labels)
    struct_array = np.array([np.array([element[0], element[1]]) for element in foo])
    for name in set(struct_array[:,1]):
        mask = struct_array[:,1] == name
        print name
        print Counter(struct_array[mask][:,0])

def plot_data(df, index_mask, frequency_bins, plot_label, songs):
    plt.axes([.125, .175, .75, .75])
    plt.ylabel(plot_label, fontsize = 20)
    plt.xlabel("Frequency (hz)", fontsize = 20)
    x = np.log(frequency_bins)
    for song in songs:
        plt.plot(x, df.values[song,:][index_mask], linewidth=5)
    modded_bins = (np.array(frequency_bins)).astype(int)
    plt.xticks(x,modded_bins, rotation = 'vertical')
    plt.xlim([np.min(x),np.max(x)])
    
reader = WaveRead(60000, .05)
#reader.convert_all()
modeler = Modeler()
#nmf_matrix, components, song_names = modeler.run_all_models()
components, song_names, labels, components2 = modeler.run_all_models()
# blah = pairwise_distances(nmf_matrix, metric='cosine')
# print song_names[np.argsort(blah[100])[0:5]]
print "KMEANS"
print_majors(components, labels)
print "--------------"
print "NMF"
print_majors(components2, labels)
print "--------------"
# print "DBSCAN"
# print_majors(dbscan, labels)






#modeler.run_all_models()

#df = modeler.df
#beat_mask, power_mask, frequency_bins = modeler.extract_frequencies_and_indeces()
# plot_data(df, power_mask, frequency_bins, "Song-averaged power", [0,1])
# plot_data(df, beat_mask, frequency_bins, "Beat strength", [0,1])