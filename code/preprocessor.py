import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, quad
import wave, struct
import numpy as np
import pylab
from scipy.fftpack import fft
from scipy.io import wavfile  # get the api
from os import listdir
from os.path import isfile, join
from Globals import Globals
import csv
import pandas as pd
from statsmodels.tsa.stattools import acf

'''
The main purpose of this is to get the song names of all the 
.wav files and then create a pre_labels.csv file containing all of the names,
making it easier to label the genre of each song
'''
class PreProcessor:
    def write_pre_labels_to_csv(self, song_names, input_dir):
        '''
        This writes all of the .wav song names into a pre_labels csv
        INPUT: song_names (string), input_dir (string)
        OUTPUT: none
        '''
        csv_dir = input_dir
        with open(csv_dir + "\\" + "pre_labels" + '.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["Song", "Labels"])
            for row in song_names:
                writer.writerow([row])

    def write_pre_labels(self):
        '''
        This loads all of the song names in the song directory
        and creates a csv called pre_labels where each song will be 
        manually labeled. the resulting file should be saved as labels.csv 
        INPUT: NONE
        OUTPUT: NONE
        '''
        song_dir = Globals.getSongDir()
        file_list = [f for f in listdir(song_dir) if isfile(join(song_dir, f))]
        read_count = 0
        song_names = []
        for file_name in file_list:
            if ".wav" not in file_name:
                break
            read_count += 1
            song_names.append(file_name)
        self.write_pre_labels_to_csv(song_names, Globals.getLabelsDir())
        
    def getSongLabels(self):
        '''
        This is used externally to get all of 
        the song labels after the songs have been labeled.  It returns 
        the dictionary of song labels
        INPUT: NONE
        OUTPUT: dictionary, keys = song names, values = song labels
        '''
        df = pd.read_csv(Globals.getLabelsDir()+"\\"+"labels.csv")
        return dict(df.values)    
            
        