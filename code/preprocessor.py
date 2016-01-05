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


class PreProcessor:

    def write_pre_labels_to_csv(self, song_names, input_dir):
        csv_dir = input_dir
        with open(csv_dir + "\\" + "pre_labels" + '.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["Song", "Labels"])
            for row in song_names:
                writer.writerow([row])

    def write_pre_labels(self):
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
        df = pd.read_csv(Globals.getLabelsDir()+"\\"+"labels.csv")
        return dict(df.values)    
            
        