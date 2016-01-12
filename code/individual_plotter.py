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
# cd C:\Users\carls\Google Drive\PythonStocks\src\root\nested
# prepro = PreProcessor()
# prepro.write_pre_labels()

# df = modeler.df
file_name = "Kaskade - 4 AM (Adam K & Soha remix).wav"
modeler = WaveRead()
sample_rate, recording = modeler.get_recording(file_name)
song_position = 1
modeler.song_begin = float(song_position * 30)
modeler.song_end = float(modeler.song_begin + 30)
segment_length, sample_rate, segments, recording = modeler.get_segments_temporally(file_name, sample_rate, recording)
#song_data, column_labels = modeler.convert_one(file_name, segment_length, sample_rate, segments, recording)
all_spectra,frequencies = modeler.get_full_fourier_averages(file_name)



