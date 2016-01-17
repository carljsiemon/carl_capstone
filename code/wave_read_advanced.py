import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, quad
import wave, struct
import numpy as np
import pylab
from scipy.fftpack import fft
from scipy.io import wavfile
from os import listdir
from os.path import isfile, join
from Globals import Globals
import csv
import pandas as pd
from statsmodels.tsa.stattools import acf
from preprocessor import PreProcessor
import os, shutil


""" 
Main Purpose of this class: read .wav data and output time 
and frequency domain row data into a .csv for each song (one 
row per song).  The main idea is to split the song into a group of short segments and
perform Fourier Transform on these segments to extract important 
data about the composition of the song.
"""

class WaveReadAdvanced:
    def __init__(self, read_count=5000, seg_length=.05):
        '''
        INPUTS: read_count = total number of songs to be read;
        seg_length = length of song segments that song will be split into (in seconds)
        '''
        
        '''
        band_type controls how frequencies are binned, can have 
        logarithmic (nonlinear) or linear spacing
        '''
        self.band_type = "log"
        '''limit the number of files that are read'''
        self.files_read_count = read_count
        ''' total number of frequency bins'''
        self.freq_bin_total = 15          
        '''
        song is splitted into chunks of length segment_length (.05 seconds default)
        and then Fourier transform is performed on each of these chunks
        '''
        self.segment_length = seg_length
        '''read in song class labels'''
        prepro = PreProcessor()
        self.labels = prepro.getSongLabels()

    def split_recording(self, recording, segment_length, sample_rate):
        ''' 
        The code below splits the recording 
        into overlapping segments (1/2 segment overlap)
        INPUT: recording = 2d numpy array of song data in time domain; segment_length = 
        length of song segments (float); 
        sample_rate = float, sampling rate of song
        '''
        '''RETURNS: segments =  a list of 2d numpy arrays (floats) corresponding to overlapping
        song segments of length segment_length (this is all in time domain here)'''
        try:
            segments = []
            index = 0
            pre_count = 0
            while index < len(recording):    
                if segment_length * pre_count < self.song_begin: 
                    index += segment_length * sample_rate  
                    pre_count += 1
                    continue
                segment = recording[index:(index + segment_length * sample_rate)]
                segments.append(segment)
                index += np.round(segment_length * sample_rate * .5)
                pre_count += 1
                if segment_length * pre_count > self.song_end:
                    break
        except Exception, e:
            print e
            raise
        return segments
    
    def get_segments_temporally(self, file_name, sample_rate, recording):
        '''
        This is a somewhat redundant function, but I kept it modular 
        anyways.  This calls the split_recording
        function to return the splitted-into-segments song data as segments
        '''
        ''' 
        INPUT: file_name = name of wav file as string;
        sample_rate = sampling rate of song (float); recording = 2d
        numpy array (floats) containing song data
        '''
        '''
        RETURNS: segments =  a list of 2d numpy arrays 
        (floats) corresponding to overlapping
        song segments of length segment_length (this is all in time domain)   
        '''
    
        self.recording = recording 
        segment_length = self.segment_length
        try:
            segments = self.split_recording(recording, segment_length, sample_rate)
        except Exception, e:
            print e
            raise
        return segments
    
    
    
    def calculate_normalized_power_spectrum(self, segment, sample_rate):
        '''
        This code was pulled from online, I did not write it- 
        it is standard Fast Fourier Transform analysis using the 
        numPy framework.  This takes the Fourier transform of a song segment
        and returns the song's frequencies and corresponding magnitudes
        INPUT: segment = song segment as
        2d numpy array (floats) containing song data (2 song tracks per segment); 
        sample_rate = sampling rate of song (float).
        RETURNS: frequencies = 1d numpy array of frequencies returned by FFT;
        magnitudes = 1d numpy array (of floats) of FFT magnitudes at each frequency
        '''
        
        '''np.fft.fft returns the discrete fourier transform of the recording'''
        fft = np.fft.fft(segment) 
        number_of_samples = len(segment)        
        '''sample_length is the length of each sample in seconds'''
        sample_length = 1. / sample_rate 
        '''
        fftfreq is a convenient function which returns
        the list of frequencies measured by the fft
        '''
        frequencies = np.fft.fftfreq(number_of_samples, sample_length)  
        positive_frequency_indices = np.where(frequencies > 0) 
        '''positive frequences returned by the fft'''
        frequencies = frequencies[positive_frequency_indices]
        '''magnitudes of each positive frequency in the recording'''
        magnitudes = abs(fft[positive_frequency_indices]) 
        '''some segments are louder than others, so normalize each segment'''
        magnitudes = magnitudes / np.linalg.norm(magnitudes)
        return frequencies, magnitudes
    

    def create_power_spectra_array(self, segment_length, sample_rate):
        '''
        I did not write this code, it was pulled from online.  
        INPUTS: segment_length = (float) song segment length; 
        sample_rate = sampling rate of song (float)
        RETURNS: zero-initialized 1d numpy array
        '''
        number_of_samples_per_segment = int(segment_length * sample_rate)
        time_per_sample = 1. / sample_rate
        frequencies = np.fft.fftfreq(number_of_samples_per_segment, time_per_sample)
        positive_frequencies = frequencies[frequencies > 0]
        power_spectra_array = np.empty((0, len(positive_frequencies)))
        return power_spectra_array
    
    
    def fill_power_spectra_array(self, segments, power_spectra_array, sample_rate):
        '''
        This function takes all of the song's segments, performs FFT 
        on them to get the power vs. frequency information,
        and then returns the full power vs. frequency/time 
        data (time because each song's segment corresponds
        to a different time)
        INPUTS: segments =  a list of 2d numpy arrays (floats) 
        corresponding to overlapping song segments of length segment_length 
        (this is all in time domain); power_spectra_array = zero-initialized
        1d numpy array; sample_rate = sampling rate of song (float)
        OUTPUTS: 2d numpy array (floats) containing all FFT magnitude data 
        from the inputted song segments
        '''
        try:
            filled_array = power_spectra_array
            curr = 0
            prevLen = len(segments[0])
            currLen = len(segments[0])
            for segment in segments:
                freqs, mags = self.calculate_normalized_power_spectrum(segment, sample_rate)
                '''
                there is some redundancy in the next several 
                lines because I have surrounded everything in a try/catch, but in any case
                these lines are meant to handle errors 
                stemming from different segment lengths
                '''
                if curr == 0 :
                    prevMags = mags
                curr += 1
                currLen = len(segment)
                if prevLen != currLen:
                    break
                '''
                the .wav files actually contain two tracks, and the 
                corresponding Fourier magnitudes are contained in
                mags.T[0] and mags.T[1].  We use a sum of squares 
                to get a metric resembling energy (e.g. electromagnetic
                field energy is proportional to E**2 + B**2)
                '''
                filled_array =\
                np.vstack((filled_array, mags.T[0]*mags.T[0]+mags.T[1]*mags.T[1]))
                prevMags = mags
            return filled_array
        except Exception, e:
            print e
            print "FAILURE"
            raise

    def get_recording(self, file_name):
        '''
        This reads the song's wave data into recording, which gets returned
        INPUT: file_name = string, name of wav file
        OUTPUT: recording = 2d numpy float array of song data in time domain;
        sample_rate = float, sampling rate of song
        '''
        song_name = Globals.getSongDir() + "\\" + file_name
        sample_rate, recording = wavfile.read(song_name)
        return sample_rate, recording
    
    def extract_beat_via_acf(self, power_w_t):
        '''
        This function uses autocorrelation to extract beat data from the song.
        
        INPUT: power_w_t = 2d numpy float array containing maximum 
        power in each frequency bin w at all times t
        
        We apply autocorrelation function to each frequency band in power_w_t.
        The idea is to look at beat separation data for several sub-segments 
        within the song.  We want to see how this
        beat separation looks and varies throughout our song for each 
        frequency band.  By analyzing beat data in each
        frequency band, we can extract beat information for 
        different instruments, both low and high frequencies. We return beat 
        statistics like mean, median, and standard deviation.
        
        RETURNS: ALL OF FOLLOWING ARE 1d NUMPY ARRAYS CONTAINING FLOATS-
        mean beat separation, standard deviation of beat separation (float),
        median of beat separation (float), total-song beat strengths.   Each location
        in these outputs corresponds to a given frequency bin
        '''
        
        '''total number of sub-segments'''
        sub_segment_total = 5.
        '''size of sub-segments used in autocorrelation analysis'''
        window_size = int(np.round((self.song_end - self.song_begin) / float(self.segment_length) / sub_segment_total))
        starting_time = window_size
        ending_time = power_w_t.shape[0]
        all_strengths = []
        all_locations = []
        acf_delay = 4
        counts = 0
        '''
        This for loop gets autocorrelation or 'beat' 
        data for each of the song's sub-segments as described above
        '''
        for current_time in xrange(starting_time, ending_time, window_size):
            corr_strengths = []
            corr_locations = []
            '''
            get the autocorrelation data for 
            each of the frequency bins in power_w_t
            '''
            for j in xrange(len(power_w_t[0])):
                ac = acf(power_w_t[current_time - window_size: current_time, j], nlags = 120)[acf_delay:]
                corr_strengths.append(np.max(ac))
                corr_locations.append(np.argmax(ac) + acf_delay)
            corr_strengths = np.array(corr_strengths)
            corr_locations = np.array(corr_locations)
            counts += 1
            '''
            We want to stack our correlation data for each song segment
            '''
            if counts == 1:
                all_strengths = corr_strengths
                all_locations = corr_locations
            else:
                all_strengths = np.vstack((all_strengths,corr_strengths))
                all_locations = np.vstack((all_locations,corr_locations))
        '''
        This next analysis is a bit different.  Now we look at the correlation 
        strength for the entire song segment, not the sub segments.
        This helps to answer the question of does the song have a strong, 
        consistent beat in a given frequency band
        '''
        acf_glob_maxs, acf_glob_locs = [],[]
        for j in xrange(len(power_w_t[0])):
            acf_glob_maxs.append(np.max(acf(power_w_t[:,j], nlags = np.round(window_size / sub_segment_total))[acf_delay:]))
            acf_glob_locs.append(np.argmax(acf(power_w_t[:,j], nlags = np.round(window_size / sub_segment_total))[acf_delay:]) + acf_delay)
        return np.mean(all_locations, axis = 0) * self.segment_length,\
            np.std(all_locations, axis = 0) * self.segment_length,\
            np.median(all_locations, axis = 0) * self.segment_length, acf_glob_maxs
    
    def get_frequency_domain_data(self, segment_length, sample_rate, segments):
        ''' 
            This function returns both the song-averaged power data and
            beat data as a function of frequency band.
            Definition: Power_w_t contains the maximal power 
            contained in a frequency bin at w at a given time t
            
        INPUT:  segment_length = length of song segments (float); 
        sample_rate = float, sampling rate of song;
        segments =  a list of 2d numpy arrays 
        (floats) corresponding to overlapping
        song segments of length segment_length
        
        OUTPUT: 1d numpy array of power and beat data (floats); 
        1d numpy arrray of corresponding power and beat labels (strings); 
        mean_power (float); mean_maxs (float)
        '''
        try:
            power_w_t, corresponding_freqs, mean_power, mean_maxs =\
            self.get_power_vs_t_and_w(segment_length, sample_rate, segments)
        except Exception, e:
            print e
            raise
        '''get the mean power in each frequency for both of the tracks'''
        maximal_power_w = np.mean(power_w_t, axis = 0)
        '''use autocorrelation to extract the beat information from power_w_t'''
        beat_data = self.extract_beat_via_acf(power_w_t)
        '''get the labels of our power data'''
        power_labels = ["Pmax in " + freq for freq in corresponding_freqs]
        stacked_beat_data, beat_labels = self.h_stacker(beat_data, corresponding_freqs)
        '''
        Return the maximal_power_w combined with stacked_beat_data 
        as a single fequency domain row. Corresponding labels are also returned
        '''
        return np.hstack((maximal_power_w, stacked_beat_data)),\
            np.hstack((np.array(power_labels), np.array(beat_labels))), mean_power, mean_maxs
    
    def get_zcr_data(self, recording):
        '''
        This function gets the crossing data for each recording track
        (the associated analysis is in the time domain)
        INPUTS: recording = 2d numpy array of song data in time domain
        '''
        '''
        RETURNS: mean, median, and variance data for both song
        tracks, all as floats
        '''
        recording = np.array(recording)
        median_a, mean_a, var_a = self.get_crossings_one_track(recording[:, 0])
        median_b, mean_b, var_b = self.get_crossings_one_track(recording[:, 1])
        return median_a, mean_a, var_a, median_b, mean_b, var_b
        
    def get_crossings_one_track(self,input_track):
        '''
        This function returns all crossing separation data
        in the time domain for one track
        INPUTS: input_track = 2d numpy float array of song data in time domain
        RETURNS: mean, median, and variance data, all as floats
        '''
        a = input_track
        '''get the zero crossing locations'''
        zero_crossings = np.nonzero(np.diff(np.sign(a)))[0]
        '''get the separation between each crossing'''
        crossing_sep_lengths = np.diff(zero_crossings)
        mean_sep_length = crossing_sep_lengths.mean()
        var_sep_length = crossing_sep_lengths.var()
        median_sep_length = np.median(crossing_sep_lengths)
        return median_sep_length, mean_sep_length, var_sep_length   
    
    def get_time_domain_data(self, recording):
        '''
        We return the median, median, and variance of the crossing data 
        for both of the song's tracks
        INPUTS: recording = 2d numpy array of song data in time domain
        RETURNS: 1d numpy array- median, mean, and variance of crossing time data for tracks 
        a and b of the recording; 1d numpy array of crossing data labels
        '''
        '''
        c.t. stands for crossing time (time between x-axis crossings) in this context. 
        It encodes the same information as crossing rate
        '''
        median_a, mean_a, var_a, median_b, mean_b, var_b = self.get_zcr_data(recording)
        ''' corresponding labels '''
        labels = ["A- Med(c.t.)", "A- Mean(c.t.)", "A- Var(c.t.)", "B- Med(c.t.)", "B- Mean(c.t.)", "B- Var(c.t.)"]
        '''
        Return the data as a numpy array that 
        will be combined with the frequency domain data
        '''
        return np.array([median_a, mean_a, var_a, median_b, mean_b, var_b]), np.array(labels)
        
    def h_stacker (self, input_data, corresponding_freqs):
        '''
        This stacks inputted data and their corresponding 
        labels and returns nicely structured numpy arrays
        INPUT: input_data = list of 1d numpy arrays (floats); 
        corresponding_freqs = list of floats
        RETURNS: output = 2d numpy array, beat_labels = 1d numpy array
        '''
        output = []
        labels = ["<Beat sep> (seconds) in ", "std(Beat sep) (seconds) in ", "med(Beat sep) (seconds) in ", "Beat strength in " ]
        beat_labels = []
        for i in xrange(0, len(input_data)):
            output = np.hstack((output, input_data[i]))  
            curr_lab = labels[i]
            curr_labels = np.array([curr_lab + freq for freq in corresponding_freqs])
            beat_labels = np.hstack((beat_labels, curr_labels))
        return np.array(output), beat_labels
    
    def get_power_vs_t_and_w(self, segment_length, sample_rate, segments):
        '''
        This function returns the maximal power in 
        each frequency bin at a given song time
        INPUT:  segment_length = length of song segments (float); 
        sample_rate = float, sampling rate of song;
        segments =  a list of 2d numpy arrays 
        (floats) corresponding to overlapping
        song segments of length segment_length
        OUTPUT: all_max_by_bin = 2d numpy array (floats) containing
        maximal power in each frequency bin at a given song time; 
        corresponding_freqs = 1d numpy array containing frequency for each bin (floats);
        power_mean = (float) mean power in song, max_mean=(float) mean of max data
        '''
        
        '''
        This creates an empty power spectra array
        ''' 
        power_spectra_array = self.create_power_spectra_array(segment_length, sample_rate)
        try:
            '''
            this is the unbinned or raw FFT amplitude 
            data for each song segment
            '''
            power_spectra_array = self.fill_power_spectra_array(segments, power_spectra_array, sample_rate)
        except Exception, e:
            print e
            raise
        '''
        this gets the volume level of the song
        (e.g. this will be lower for classical music)
        '''
        power_mean = np.mean(power_spectra_array)
        '''scale our power spectra array to the mean'''
        power_spectra_array = power_spectra_array / power_mean
        number_of_samples = int(segment_length * sample_rate)
        sample_length = 1. / sample_rate 
        '''get the corresponding frequencies'''
        frequencies = np.fft.fftfreq(number_of_samples, sample_length)
        try:
            bands, corresponding_freqs = self.get_bands(frequencies)
            all_max_by_bin = [self.get_max_by_bin(power_spectra_array[song_time], bands) for song_time in xrange(len(power_spectra_array))]
            max_mean = np.mean(all_max_by_bin)
            all_max_by_bin = np.array(all_max_by_bin) / max_mean
        except Exception, e:
            print e
            raise
        return all_max_by_bin, corresponding_freqs, power_mean, max_mean
    
    
    def convert_one(self, file_name, segment_length, sample_rate, segments, recording):
        '''
        This function returns all of the feature data which includes
        both frequency domain and time domain data
        
        INPUT: recording = 2d numpy float array of song data in time domain; 
        segment_length = length of song segments (float); 
        sample_rate = (float) sampling rate of song;
        segments =  a list of 2d numpy arrays 
        (floats) corresponding to overlapping
        song segments of length segment_length (this is all in time domain);
        file_name = (string) name of wav
        
        OUTPUT: song_data = 1d numpy array of floats containing all of the 
        song's feature data, column_labels = 1d numpy array of strings
        corresponding to feature labels
        '''
        try:
            '''
            this gets the song's aggregated feature 
            row data in the frequency domain (just one row!)
            '''
            freq_row, freq_labels, mean_power, mean_maxs =\
            self.get_frequency_domain_data(segment_length, sample_rate, segments)
            '''
            this gets the song's aggregated feature row data
            in the time domain (just one row!)
            '''
            temporal_row, time_labels = self.get_time_domain_data(recording)
            
            '''
            we then combine the time and frequency domain data 
            to get the song's feature data (just one row per song)
            '''
            feature_row = np.hstack((freq_row, temporal_row))
            '''
            poor style here- I added some other information to 
            the song's feature row data in the next code-line.
            It would be better to make the code more 
            modular so that any type of feature can be
            added into the feature row with ease.
            '''         
            feature_row = np.hstack ((feature_row, np.array([mean_power, mean_maxs])))
            '''
            return the entire feature row as 
            well as the labels for each frequency
            '''
            return feature_row, np.hstack((np.hstack((freq_labels,time_labels)), np.array(["Mean power", "Mean _maxs"])))
        except Exception, e:
            print e
            raise

    def get_bands(self, frequencies):
        '''
        This function divides the song's frequencies into bins,
        can have logarithmic or linear binning.  Returns
        the frequency bins
        INPUT: 1d numpy array of frequencies
        OUTPUT: list of lists (containing ints) (frequency band locations);
        1d numpy array of strings (frequency band labels)
        '''
        try:
            start_freq = 0
            '''
            some initializations depending on how we split the bands
            we can't hear past 20,000 hz so stop at 20,000 hz
            '''
            if self.band_type == "log":
                inc_fac = (20000. / 100.)**(1. / (self.freq_bin_total - 1.))
                end_freq = 100.
                increment_function = self.multiply_function
            else:
                inc_fac = (20000. /  (self.freq_bin_total))
                end_freq = inc_fac
                increment_function = self.add_function
            bands = []
            corresponding_freqs = []
            '''
            this generates the actual frequency bands 
            as a list of lists in the python variable 'bands'
            '''
            for i in xrange(self.freq_bin_total):
                start_arg = np.argmin(np.abs(frequencies - start_freq)) + 1
                end_arg = np.argmin(np.abs(frequencies - end_freq))
                bands.append(range(start_arg, end_arg+1))
                corresponding_freqs.append(str(np.round(end_freq * 100.)/100.)+ " hz range")
                start_freq = end_freq
                end_freq = increment_function(start_freq, inc_fac)
            return bands, corresponding_freqs
        except Exception, e:
            print e
            raise
    
    def multiply_function(self, start_freq, inc_fac):
        return start_freq * inc_fac
    
    def add_function(self, start_freq, inc_fac):
        return start_freq + inc_fac

    def get_max_by_bin(self, target, bands): 
        '''
        Returns the maximum power within a given frequency band
        INPUT: target = 1d numpy array of power data at a given time (floats);
        bands = list of integer lists (frequency band data)
        '''
        nbins = self.freq_bin_total
        n = np.round(len(target) / nbins)
        maxs = [np.max(target[np.array(band)]) for band in bands]
        return maxs
        
    def delete_files(self):
        '''
        This function deletes all waveform data .csv 
        when convert_all is run
        NO INPUTS OR OUTPUTS
        '''
        folder = Globals.getCsvDir()
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception, e:
                print e
                raise    
            
    def write_data_to_csv(self, song_data, song_name, genre, column_labels):
        '''
        This function writes the song's feature row data
        into a csv file for modeling
        INPUT: song_data = 1d numpy array of floats containing
        song feature data; song_name (string); genre (string);
        column_labels = 1d numpy array of strings, song feature labels
        OUTPUT: NONE
        ''' 
        csv_dir = Globals.getCsvDir()
        song_data = song_data.tolist()
        song_data.append(song_name)
        song_data.append(genre)
        column_names = [str(lab) for lab in column_labels]
        column_names.append("Song Name")
        column_names.append("Label")
        with open(csv_dir + "\\" + song_name + '.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(column_names)
            writer.writerow(song_data)
       
    def convert_all(self):
        '''
        This function can be thought of as the main, where all
        song wav data is converted. nothing is returned
        INPUT: NONE
        OUTPUT: NONE
        '''
        self.delete_files()
        song_dir = Globals.getSongDir()
        file_list = [f for f in listdir(song_dir) if isfile(join(song_dir, f))]
        read_count = 0
        incr = 1
        '''Begin conversion of all songs'''
        for file_name in file_list:
            if read_count > self.files_read_count:
                break
            if ".wav" not in file_name:
                break
            song_position = 1
            cycles = 0
            '''get the .wave recording using numpy functionality'''
            sample_rate, recording = self.get_recording(file_name)
            '''
            This while loop breaks only when the cycles condition 
            below is reached. the code allows for the song to be analyzed 
            in separate segments, but for now we just work with one segment
            as noted by the cycles > 0 termination condition.  
            the best way to solve this problem is to analyze the song as
            multiple segments, where each segment is individually listened 
            to by a human and classified manually.  This adds 
            a level of diversification to modeling songs.  I did not get 
            around to multiple song feature rows per song so here we
            just output one feature row per song for simplicity
            '''
            while 1==1:
                self.song_begin = float(song_position * 30)
                self.song_end = float(self.song_begin + 90)
                try:
                    '''split the song into .05 second segments'''
                    segments =\
                    self.get_segments_temporally(file_name, sample_rate, recording)
                    '''get all of the song's feature row data'''
                    song_data, column_labels = \
                    self.convert_one(file_name, self.segment_length, sample_rate, segments, recording)  
                    print file_name
                except Exception, e:
                    print e
                    break
                '''
                write one feature row for this part of the 
                song using write_data_to_csv
                '''
                '''
                ZULU6FOX6TROT is nothing but a delimiter 
                to ease splitting later on
                '''
                self.write_data_to_csv(song_data,\
                file_name.split(".wav")[0]+"ZULU6FOX6TROT" +str(int(self.song_begin))\
                + str(int(self.song_end)), self.labels[file_name], column_labels)
                song_position += incr
                cycles += 1
                if cycles > 0:
                    break 
            read_count += 1
        
    def df_stitcher(self):
        dir = Globals.getCsvDir()
        '''
        This loads all of the outputted csv data from 
        convert_all into the pandas data frame for analysis.
        It returns just the pandas data frame
        INPUT: none
        OUTPUT: pandas data frame containing feature data
        '''
        '''
        this could also go into the Modeler file/class
        '''
        file_list = [f for f in listdir(dir) if isfile(join(dir, f))]
        df = pd.DataFrame()
        for file_name in file_list:
            try:
                df_dum = pd.read_csv(dir + "\\" + file_name)
            except:
                continue
            df = df.append(df_dum, ignore_index=True)
        df = df.dropna()
        return df