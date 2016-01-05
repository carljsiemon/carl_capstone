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
from preprocessor import PreProcessor
import os, shutil


class WaveReadAdvanced:
    def __init__(self, read_count=5000, seg_length=.05):
        self.band_type = "log"
        self.files_read_count = read_count  # # number of files to be read
        self.freq_bin_total = 15
        self.segment_length = seg_length
        prepro = PreProcessor()
        self.labels = prepro.getSongLabels()

    def split_recording(self, recording, segment_length, sample_rate):
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
    
    def calculate_normalized_power_spectrum(self, recording, sample_rate):
        # np.fft.fft returns the discrete fourier transform of the recording
        fft = np.fft.fft(recording) 
        number_of_samples = len(recording)        
        # sample_length is the length of each sample in seconds
        sample_length = 1. / sample_rate 
        # fftfreq is a convenience function which returns the list of frequencies measured by the fft
        frequencies = np.fft.fftfreq(number_of_samples, sample_length)  
        positive_frequency_indices = np.where(frequencies > 0) 
        # positive frequences returned by the fft
        frequencies = frequencies[positive_frequency_indices]
        # magnitudes of each positive frequency in the recording
        magnitudes = abs(fft[positive_frequency_indices]) 
        # some segments are louder than others, so normalize each segment
        magnitudes = magnitudes / np.linalg.norm(magnitudes)
        return frequencies, magnitudes
    
    
    
    
    
    
    def create_power_spectra_array(self, segment_length, sample_rate):
        number_of_samples_per_segment = int(segment_length * sample_rate)
        time_per_sample = 1. / sample_rate
        frequencies = np.fft.fftfreq(number_of_samples_per_segment, time_per_sample)
        positive_frequencies = frequencies[frequencies > 0]
        power_spectra_array = np.empty((0, len(positive_frequencies)))
        return power_spectra_array
    
    def fill_power_spectra_array(self, splits, power_spectra_array, fs):
        try:
            filled_array = power_spectra_array
            curr = 0
            prevLen = len(splits[0])
            currLen = len(splits[0])
            for segment in splits:
                freqs, mags = self.calculate_normalized_power_spectrum(segment, fs)
                if curr == 0 :
                    prevMags = mags
                curr += 1
                currLen = len(segment)
                if prevLen != currLen:
                    break
                filled_array = np.vstack((filled_array, mags.T[0]*mags.T[0]+mags.T[1]*mags.T[1]))
                    #filled_array = np.vstack((filled_array, mags.T[1]*mags.T[1]))
                prevMags = mags
            return filled_array
        except Exception, e:
            print e
            print "FAILUREEEEEEEEEEEEEEEEEEEEE"
            raise
    
    def convert_one(self, file_name, segment_length, sample_rate, segments, recording):
        try:
            freq_row, freq_labels = self.get_frequency_domain_data(segment_length, sample_rate, segments)
            temporal_row, time_labels = self.get_time_domain_data(recording)
            feature_row = np.hstack((freq_row, temporal_row))
            return feature_row, np.hstack((freq_labels,time_labels))
        except Exception, e:
            print e
            raise
    
    def get_segments_temporally(self, file_name, sample_rate, recording):
        self.recording = recording 
        segment_length = self.segment_length
        try:
            segments = self.split_recording(recording, segment_length, sample_rate)
        except Exception, e:
            print e
            raise
        return segment_length, sample_rate, segments, recording
        
    def get_recording(self, file_name):
        song_name = Globals.getSongDir() + "\\" + file_name
        sample_rate, recording = wavfile.read(song_name)
        return sample_rate, recording
    
    def get_frequency_domain_data(self, segment_length, sample_rate, segments):
        ### this returns the maximal a(w,t)^2+b(w,t)^2 data by frequency band where the rows are different times and columns represent frequencies
        try:
            power_w_t, corresponding_freqs = self.get_power_vs_t_and_w(segment_length, sample_rate, segments)
        except Exception, e:
            print e
            raise
        #### get the mean power in each frequency for both of the tracks
        maximal_power_w = np.mean(power_w_t, axis = 0)
        ######## beat data
        beat_data = self.extract_beat_via_acf(power_w_t)
        power_labels = ["Pmax in " + freq for freq in corresponding_freqs]
        stacked_beat_data, beat_labels = self.h_stacker(beat_data, corresponding_freqs)
        return np.hstack((maximal_power_w, stacked_beat_data)), np.hstack((np.array(power_labels), np.array(beat_labels)))
    
    def h_stacker (self, input_data, corresponding_freqs):
        output = []
        labels = ["<Beat sep> (seconds) in ", "std(Beat sep) (seconds) in ", "med(Beat sep) (seconds) in ", "Beat strength in " ]
        beat_labels = []
        for i in xrange(0, len(input_data)):
            output = np.hstack((output, input_data[i]))
            #np.mean(all_locations, axis = 0), np.std(all_locations, axis = 0), np.median(all_locations, axis = 0), acf_glob_maxs   
            curr_lab = labels[i]
            curr_labels = np.array([curr_lab + freq for freq in corresponding_freqs])
            beat_labels = np.hstack((beat_labels, curr_labels))
        return np.array(output), beat_labels
        

    def get_power_vs_t_and_w(self, segment_length, sample_rate, segments):
        power_spectra_array = self.create_power_spectra_array(segment_length, sample_rate)
        try:
            power_spectra_array = self.fill_power_spectra_array(segments, power_spectra_array, sample_rate)
        except Exception, e:
            print e
            raise
        print power_spectra_array.shape
        power_spectra_array = power_spectra_array / np.mean(power_spectra_array)
        number_of_samples = int(segment_length * sample_rate)
        sample_length = 1. / sample_rate 
        frequencies = np.fft.fftfreq(number_of_samples, sample_length)
        stop_arg = np.argmin(np.abs(frequencies - 20000))
        #frequencies = frequencies[0:stop_arg + 1]
        try:
            bands, corresponding_freqs = self.get_bands(frequencies)
            all_max_by_bin = [self.get_max_by_bin(power_spectra_array[song_time], bands) for song_time in xrange(len(power_spectra_array))]
            all_max_by_bin = np.array(all_max_by_bin) / np.mean(all_max_by_bin)
        except Exception, e:
            print e
            raise
        return all_max_by_bin, corresponding_freqs
#         GROUP BY LOGARITHMIC FREQUENCY BANDS YA MOOK!!!!!!!!!!!!!!!

    def get_bands(self, frequencies):
        try:
            if self.band_type == "log":
                inc_fac = (20000. / 100.)**(1. / (self.freq_bin_total - 1.))
                start_freq = 0
                end_freq = 100.
                bands = []
                corresponding_freqs = []
                for i in xrange(self.freq_bin_total):
                    start_arg = np.argmin(np.abs(frequencies - start_freq)) + 1
                    end_arg = np.argmin(np.abs(frequencies - end_freq))
                    bands.append(range(start_arg, end_arg+1))
                    corresponding_freqs.append(str(np.round(end_freq * 100.)/100.)+ " hz range")
                    start_freq = end_freq
                    end_freq = start_freq * inc_fac
            else:
                inc_fac = (20000. /  (self.freq_bin_total))
                start_freq = 0
                end_freq = inc_fac
                bands = []
                corresponding_freqs = []
                for i in xrange(self.freq_bin_total):
                    start_arg = np.argmin(np.abs(frequencies - start_freq)) + 1
                    end_arg = np.argmin(np.abs(frequencies - end_freq))
                    bands.append(range(start_arg, end_arg+1))
                    corresponding_freqs.append(str(np.round(end_freq * 100.)/100.)+ " hz range")
                    start_freq = end_freq
                    end_freq = start_freq + inc_fac
            return bands, corresponding_freqs
        except Exception, e:
            print e
            raise
        
    def get_time_domain_data(self, recording):
        median_a, mean_a, var_a, median_b, mean_b, var_b = self.get_zcr_data(recording)
        labels = ["A- Med(c.t.)", "A- Mean(c.t.)", "A- Var(c.t.)", "B- Med(c.t.)", "B- Mean(c.t.)", "B- Var(c.t.)"]
        return np.array([median_a, mean_a, var_a, median_b, mean_b, var_b]), np.array(labels)
        
    def get_zcr_data(self, recording):
        recording = np.array(recording)
        median_a, mean_a, var_a = self.get_crossings_one_track(recording[:, 0])
        median_b, mean_b, var_b = self.get_crossings_one_track(recording[:, 1])
        return median_a, mean_a, var_a, median_b, mean_b, var_b
        
    def get_crossings_one_track(self,input_track):
        a = input_track
        zero_crossings = np.nonzero(np.diff(np.sign(a)))[0]
        crossing_sep_lengths = np.diff(zero_crossings)
        mean_sep_length = crossing_sep_lengths.mean()
        var_sep_length = crossing_sep_lengths.var()
        median_sep_length = np.median(crossing_sep_lengths)
        return median_sep_length, mean_sep_length, var_sep_length     

    def get_max_by_bin(self, target, bands): 
        nbins = self.freq_bin_total
        n = np.round(len(target) / nbins)
        maxs = [np.max(target[np.array(band)]) for band in bands]
        return maxs
        
    def delete_files(self):
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
        self.delete_files()
        song_dir = Globals.getSongDir()
        file_list = [f for f in listdir(song_dir) if isfile(join(song_dir, f))]
        read_count = 0
        incr = 3
        for file_name in file_list:
            if read_count > self.files_read_count:
                break
            if ".wav" not in file_name:
                break
            song_position = 1
            cycles = 0
            sample_rate, recording = self.get_recording(file_name)
            while song_position > 0:
                self.song_begin = float(song_position * 30)
                self.song_end = float(self.song_begin + 30)
                try:
                    segment_length, sample_rate, segments, recording = self.get_segments_temporally(file_name, sample_rate, recording)
                    song_data, column_labels = self.convert_one(file_name, segment_length, sample_rate, segments, recording)
                    print file_name
                except Exception, e:
                    print e
                    break
                self.write_data_to_csv(song_data, file_name.split(".wav")[0]+"ZULU6FOX6TROT" +str(int(self.song_begin)) + str(int(self.song_end)), self.labels[file_name], column_labels)
                song_position += incr
                cycles += 1
                if cycles > 0:
                    break 
            read_count += 1
        
    def df_stitcher(self):
        dir = Globals.getCsvDir()
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
            
    def extract_beat_via_acf(self, power_w_t):
        window_size = int(np.round((self.song_end - self.song_begin) / float(self.segment_length) / 5.))
        starting_time = window_size
        ending_time = power_w_t.shape[0]
        all_strengths = []
        all_locations = []
        acf_delay = 4
        counts = 0
        for current_time in xrange(starting_time, ending_time, window_size):
            corr_strengths = []
            corr_locations = []
            for j in xrange(len(power_w_t[0])):
                ac = acf(power_w_t[current_time - window_size: current_time, j], nlags = 120)[acf_delay:]
                corr_strengths.append(np.max(ac))
                corr_locations.append(np.argmax(ac) + acf_delay)
            corr_strengths = np.array(corr_strengths)
            #print corr_strengths
            corr_locations = np.array(corr_locations)
            counts += 1
            if counts == 1:
                all_strengths = corr_strengths
                all_locations = corr_locations
            else:
                all_strengths = np.vstack((all_strengths,corr_strengths))
                all_locations = np.vstack((all_locations,corr_locations))
        acf_glob_maxs, acf_glob_locs = [],[]
        for j in xrange(len(power_w_t[0])):
            acf_glob_maxs.append(np.max(acf(power_w_t[:,j], nlags = np.round(window_size / 5.))[acf_delay:]))
            acf_glob_locs.append(np.argmax(acf(power_w_t[:,j], nlags = np.round(window_size / 5.))[acf_delay:]) + acf_delay)
        return np.mean(all_locations, axis = 0) * self.segment_length, np.std(all_locations, axis = 0) * self.segment_length, np.median(all_locations, axis = 0) * self.segment_length, acf_glob_maxs   