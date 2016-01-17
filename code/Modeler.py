import numpy as np
import pylab
from os import listdir
from os.path import isfile, join
from Globals import Globals
import pandas as pd
from statsmodels.tsa.stattools import acf
from preprocessor import PreProcessor
from wave_read import WaveRead
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize, scale
from collections import Counter
from random import sample as smp
from ModelerAdvanced import ModelerAdvanced as advanced_model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_mlab_linkage
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from sklearn.decomposition import NMF
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report

'''
This class conducts unsupervised and supervised learning on the feature data 
'''
class Modeler:
    def __init__(self, read_count=5000, seg_length=.05):
        '''
        INPUTS: read_count = total number of songs to be read;
        seg_length = length of song segments that song will be split into
        '''
        reader = WaveRead(read_count, seg_length)
        '''load all csv featurized data into data frame'''
        self.df = reader.df_stitcher()
        self.corr_freq = []

    def fit_and_test_model(self,input_model, X_train, X_test, y_train, y_test ):
        '''
        From feature data and class labels, build supervised 
        learning model.  This function can be 
        made more polymorphic by passing in a dictionary
        of the model parameters, depending on the type of model.  
        GBC was found to be the best supervised model for this problem, 
        so we specify the model parameters within this function instead.
        
        INPUTS: input_model=inputted model; X_train, X_test, y_train, y_test = 
        2-d float (X) and 1-d string (y) numpy arrays of train and test data 
        for features and labels
        RETURNS: accuracy as string, confusion matrix as 2-d numpy array (ints),
        feature importance as 1-d numpy array (floats), the actual/predicted 
        test label results as 1-d numpy string arrays
        '''
        model = input_model(learning_rate = .05,\
        n_estimators=1000, min_samples_split=40, min_samples_leaf=3, max_depth=3)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        success_mask = y_predict == y_test
        '''
        return not only score but the confusion matrix and
        feature importances
        '''
        return str(float(np.sum(success_mask))/ float(len(success_mask))),\
            classification_report(y_test, y_predict),\
            y_test, y_predict, model.feature_importances_ 

    def run_all_models(self):
        '''
        This function conducts both supervised and unsupervised learning 
        on the feature matrix.  It returns kmeans and nmf-kmeans clustering
        results, feature importances, and confusion matrix data
        for supervised learning.  It also returns the predictions and actual
        test results for each of the kfolds
        INPUTS: none
        OUTPUTS: kmeans_out = 1-d numpy string array of kmeans cluster labels; 
        song_names = 1-d numpy string array of song names; 
        labels = 1-d numpy string array of genre labels ; 
        nmf_out = 1-d numpy string array of kmeans/nmf cluster labels; 
        all_reports = list of confusion matrices 
        (2-d numpy int arrays) from each KFold output; 
        all_tests = list of actual labels 
        (1-d numpy string arrays) from each KFOLD test, 
        all_predicts= list of predicted labels 
        (1-d numpy string arrays) from each KFOLD test; 
        all_importances= list of feature importances 
        (1-d numpy float arrays) for each KFOLD test
        '''
        
        '''
        Sub sample n_min from each class where n_min 
        is the smallest number of songs in a given genre, giving equals numbers
        in each class.  Little variation in accuracy between sub samples is
        observed
        '''
        X, labels, song_names = self.sub_sample_min_label_amount()
        X_unscaled = X
        '''
        Work with normalized data here, which is required for the 
        KMeans clustering performed below
        '''
        X = normalize(X, axis = 0)
        
        '''*************** SUPERVISED LEARNING ANALYSIS ******************* '''
        '''
        We use KFold cross validation here and 
        shuffle the data before separating into folds
        '''
        kf = KFold(X.shape[0], n_folds=4, shuffle=True)  
        '''Aggregate results for each of our folds'''
        all_scores = []  
        all_reports = []  
        all_tests = []
        all_predicts = []
        all_importances = []
        for train, test in kf:  
            '''Get model results for one fold'''
            results =\
            self.run_supervised_learning(X[train], X[test], labels[train], labels[test]) 
            '''Add results to aggregate lists''' 
            all_scores.append(results[0])
            all_reports.append(results[1])
            all_tests.append(results[2])
            all_predicts.append(results[3])
            all_importances.append(results[4])
        print "K FOLD CROSS VAL SCORE " + str(np.mean(all_scores))     
        #pca = PCA(n_components=5)
        #X = pca.fit_transform(X)  
        all_score = np.mean(all_scores) 
        
        '''*************** UNSUPERVISED LEARNING ANALYSIS *******************'''
        '''Remove outliers for kmeans algorithm'''
        X, labels, song_names, good_mask = self.remove_outliers(X, labels,song_names)
        np.save("X_unscaled.npy", X_unscaled[good_mask])
        np.save("good_mask.npy", good_mask)  
        '''
        Can check and see run to run variation in results
        produced by KMeans, hence the for loop below.
        For more than 1000 songs total, there is a very high probability of 
        kmeans' clusters containing 5 unique majority classes with
        little variation in purity, so we simply break from the for 
        loop here after one iteration,
        but leave the for loop for sake of completeness.
        '''
        for i in xrange(2):
            kmeans_out, nmf_out = self.run_unsupervised_learning(X, labels)
            print all_score
            break
        return kmeans_out, song_names, \
        labels, nmf_out, all_reports, all_tests, all_predicts, all_importances

    def remove_outliers(self, X, labels, song_names):
        ''' 
        This removes any outliers in our data set.  It returns
        the 'cleaned' inputted data
        
        INPUTS: X = feature data as 2d numpy array (floats); labels = genre 
        labels as 1d numpy array (strings);
        song_names = song names as 1d numpy array (strings)
        OUTPUTS: 'cleaned' input data X (floats), genre labels (1d numpy array of strings),
        1-d numpy array of song_names as strings; good_mask = mask 
        pointing to non-outlier data (Boolean 1d numpy array)
        '''
        mean_X = np.mean(X, axis = 0)
        std_X = np.std(X, axis = 0)
        diff = np.abs(mean_X - X) / std_X
        '''
        If any data is more than 3.5 standard deviations 
        away from the mean, we exclude it from our data set
        '''
        good_mask = np.max(diff, axis = 1) < 3.5
        X = X[good_mask]
        labels = np.array(labels)[good_mask]
        song_names = np.array(song_names)[good_mask]
        return X, labels, song_names, good_mask
    
    def run_supervised_learning(self, X_train, X_test, y_train, y_test):
        '''
        This function enables numerous models to be tested here. 
        Gradient boosting classifier gives the best results. 
        We return accuracy, the confusion matrix data, test/prediction
        results, as well as the feature importances
        
        INPUTS:
        X_train, X_test, y_train, y_test = 
        2d floats (X) and 1d strings (y) numpy arrays of train and 
        test data for features and labels
        
        RETURNS: accuracy score as float, confusion matrix as 2d numpy 
        integer array, actual 
        and predicted labels as 1d numpy arrays for KFOLD, and 
        feature importances as 2d numpy array of floats
        
        ALL of this is for 1 KFold
        '''
        #print "RANDOM FOREST SCORE " + self.fit_and_test_model(RandomForestClassifier, X_train, X_test, y_train, y_test)
        accuracy_score, report, y_test, y_predict, importances =\
        self.fit_and_test_model(GradientBoostingClassifier, X_train, X_test, y_train, y_test)
        print "GRADIENT BOOST SCORE " + accuracy_score
        #print "SVM SCORE " + self.fit_and_test_model(SVC, X_train, X_test, y_train, y_test)
        return float(accuracy_score), report, y_test, y_predict, importances
        
    def run_unsupervised_learning(self, X, input_labels):
        '''
        This function performs both KMeans and NMF-KMeans
        clustering.  KMeans by itself performs better.
        
        INPUTS: X = feature data as 2d numpy array (floats); 
        input_labels = 1d numpy array of genre labels (strings)
        RETURNS: cluster labels from KMEAN and cluster labels from
        KMEAN/NMF, both as 1d numpy arrays of strings
        '''
        
        '''get kmeans clusters'''
        labels_kmean = self.get_kmean_clusters(X)
        '''perform nonnegative matrix factorization'''
        model = NMF(n_components=5, init='random', random_state=0)
        '''get song-latent features matrix'''
        nmf_matrix = model.fit_transform(X)
        '''
        returns both plain kmeans and kmeans on
        nmf song-latent features matrix
        '''
        return labels_kmean, self.get_kmean_clusters(nmf_matrix)
     
    def get_kmean_clusters(self,X):
        '''
        Returns labels of kmeans clustering
        INPUTS: X = feature matrix as 2d numpy float array
        OUTPUTS: KMeans cluster labels as 1d numpy array of strings
        '''
        kmeans = KMeans(5)
        kmeans.fit_transform(X)
        return kmeans.labels_ 
  
    def sub_sample_min_label_amount(self):
        '''
        This function sub samples to ensure
        class balance
        '''
        '''
        INPUTS: none
        RETURNS: feature matrix X (2d), labels(1d), and the song_names (1d)
        as numpy arrays containing floats, strings, and strings, respectively
        '''
        smallest_count = Counter(self.df["Label"].values).most_common()[-1:][0][1]
        df = self.df
        cnt = 0
        for label in df["Label"].unique():
            curr_count = smallest_count
            if cnt == 0:
                output = np.array(smp(df[df["Label"] == label].values, curr_count))
                cnt = 1
            else:
                output =\
                np.vstack((output, np.array(smp(list(df[df["Label"] == label].values), curr_count))))
        return output[:,:-2], output[:,-1:].flatten(), output[:,-2:-1].flatten()
        
        
    def extract_frequencies_and_indeces(self):
        '''
        This is used for post-processing and visualization.  
        The main purpose is to return the column locations 
        (as masks) of each type of feature, as each feature type 
        has multiple values for each of the frequency bins.
        RETURNS: various masks (1d numpy Boolean arrays) that say where each feature groups'
        columns are located, as well as the corresponding column names. frequency names = 1d
        numpy array of strings giving names of each frequency band
        INPUTS: NONE
        '''
        df = self.df
        beat_strength_mask = np.array(['Beat strength in' in element for element in df.columns])
        power_mask =  np.array(['Pmax in' in element for element in df.columns])
        bpm_mask = np.array(['<Beat sep>' in element for element in df.columns]) 
        bpm_std_mask =  np.array(['std(Beat sep)' in element for element in df.columns]) 
        bpm_med_mask = np.array(['med(Beat sep)' in element for element in df.columns]) 
        zcr_mask = np.array(['c.t.' in element for element in df.columns]) 
        mean_power_mask = np.array(['Mean power' in element for element in df.columns])
        mean_maxs_mask = np.array(['Mean _maxs' in element for element in df.columns])  
        frequencies = [float(element.split("strength in ")[1].split(" hz")[0])\
                    for element in  df.columns[beat_strength_mask]]
        df = df.drop(['Mean power','Song Name', 'Label'], axis = 1)
        column_names = df.columns
        return power_mask,beat_strength_mask, bpm_mask,\
            bpm_med_mask, bpm_std_mask, zcr_mask, mean_maxs_mask, frequencies, column_names