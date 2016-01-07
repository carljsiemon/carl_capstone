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
from sklearn.preprocessing import normalize
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



class Modeler:
    def __init__(self, read_count=5000, seg_length=.05):
        reader = WaveRead(read_count, seg_length)
        self.df = reader.df_stitcher()
        self.corr_freq = []

    def fit_and_test_model(self,input_model, X_train, X_test, y_train, y_test ):
        model = input_model(learning_rate = .05, n_estimators=1000, min_samples_split=40, min_samples_leaf=3, max_depth=3)#learning_rate=0.5, n_estimators=100)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        success_mask = y_predict == y_test
        return str(float(np.sum(success_mask))/ float(len(success_mask)))



    def run_all_models(self):
        song_name = "Song Name"
        label_name = "Label"
        #X, labels, song_names = self.sub_sample_min_label_amount()
        #X = normalize(X, axis = 0)
        #np.save('X.npy', X)
        #np.save('song_names.npy', song_names)
        #np.save('labels.npy', labels)
        X = np.load('X.npy') ### this is the saved normalized data
        labels = np.load('labels.npy')
        song_names = np.load('song_names.npy')
        
        kf = KFold(X.shape[0], n_folds=5, shuffle=True)  
        all_scores = []    
        for train, test in kf:
            #X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=42)
            #self.run_supervised_learning(X_train, X_test, y_train, y_test)    
            print test.shape
            print train.shape
            all_scores.append(self.run_supervised_learning(X[train], X[test], labels[train], labels[test])) 
        print all_scores
        print "K FOLD CROSS VAL SCORE " + str(np.mean(all_scores))     
#       song_names = np.array([str(song).decode('UTF-8', 'replace') for song in song_names]) 
        #pca = PCA(n_components=5)
        #X = pca.fit_transform(X)  
        
        X, labels, song_names = self.remove_outliers(X, labels,song_names)
        output, nmf_out = self.run_unsupervised_learning(X, labels)
        #nmf_matrix, components = self.run_unsupervised_learning(X, labels)
        return output, song_names, labels, nmf_out

    def remove_outliers(self, X, labels, song_names):
        mean_X = np.mean(X, axis = 0)
        std_X = np.std(X, axis = 0)
        diff = np.abs(mean_X - X) / std_X
        good_mask = np.max(diff, axis = 1) < 3.5
        X = X[good_mask]
        print 'dead song size'
        print song_names[~good_mask].shape
        labels = np.array(labels)[good_mask]
        song_names = np.array(song_names)[good_mask]
        return X, labels, song_names
    
    
   
    
    
    def run_supervised_learning(self, X_train, X_test, y_train, y_test):
        #print "RANDOM FOREST SCORE " + self.fit_and_test_model(RandomForestClassifier, X_train, X_test, y_train, y_test)
        accuracy_score = self.fit_and_test_model(GradientBoostingClassifier, X_train, X_test, y_train, y_test)
        print "GRADIENT BOOST SCORE " + accuracy_score
        #print "SVM SCORE " + self.fit_and_test_model(SVC, X_train, X_test, y_train, y_test)
        return float(accuracy_score)
        
    def run_unsupervised_learning(self, X, input_labels):
        labels_kmean = self.get_kmean_clusters(X)
#         dendro = self.get_hierarchal_clusters(X, input_labels)
        model = NMF(n_components=5, init='random', random_state=0)
        nmf_matrix = model.fit_transform(X)
        #return nmf_matrix, model.components_
        return labels_kmean, self.get_kmean_clusters(nmf_matrix)
     
    def get_kmean_clusters(self,X):
        kmeans = KMeans(5)#, max_iter = 1000, n_init = 100)
        kmeans.fit_transform(X)
        return kmeans.labels_ 
     
    def get_hierarchal_clusters(self,X, input_labels):
        distxy = squareform(pdist(X, metric='cosine'))
        link = linkage(distxy, method='complete')
        np.save('linkage.npy', link)
        dendro = dendrogram(link, p = 2, color_threshold=2, leaf_font_size=9,  labels=input_labels)
        return dendro
    
    def sub_sample_min_label_amount(self):
        smallest_count = Counter(self.df["Label"].values).most_common()[-1:][0][1]
        df = self.df
        cnt = 0
        for label in df["Label"].unique():
            print label
            if label == 'roc' or label == 'pop':
                curr_count = smallest_count
            else:
                curr_count = smallest_count
            if cnt == 0:
                output = np.array(smp(df[df["Label"] == label].values, curr_count))
                cnt = 1
            else:
                output = np.vstack((output, np.array(smp(list(df[df["Label"] == label].values), curr_count))))
        return output[:,:-2], output[:,-1:].flatten(), output[:,-2:-1].flatten()
        
        
    def extract_frequencies_and_indeces(self):
        df = self.df
        mask = np.array(['Beat strength in' in element for element in df.columns])
        power_mask =  np.array(['Pmax in' in element for element in df.columns])
        frequencies = [float(element.split("strength in ")[1].split(" hz")[0]) for element in  df.columns[mask]]
        return mask, power_mask, frequencies
            