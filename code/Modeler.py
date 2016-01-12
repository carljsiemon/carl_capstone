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



class Modeler:
    def __init__(self, read_count=5000, seg_length=.05):
        reader = WaveRead(read_count, seg_length)
        self.df = reader.df_stitcher()
        self.corr_freq = []

    def fit_and_test_model(self,input_model, X_train, X_test, y_train, y_test ):
        model = input_model(learning_rate = .05, n_estimators=1000, min_samples_split=40, min_samples_leaf=3, max_depth=3)#learning_rate=0.5, n_estimators=100)
        ### **params
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        success_mask = y_predict == y_test
        return str(float(np.sum(success_mask))/ float(len(success_mask))), classification_report(y_test, y_predict), y_test, y_predict, model.feature_importances_ 

    def run_all_models(self):
        song_name = "Song Name"
        label_name = "Label"
        all_score = 0
        X, labels, song_names = self.sub_sample_min_label_amount()
        X_unscaled = X#scale(X, axis = 0)
        X = normalize(X, axis = 0)
        kf = KFold(X.shape[0], n_folds=4, shuffle=True)  
        all_scores = []  
        all_reports = []  
        all_tests = []
        all_predicts = []
        all_importances = []
        for train, test in kf:
            #X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=42)
            #self.run_supervised_learning(X_train, X_test, y_train, y_test)  
            results = self.run_supervised_learning(X[train], X[test], labels[train], labels[test])  
            all_scores.append(results[0])
            all_reports.append(results[1])
            all_tests.append(results[2])
            all_predicts.append(results[3])
            all_importances.append(results[4])
            if all_scores[len(all_scores) - 1] < .53:
                break
        #print all_scores
        print "K FOLD CROSS VAL SCORE " + str(np.mean(all_scores))     
#       song_names = np.array([str(song).decode('UTF-8', 'replace') for song in song_names]) 
        #pca = PCA(n_components=5)
        #X = pca.fit_transform(X)  
        all_score = np.mean(all_scores) 
        print X.shape
        print labels.shape
        print song_names.shape
        X, labels, song_names, good_mask = self.remove_outliers(X, labels,song_names)
        np.save("X_unscaled.npy", X_unscaled[good_mask])
        np.save("good_mask.npy", good_mask)
        end = False
        for i in xrange(2):
            output, nmf_out = self.run_unsupervised_learning(X, labels)
            if self.is_separate(nmf_out, labels):
                print "JUJU"
                end = True
                break
            if not self.is_separate(nmf_out, labels):
                print "MALO"
                end = False
            print "RECESSSSSSSSSSSSSSSSSSSSSSSS"
            print end
            print all_score
        return output, song_names, labels, nmf_out, all_reports, all_tests, all_predicts, all_importances

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
        return X, labels, song_names, good_mask
    
    
   
    
    
    def run_supervised_learning(self, X_train, X_test, y_train, y_test):
        #print "RANDOM FOREST SCORE " + self.fit_and_test_model(RandomForestClassifier, X_train, X_test, y_train, y_test)
        accuracy_score, report, y_test, y_predict, importances = self.fit_and_test_model(GradientBoostingClassifier, X_train, X_test, y_train, y_test)
        print "GRADIENT BOOST SCORE " + accuracy_score
        #print "SVM SCORE " + self.fit_and_test_model(SVC, X_train, X_test, y_train, y_test)
        print accuracy_score
        print report
        print " TYUPE IS"
        print type(report)
        return float(accuracy_score), report, y_test, y_predict, importances
        
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
        beat_strength_mask = np.array(['Beat strength in' in element for element in df.columns])
        power_mask =  np.array(['Pmax in' in element for element in df.columns])
        bpm_mask = np.array(['<Beat sep>' in element for element in df.columns]) 
        bpm_std_mask =  np.array(['std(Beat sep)' in element for element in df.columns]) 
        bpm_med_mask = np.array(['med(Beat sep)' in element for element in df.columns]) 
        zcr_mask = np.array(['c.t.' in element for element in df.columns]) 
        mean_power_mask = np.array(['Mean power' in element for element in df.columns])
        mean_maxs_mask = np.array(['Mean _maxs' in element for element in df.columns])  
        frequencies = [float(element.split("strength in ")[1].split(" hz")[0]) for element in  df.columns[beat_strength_mask]]
        
        df = df.drop(['Mean power','Song Name', 'Label'], axis = 1)
        
        column_names = df.columns
        
        return power_mask,beat_strength_mask, bpm_mask, bpm_med_mask, bpm_std_mask, zcr_mask, mean_maxs_mask, frequencies, column_names
    
    
    def is_separate(self,components, labels):
        foo = zip(components, labels)
        struct_array = np.array([np.array([element[0], element[1]]) for element in foo])
        maj_labels = []
        for name in set(struct_array[:,1]):
            mask = struct_array[:,1] == name
            maj_lab = Counter(struct_array[mask][:,0]).most_common(1)[0][0]
            print maj_lab
            maj_labels.append(maj_lab)
        print "LEN SET " + str(len(set(maj_labels)))
        if len(set(maj_labels)) >= 5:
            return True
        else: 
            return False    
        