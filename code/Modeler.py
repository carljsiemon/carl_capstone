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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import normalize
from collections import Counter
from random import sample as smp
from ModelerAdvanced import ModelerAdvanced as advanced_model
class Modeler:
    def __init__(self, read_count=5000, seg_length=.05):
        reader = WaveRead(read_count, seg_length)
        self.df = reader.df_stitcher()
        self.corr_freq = []

    def fit_and_test_model(self,input_model, X_train, X_test, y_train, y_test ):
        model = input_model(n_estimators=1000, min_samples_split=40, min_samples_leaf=3, max_depth=3)#learning_rate=0.5, n_estimators=100)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        success_mask = y_predict == y_test
        return str(float(np.sum(success_mask))/ float(len(success_mask)))




#PLUCK OUT THE BEST POP AND ROCK SONGS FOR CLASSIFICATION
# >>> import numpy as np
# >>> from sklearn.cross_validation import KFold
# 
# >>> kf = KFold(4, n_folds=2)
# >>> for train, test in kf:
# ...     print("%s %s" % (train, test))
# [2 3] [0 1]
# [0 1] [2 3]




    def run_all_models(self):
        song_name = "Song Name"
        label_name = "Label"
#         X = self.df.drop([song_name, label_name], axis = 1)
#         labels = self.df[label_name]
#         
        X, labels = self.sub_sample_min_label_amount()
        X = normalize(X, axis = 0)
        print Counter(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=42)
#         modeler = advanced_model()
#         X_train, X_test, y_train, y_test = modeler.my_train_test_split(X,labels) 
        print "RANDOM FOREST SCORE " + self.fit_and_test_model(RandomForestClassifier, X_train, X_test, y_train, y_test)
        print 
        print "GRADIENT BOOST SCORE " + self.fit_and_test_model(GradientBoostingClassifier, X_train, X_test, y_train, y_test)
#         print "remember GBD is best with 1000 selectors"
        #print "SVM SCORE " + self.fit_and_test_model(SVC, X_train, X_test, y_train, y_test)

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
        return output[:,:-2], output[:,-1:].flatten()
        
        
    def extract_frequencies_and_indeces(self):
        df = self.df
        mask = np.array(['Beat strength in' in element for element in df.columns])
        power_mask =  np.array(['Pmax in' in element for element in df.columns])
        frequencies = [float(element.split("strength in ")[1].split(" hz")[0]) for element in  df.columns[mask]]
        return mask, power_mask, frequencies
            