##WaveToGenre: Dr. Carl Siemon's Capstone Project

### Overview

WaveToGenre is a data science project that uses machine learning (both unsupervised/supervised) to classify a song's musical genre based solely on its waveform. It uses Fourier Analysis techniques to convert a song's unstructured waveform data into a meaningful, structured feature space that allows for modeling.  The fully documented code for this project can be found [here](https://github.com/carljsiemon/carl_capstone/tree/master/code). 

### Motivation

There are several potential business applications for this technology.

The first applies to musical services like Pandora.  WaveToGenre can be used to recommend new songs to users that sound similar to what they are listening to.

WaveToGenre could also be useful to record labels.  They could use WaveToGenre to choose what singles to release next based on their similarity in sound to what is 'hot' or 'trending'.  

### Data

#### Data Source:

1000 .mp3 songs from my personal music library.  The total number of songs has been observed to have a dramatic effect on the predictive capability of the models used in this work.  Future work will involve a much larger data set of ~10,000 songs.

#### Data Scope:

I used 200 songs per genre for each of the 5 genres: hip-hop (rap), classical, techno, rock, and pop.  Itunes was used to convert the .mp3's to .wav's, which was required for my analysis.  To avoid difficulty in classifying the beginning of songs, song data between the 30 second and 2 minute mark was used.  

Below is a plot of the waveform data of 'Smile' by Tupac in the time domain.  All songs are divided into two tracks, which is why there are two curves displayed.

<p align="center">![im1](https://github.com/carljsiemon/carl_capstone/blob/master/images/signatime.png)


### Feature Engineering in the Frequency Domain:
The following steps are taken to begin the featurization of the the unstructured song data in the frequency domain:

1. Divide song into .05 second segments with 1/2 segment overlap between each segment.  These segments can be thought of 'instantaneous' snapshots of the song.
2. Perform Fast Fourier Transform on each segment.  This yields the Fourier wave amplitudes vs. frequency at each song time.  I will refer to the resulting data as P(w,t), where P represents the song 'power' contained in each frequency w at song time t.  The variables w and t are discrete.
3. The frequencies are binned (or grouped) into bands, where the width of each band can be either nonlinear or linear.  Since humans hear on a logarithmic scale, the bands are divided logarithmically (15 total).  We will refer to each frequency band as w'.
4. At each discrete time t, the maximal power P(w,t) is found in each band and stored in a new matrix Pmax(w',t).  The maximum is taken because humans tend to hear dominant amplitudes as opposed to the average or median.  There are many ways to work with Pmax(w',t).
### Extracting features from Pmax(w',t)
The following steps describe both what and how features are extracted from Pmax(w',t):
 
1. **Total song 'energy'**. Computed by averaging Pmax(w',t) over both w' and t.
2. **Average power (or simply 'energy') in each frequency band**.  Computed by averaging Pmax(w',t) temporally and dividing by total song energy. 
3. **Average separation between beats by frequency band**.  Computed by dividing Pmax(w',t) into several 'sub-songs' (we use 5 sub-songs here).  The location or 'lag' of the first peak (after lag 0) of the autocorrelation function is found for each segment, and the resulting lags are then averaged.
4. **Median separation between beats by frequency band**.  Similar to 3 except the median of the resulting lags is taken.
5. **Standard deviation of separation between beats by frequency band**.  Similar to 3 except the standard deviation of the resulting lags is taken.
6. **Total-song beat strength in each frequency band**.  The magnitude of the first peak (after lag 0) of the autocorrelation function of Pmax(w',t) is found for the entire song in each frequency band w'.

### Feature Engineering in the Time Domain:
The following describes how song feature data in the time-domain (no Fourier analysis) is extracted.  Specifically, we look at ZCR, the zero crossing rate, which is the rate at which the song's signal crosses zero per unit time.
  
To characterize ZCR for each song, the following features are computed:

* **Average crossing time separation**. Computed by finding the time separations between each zero crossing and then averaging the resulting data.
* **Median crossing time separation**. Computed by finding the time separations between each zero crossing and then taking the median of the resulting data.
* **Variance of crossing time separation**. Computed by finding the time separations between each zero crossing and then finding the variance of the resulting data.
### Summary of features:
1. In summary, we have the following types of features:
 	* **Total song 'energy'**
	* **Average power (or simply 'energy') in each frequency band**
	* **Average separation between beats by frequency band**
	* **Median separation between beats by frequency band**
	* **Standard deviation of separation between beats by frequency band**
	* **Total-song beat strength in each frequency band**
	* **Average crossing time separation**
	* **Median crossing time separation**
	* **Variance of crossing time separation**

### Collecting and Storing Feature Data:
The above described feature data is stored as a single feature row in one .csv file per song.  In addition to the feature data, each .csv file contains the song name and genre label, which are denoted by either 'hip', 'cla', 'roc', 'pop', and 'tec' for the genres hip-hop, classical, rock, pop, and techno, respectively.

### Loading and Cleaning Data for Analysis
1.  All .csv's are loaded into a Pandas data frame for analysis.  Any rows containing null data are dropped from the dataframe.    
2.  The Pandas dataframe is separated into:
	* Numerical (floats) feature data as a 2-d numpy array, commonly referred to as X. 
	* Genre labels as a 1-d numpy string array, commonly referred to as y.
### Do we have signal from our feature data? Let's examine the plots below.
<p align="center">![im2](https://github.com/carljsiemon/carl_capstone/blob/master/images/scatterplots.png)
<p align="center">![im3](https://github.com/carljsiemon/carl_capstone/blob/master/images/normalizedenergy.png)
<p align="center">![im4](https://github.com/carljsiemon/carl_capstone/blob/master/images/beatstrength.png)

The evident separation in these plots is highly suggestive that our feature engineering has produced classifiable, structured data.

### Supervised Modeling: Gradient Boosting Classifier
Gradient boosting classifier (GBC) was found to give the highest supervised classification scores, with a right/(wrong+right) K-Fold (4 folds) cross-validated accuracy of 67 percent.  The following sklearn model parameters were used: learning-rate = .05, n-estimators=1000, min-samples-split=40, min-samples-leaf=3, and max-depth=3. 

The plots below summarize the overall ability of the GBC to classify data.  As indicated by the confusion matrix, we observe significant confusion between rock and pop, probably due to the fact that these two genres can sometimes sound similar. 

<p align="center">![im5](https://github.com/carljsiemon/carl_capstone/blob/master/images/confusionmat.png)
<p align="center">![im6](https://github.com/carljsiemon/carl_capstone/blob/master/images/precrec.png)

The GBC returns the feature importances displayed below. The top and bottom of the 'error' bars on the first plot denote the maximum and minimum of the feature importances within each group.  The levels of the blue bar plots in the first plot corresponds to the average feature importance within the group.  The individual features in the second plot can be read using the the zooming feature of your browser. 

<p align="center">![im7](https://github.com/carljsiemon/carl_capstone/blob/master/images/groupedimportances.png)
<p align="center">![im8](https://github.com/carljsiemon/carl_capstone/blob/master/images/allimportances.png)

### Unsupervised learning: K-Means Clustering
Before unsupervised learning was conducted, our feature data was normalized and then outliers were removed from the data set due to the importance of distances in the algorithms that were used.  Feature rows containing any column data that was more than 3.5 standard deviations from the column-mean were considered to be outliers.

In an ideal world, we can use unsupervised learning techniques to separate our song data into 5 separate clusters where the majority composition of each cluster corresponds to a different song genre, and the majority-percentage is close to 100 percent.  In other words, clusters labeled '1', '2', '3', '4', and '5' would contain only hip-hop, techno, rock, classical, and pop songs, respectively. 


K-Means Clustering:

1. K-Means clustering was performed to classify our feature data into 5 clusters and this technique was found to achieve the highest average cluster purity of around 60%.  The following model parameters were used: n-clusters=5, init='k-means++', n-init=10, max-iter=300, tol=.0001.
2. The summary of cluster purity from K-Means clustering is given below:
<p align="center">![im9](https://github.com/carljsiemon/carl_capstone/blob/master/images/clusterpurity.png)

NMF/K-Means clustering:

Non-negative Matrix Factorization (NMF) combined with K-Means was also attempted, but this resulted in lower average cluster purity.  We describe the process here for completeness.

* Main idea behind the approach: NMF reduces the n-dimensional feature data into a lower dimensional space containing k 'latent features', increasing the density of our data and hopefully enabling better classification.
* NMF results in two matrices whose product approximates our feature data: X ~ W X H. W is an m x k dimension matrix whose rows correspond to m songs and columns correspond to the k latent features.  H is a k x n matrix whose rows correspond to the k latent features and columns correspond to the n original features.
* NMF with m = 1000 songs and k = 5 was used here (corresponding to 5 latent features).  
* Since we are interested in classifying songs, K-Means clustering is performed on the W matrix.  As mentioned above, this resulted in a lower accuracy than our K-Means clustering on the unmodified (but normalized) feature matrix.

### Tools

1. [Python](https://www.python.org/): the main coding language for this project.
2. [sklearn](http://scikit-learn.org/): Scikit-Learn, a Python library that provides machine learning libraries and packages.

### How can the model be improved?
Besides adding more songs to the data set, songs can be divided into a set of 30 second chunks (or smaller) instead of the 90 minute time frame used here.  Each of these song chunks could be listened to separately and then classified.  A model could then be trained on the resulting ensemble of data.  To classify a new song, its 30 second chunks would be classified by the model, and the majority genre leader for the song is then what the song would be classified as.  This adds a layer of diversification/redundancy in classification and helps to address the issues that arise from one part of a song 'sounding different' than another (for example, a rock song might sound more classical in certain parts, such as 'Bohemian Rhapsody' by Queen).   

### Acknowledgements

* Thank you [Galvanize gSchool / Zipfian Academy](http://www.zipfianacademy.com/) for sharpening my machine learning and predictive modeling sills.

* Thank you fellow Zipfian Students for both insight and advice.



