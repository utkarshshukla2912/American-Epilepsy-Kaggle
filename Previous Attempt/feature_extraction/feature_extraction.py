#!/usr/bin/env python
# encoding: utf-8
# -*- coding: utf8 -*-
from __future__ import unicode_literals
import csv
import scipy.io as si
import numpy as np 
import scipy as sc
import scipy.stats
from scipy import signal
import os
from numpy.fft import fft
from numpy import zeros, floor, log10, log, mean, array, sqrt, vstack, cumsum, \
                  ones, log2, std
from numpy.linalg import svd, lstsq
import time
import pandas as pd
import itertools

def util_pattern_space(time_series, lag, dim):
    """Create a set of sequences with given lag and dimension
    Args:
       time_series: Vector or string of the sample data
       lag: Lag between beginning of sequences
       dim: Dimension (number of patterns)
    Returns:
        2D array of vectors
    """
    n = len(time_series)

    if lag * dim > n:
        raise Exception('Result matrix exceeded size limit, try to change lag or dim.')
    elif lag < 1:
        raise Exception('Lag should be greater or equal to 1.')

    pattern_space = np.empty((n - lag * (dim - 1), dim))
    for i in range(n - lag * (dim - 1)):
        for j in range(dim):
            pattern_space[i][j] = time_series[i + j * lag]

    return pattern_space


def util_standardize_signal(time_series):
    return (time_series - np.mean(time_series)) / np.std(time_series)


def util_granulate_time_series(time_series, scale):
    """Extract coarse-grained time series
    Args:
        time_series: Time series
        scale: Scale factor
    Returns:
        Vector of coarse-grained time series with given scale factor
    """
    n = len(time_series)
    b = int(np.fix(n / scale))
    cts = [0] * b
    for i in range(b):
        cts[i] = np.mean(time_series[i * scale: (i + 1) * scale])
    return cts


def shannon_entropy(time_series):
    """Return the Shannon Entropy of the sample data.
    Args:
        time_series: Vector or string of the sample data
    Returns:
        The Shannon Entropy as float value
    """

    # Check if string
    if not isinstance(time_series, str):
        time_series = list(time_series)

    # Create a frequency data
    data_set = list(set(time_series))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))

    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    ent = -ent

    return ent


def sample_entropy(time_series, sample_length, tolerance=None):
    """Calculate and return Sample Entropy of the given time series.
    Distance between two vectors defined as Euclidean distance and can
    be changed in future releases
    Args:
        time_series: Vector or string of the sample data
        sample_length: Number of sequential points of the time series
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))
    Returns:
        Vector containing Sample Entropy (float)
    References:
        [1] http://en.wikipedia.org/wiki/Sample_Entropy
        [2] http://physionet.incor.usp.br/physiotools/sampen/
        [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis
            of biological signals
    """
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    n = len(time_series)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((sample_length, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((sample_length, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = time_series[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(time_series[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) / 2
    B = np.vstack(([N], B[:sample_length - 1]))
    similarity_ratio = A / B
    se = - np.log(similarity_ratio)
    se = np.reshape(se, -1)
    return se


def multiscale_entropy(time_series, sample_length, tolerance):
    """Calculate the Multiscale Entropy of the given time series considering
    different time-scales of the time series.
    Args:
        time_series: Time series for analysis
        sample_length: Bandwidth or group of points
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))
    Returns:
        Vector containing Multiscale Entropy
    Reference:
        [1] http://en.pudn.com/downloads149/sourcecode/math/detail646216_en.html
    """
    n = len(time_series)
    mse = np.zeros((1, sample_length))

    for i in range(sample_length):
        b = int(np.fix(n / (i + 1)))
        temp_ts = [0] * int(b)
        for j in range(b):
            num = sum(time_series[j * (i + 1): (j + 1) * (i + 1)])
            den = i + 1
            temp_ts[j] = float(num) / float(den)
        se = sample_entropy(temp_ts, 1, tolerance)
        mse[0, i] = se

    return mse[0]


def permutation_entropy(time_series, m, delay):
    """Calculate the Permutation Entropy
    Args:
        time_series: Time series for analysis
        m: Order of permutation entropy
        delay: Time delay
    Returns:
        Vector containing Permutation Entropy
    Reference:
        [1] Massimiliano Zanin et al. Permutation Entropy and Its Main Biomedical and Econophysics Applications:
            A Review. http://www.mdpi.com/1099-4300/14/8/1553/pdf
        [2] Christoph Bandt and Bernd Pompe. Permutation entropy — a natural complexity
            measure for time series. http://stubber.math-inf.uni-greifswald.de/pub/full/prep/2001/11.pdf
        [3] http://www.mathworks.com/matlabcentral/fileexchange/37289-permutation-entropy/content/pec.m
    """
    n = len(time_series)
    permutations = np.array(list(itertools.permutations(range(m))))
    c = [0] * len(permutations)

    for i in range(n - delay * (m - 1)):
        # sorted_time_series =    np.sort(time_series[i:i+delay*m:delay], kind='quicksort')
        sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort'))
        for j in range(len(permutations)):
            if abs(permutations[j] - sorted_index_array).any() == 0:
                c[j] += 1

    c = [element for element in c if element != 0]
    p = np.divide(np.array(c), float(sum(c)))
    pe = -sum(p * np.log(p))
    return pe


def multiscale_permutation_entropy(time_series, m, delay, scale):
    """Calculate the Multiscale Permutation Entropy
    Args:
        time_series: Time series for analysis
        m: Order of permutation entropy
        delay: Time delay
        scale: Scale factor
    Returns:
        Vector containing Multiscale Permutation Entropy
    Reference:
        [1] Francesco Carlo Morabito et al. Multivariate Multi-Scale Permutation Entropy for
            Complexity Analysis of Alzheimer’s Disease EEG. www.mdpi.com/1099-4300/14/7/1186
        [2] http://www.mathworks.com/matlabcentral/fileexchange/37288-multiscale-permutation-entropy-mpe/content/MPerm.m
    """
    mspe = []
    for i in range(scale):
        coarse_time_series = util_granulate_time_series(time_series, i + 1)
        pe = permutation_entropy(coarse_time_series, m, delay)
        mspe.append(pe)
    return mspe

def composite_multiscale_entropy(time_series, sample_length, scale, tolerance=None):
    """Calculate the Composite Multiscale Entropy of the given time series.
    Args:
        time_series: Time series for analysis
        sample_length: Number of sequential points of the time series
        scale: Scale factor
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))
    Returns:
        Vector containing Composite Multiscale Entropy
    Reference:
        [1] Wu, Shuen-De, et al. "Time series analysis using
            composite multiscale entropy." Entropy 15.3 (2013): 1069-1084.
    """
    cmse = np.zeros((1, scale))

    for i in range(scale):
        for j in range(i):
            tmp = util_granulate_time_series(time_series[j:], i + 1)
            cmse[i] += sample_entropy(tmp, sample_length, tolerance) / (i + 1)

    return cmse

def CorrelationMatrix(data):
    return np.corrcoef(data)

def Moments(data):  # CHECK HERE
    axis = channel.ndim - 1
    return scipy.stats.moment(channel, moment = [2,3],axis = axis)

def Stats(data):
    #print(shape)
    out = {}

    out[0] = np.std(data) # standard deviation
    out[1] = np.min(data) # min 
    out[2] = np.max(data) # max
    out[3] = np.var(data) # variance
    return(out)


def first_order_diff(X):
    """ Compute the first order difference of a time series.

        For a time series X = [x(1), x(2), ... , x(N)], its first order 
        difference is:
        Y = [x(2) - x(1) , x(3) - x(2), ..., x(N) - x(N-1)]
     """
    D=[]
    for i in xrange(1,len(X)):
        D.append(X[i]-X[i-1])
    return D


def pfd(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two 
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided, 
           recommended to speed up)

    In case 1, D is computed by first_order_diff(X) function of pyeeg

    To speed up, it is recommended to compute D before calling this function 
    because D may also be used by other functions whereas computing it here 
    again will slow down.
    """
    D=first_order_diff(X)

    N_delta= 0; #number of sign changes in derivative of the signal
    for i in xrange(1,len(D)):
        if D[i]*D[i-1]<0:
            N_delta += 1
    n = len(X)
    return log10(n)/(log10(n)+log10(n/n+0.4*N_delta))


def hurst(channel):

    x = np.array(channel)
    x = x-x.mean()
    z = np.cumsum(x)
    r = np.array((np.maximum.accumulate(z) - np.minimum.accumulate(z))[1:])
    s = pd.expanding_std(x)[1:]
    s[np.where(s == 0)] = 1e-12
    r += 1e-12
    y_axis = np.log(r / s)
    x_axis = np.log(np.arange(1, len(y_axis) + 1))
    x_axis = np.vstack([x_axis, np.ones(len(x_axis))]).T
    m, b = np.linalg.lstsq(x_axis, y_axis)[0]
    
    return(m)

def hjorthFD(X, Kmax=3):
    """ Compute Hjorth Fractal Dimension of a time series X, kmax
     is an HFD parameter. Kmax is basically the scale size or time offset.
     So you are going to create Kmax versions of your time series.
     The K-th series is every K-th time of the original series.
     This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """
    L = []
    x = []
    N = len(X)
    for k in range(1,Kmax):
        Lk = []
        
        for m in range(k):
            Lmk = 0
            for i in range(1,int(floor((N-m)/k))):
                Lmk += np.abs(X[m+i*k] - X[m+i*k-k])
                
            Lmk = Lmk*(N - 1)/floor((N - m) / k) / k
            Lk.append(Lmk)
            
        L.append(np.log(np.nanmean(Lk)))   # Using the mean value in this window to compare similarity to other windows
        x.append([np.log(float(1) / k), 1])

    (p, r1, r2, s)= np.linalg.lstsq(x, L)  # Numpy least squares solution
    
    return p[0]


def petrosianFD(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two 
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided, 
           recommended to speed up)

    In case 1, D is computed by first_order_diff(X) function of pyeeg

    To speed up, it is recommended to compute D before calling this function 
    because D may also be used by other functions whereas computing it here 
    again will slow down.
    
    This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """
    
    # If D has been previously calculated, then it can be passed in here
    #  otherwise, calculate it.
    if D is None:   ## Xin Liu
        D = np.diff(X)   # Difference between one data point and the next
        
    # The old code is a little easier to follow
    N_delta= 0; #number of sign changes in derivative of the signal
    for i in range(1,len(D)):
        if D[i]*D[i-1]<0:
            N_delta += 1

    n = len(X)
    
    return np.log10(n)/(np.log10(n)+np.log10(n/n+0.4*N_delta))


def katzFD(epoch):
    ''' 
    Katz fractal dimension 
    '''
    
    L = np.abs(epoch - epoch[0]).max()
    d = len(epoch)
    
    return (np.log(L)/np.log(d))


def hurstFD(epoch):

    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.nanstd(np.subtract(epoch[lag:], epoch[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


def higuchiFD(epoch, Kmax = 8):
    '''
    Ported from https://www.mathworks.com/matlabcentral/fileexchange/30119-complete-higuchi-fractal-dimension-algorithm/content/hfd.m
    by Salai Selvam V
    '''
    
    N = len(epoch)
    
    Lmk = np.zeros((Kmax,Kmax))
    
    #TODO: I think we can use the Katz code to refactor resampling the series
    for k in range(1, Kmax+1):
        
        for m in range(1, k+1):
               
            Lmki = 0
            
            maxI = floor((N-m)/k)
            
            for i in range(1,int(maxI+1)):
                Lmki = Lmki + np.abs(epoch[m+i*k-1]-epoch[m+(i-1)*k-1])
             
            normFactor = (N-1)/(maxI*k)
            Lmk[m-1,k-1] = normFactor * Lmki
    
    Lk = np.zeros((Kmax, 1))
    
    #TODO: This is just a mean. Let's use np.mean instead?
    for k in range(1, Kmax+1):
        Lk[k-1,0] = np.nansum(Lmk[range(k),k-1])/k/k

    lnLk = np.log(Lk) 
    lnk = np.log(np.divide(1., range(1, Kmax+1)))
    
    fit = np.polyfit(lnk,lnLk,1)  # Fit a line to the curve
    val= fit[0]
    return  val  # Grab the slope. It is the Higuchi FD




def power_spectral_density(channel):

    power_spectral = list()

    channel = map(float, channel)
    f, Pxx_den = signal.periodogram(list(channel))
    Pxx_den = list(set(Pxx_den))
    Pxx_den.sort()
    power_spectral.append(Pxx_den[len(Pxx_den)-5:len(Pxx_den)])
    return(power_spectral)
filenum=0

    #List of lists
with open('/home/lab3/Documents/HB/ESP_Data/Dog5_train_sample_feature_ext.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(["stats1","stats2","stats3","stats4","hurst","pfd","hjorth","katz","higuchi","shannon_entropy","skewness","kurtosis","psd1","psd2","psd3","psd4","psd5","ev1","ev2","ev3","ev4","ev5","ev6","ev7","ev8","ev9","ev10","ev11","ev12","ev13","ev14","ev15","Class"])
    
    for filename in os.listdir("/home/lab3/Documents/HB/ESP_Data/Dog_5/"):
        filenum = filenum + 1
        print "\n\n\nFile:", filenum
        print "\n\n\nFile:", filename
        element_writer=list()
        if filename.endswith(".mat"):
            fwr=list()
            feature_writer=list()
            file_read = si.loadmat('/home/lab3/Documents/HB/ESP_Data/Dog_5/'+filename)
            allictal=list(file_read.keys())
            print "All Ictal Values list(file_read.keys()) : ",allictal
            if 'segment' in allictal[0]:
                ictal = list(file_read.keys())[0]
            elif 'segment' in allictal[1]:
                ictal = list(file_read.keys())[1]
            elif 'segment' in allictal[2]:
                ictal = list(file_read.keys())[2]
            elif 'segment' in allictal[3]:
                ictal = list(file_read.keys())[3]
            else:
                ictal = 'None'
        
        

            all_channels = file_read[ictal][0][0][0]
            eigvals,v = np.linalg.eig(CorrelationMatrix(all_channels))
            eigvals = np.abs(eigvals)
        
            if ictal != 'None' and 'test' not in ictal:
                channelnum=0
                for channel in all_channels:
                
                    print "Channel Number: ", channelnum
                    print "Ictal Value:", ictal
                    channelnum = channelnum + 1
                    feature_writer=list()
                    stats_channel=Stats(channel)#Four feature cols
                    for i in range(len(stats_channel)):
                        feature_writer.append(stats_channel[i])
        

                    '''
The values of the Hurst exponent vary between 0 and 1. A Hurst exponent value of H = 0.5
indicates a random walk process (a Brownian motion). In a random walk, there is no correlation,
this is uncorrelated time series. If 0 ≤ H ≤ 0.5, the process is said to be antipersistent. The
system is covering smaller distances than a random walk process. This means that an increase
will tend to be followed by decrease (or decrease will be followed by an increase). This behavior
is sometimes called mean reversion. For 0.5 < H ≤ 1, the time series belongs to a persistent
process. This series covers more distance than a random walk process. Thus, if the system
increases in one period, it is more likely to keep increasing in the next period.
'''
                    hurst_channel=hurstFD(channel)              #One feature col
                    feature_writer.append(hurst_channel)    


                    pfd_channel=pfd(channel)                    #One feature col
                    feature_writer.append(pfd_channel)


                    hjorth_channel=hjorthFD(channel)            #One feature col
                    feature_writer.append(hjorth_channel)





                    katz_channel=katzFD(channel)                #One feature col
                    feature_writer.append(katz_channel)



                    higuchi_channel=higuchiFD(channel)#One feature col
                    for val in higuchi_channel:    
                        feature_writer.append(val)



                
                    shannon_channel=shannon_entropy(channel)    #One feature col
                    feature_writer.append(shannon_channel)

                    moments_channel=Moments(channel)
                    for val in moments_channel:
                        feature_writer.append(val)
                    


                    psd_channel=power_spectral_density(channel)
                    
                    for vals in psd_channel:
                        for val in vals:
                            feature_writer.append(val)


                    for val in eigvals:
                        feature_writer.append(val)#Assigning class on the basis of file name

                    #Assigning class on the basis of file name


                    
                    if 'interictal' in ictal:
                        ictal_class=0
                    else:
                        ictal_class=1
                    feature_writer.append(ictal_class)



                    writer.writerow(feature_writer)
                    
                    #fwr.append(feature_writer)



