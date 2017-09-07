import scipy.io as si
import numpy as np 
import pandas as pd
import scipy as sc
from scipy import signal
from scipy import stats
from numpy import zeros, floor, log10, log, mean, array, sqrt, vstack, cumsum, ones, log2, std, var

def Magnitude(data):
	return np.abs(data)

def FFT(data): # add unique values top 5
	axis = data.ndim - 1
	data = np.fft.rfft(data)
	return(data)


def Moments(data):
	axis = data.ndim - 1
	return scipy.stats.moment(data, moment = 2,axis = axis)

def CorrelationMatrix(data):
	return np.corrcoef(data)

def Stats(data):
	shape = data.shape
	print(shape)
	out = np.empty((shape[0], 4))
	for i in range(len(data)):

		out[i][0] = np.std(data[i]) # standard deviation
		out[i][1] = np.min(data[i]) # min 
		out[i][2] = np.max(data[i]) # max
		out[i][3] = np.var(data[i]) # variance
	return(out)

def FlattenChannels(data):
	if data.ndim == 2:
		return data.ravel()
	elif data.ndim == 3:
		s = data.shape
		return data.reshape((s[0], np.product(s[1:])))
	else:
		raise NotImplementedError()

def log(data):
	indices = np.where(data <= 0)
	data[indices] = np.max(data)
	data[indices] = (np.min(data) * 0.1)
	return np.log(data)


def Hurst(single_channel):
	
	x = np.array(single_channel)
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
	return m

def first_order_diff(X):

	D=[]
	for i in xrange(1,len(X)):
		D.append(X[i]-X[i-1])
	return D

def hjorth(clip):
	
	hjorth_clip = list()

	for channels in clip:
		l = []
		D = first_order_diff(channels)
		D.insert(0, channels[0])
		D = np.array(D)
		n = len(channels)
		M2 = float(sum(D ** 2)) / n
		TP = sum(np.array(channels) ** 2)
		M4 = 0;
		for i in xrange(1, len(D)):
			M4 += (D[i] - D[i - 1]) ** 2
		M4 = M4 / n
		l.append(np.array(sqrt(M2 / TP)))  # Hjorth Mobility  
		l.append(sqrt(float(M4) * TP / M2 / M2)) # Hjorth Complexity
		hjorth_clip.append(l)

	return(hjorth_clip)


def bin_power(X,Band,Fs):

	C = FFT(X)
	C = Magnitude(C)
	Power =zeros(len(Band));
	for Freq_Index in xrange(0,len(Band)-1):
		Freq = Band[Freq_Index]
		Next_Freq = Band[Freq_Index+1]
		Power[Freq_Index] = sum(C[floor(Freq/Fs*len(X)):floor(Next_Freq/Fs*len(X))])
	Power_Ratio = Power/sum(Power)
	return Power, Power_Ratio	



def spectral_entropy(X, Band, Fs, Power_Ratio = None):

	Power, Power_Ratio = bin_power(X, Band, Fs)

	Spectral_Entropy = 0
	for i in xrange(0, len(Power_Ratio) - 1):
		Spectral_Entropy += Power_Ratio[i] * log(Power_Ratio[i])
	Spectral_Entropy /= log(len(Power_Ratio))	# to save time, minus one is omitted
	return -1 * Spectral_Entropy


def power_spectral_density(clip):

	power_spectral = list()
	for channel in clip:
		channel = map(float, channel)
		f, Pxx_den = signal.periodogram(list(channel))
		Pxx_den = list(set(Pxx_den))
		Pxx_den.sort()
		power_spectral.append(Pxx_den[len(Pxx_den)-5:len(Pxx_den)])
	return(power_spectral)

def cross_spectral_density(clip):
	# we need to find relation with every one else
	spectral_density = []
	for i in range(0,len(clip)):
		x = clip[i]
		pair = list()
		channel = []
		for j in range(0,len(clip)):
			if j is not i:
				channel = []
				y = clip[j]
				x = map(float,x)
				y = map(float,y)
				f, Pxy = signal.csd(x, y)		
				channel.append(Magnitude(Pxy))
		
		spectral_density.append(channel)
	
	return(spectral_density)



file_read = si.loadmat("/Users/utkarsh/Documents/RESEARCH/epilepsy/Dataset/Dog_5_interictal_segment_0001.mat")
clip = file_read['interictal_segment_1'][0][0][0]
#print(cross_spectral_density(clip))


