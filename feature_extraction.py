import scipy.io as si
import numpy as np 
import pandas as pd
import scipy as sc
import scipy.stats
import pygee.py

def FFT(data):
	axis = all_channels.ndim - 1
	return (np.fft.rfft(all_channels))

def Moments(data):
	axis = all_channels.ndim - 1
	return scipy.stats.moment(all_channels, moment = 2,axis = axis)

def CorrelationMatrix(data):
	return np.corrcoef(data)

def Magnitude(self, data, meta=None):
	return np.abs(data)

def Stats(data):
	shape = data.shape
	print(shape)
	out = np.empty((shape[0], 3))
	for i in range(len(data)):

		out[i][0] = np.std(data[i]) # standard deviation
		out[i][1] = np.min(data[i])
		out[i][2] = np.max(data[i])
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








file_read = si.loadmat("/Users/utkarsh/Documents/RESEARCH/epilepsy/Dataset/Dog_5_interictal_segment_0001.mat")
all_channels = file_read['interictal_segment_1'][0][0][0]
print(pfd(all_channels[0]))


