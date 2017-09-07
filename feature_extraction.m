clc
load('/Users/utkarsh/Documents/RESEARCH/epilepsy/Dataset/Dog_5/Dog_5_interictal_segment_0001.mat');
main_array = interictal_segment_1.data;
frequency = interictal_segment_1.sampling_frequency;
channel_one = main_array(1,:);
n = length(channel_one);

% Time series to frequency 
FT = real(fft(channel_one));

% Shannon's entropy
Entropy = wentropy(channel_one,'shannon');

% Spectral edge frequency and power
FT = fftshift(FT);
Spectral_Edge_Power = abs(FT).^2/n;
PSDChannel1=FT.*conj(FT);

% Spectral correlation
Spectral_corr = xcorr(channel_one,main_array(2,:));

% Time series correlation matrix and its eigen vectors
Time_Series = corrcoef([channel_one',main_array(2,:)',main_array(3,:)',main_array(4,:)']);

% Spectral correlation density
[c,lags] =xcorr(channel_one);
SpDensity=fft(c);
SpDensity=real(SpDensity);

% Cross spectral density 
CPSD = cpsd(channel_one,main_array(2,:));

% Power spectral density
Power_Spec_Dens = periodogram(channel_one,rectwin(length(channel_one)),length(channel_one),800);

% Hjorth parameters
dxV = diff([0;channel_one']);
ddxV = diff([0;dxV]);
mx2 = mean(channel_one'.^2);
mdx2 = mean(dxV.^2);
mddx2 = mean(ddxV.^2);
mob = mdx2 / mx2;
complexity = sqrt(mddx2 / mdx2 - mob); 
mobility = sqrt(mob);

% Statistical Moments
mean(channel_one);
var(channel_one);
skewness(channel_one);
kurtosis(channel_one);

% Time Series Varience
TS_var = var(channel_one);

% Fractal Dimentions 
% You need to take care of this !!!!!