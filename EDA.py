# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 01:25:08 2019

@author: srjcp
"""
import os
import pandas as pd
import librosa 
import seaborn as sns
import utils
import numpy as np 
import matplotlib.pyplot as plt
import librosa.display
from scipy.fftpack import fft
import warnings
warnings.filterwarnings(action='ignore')

#loading csv metadata files 
tracks=utils.load('tracks.csv')
genres = utils.load('genres.csv')
#New dataframe for medium dataset
medium_subset=tracks[tracks[('set','subset')]!='large' ]

# Artist Information
print('{} artists, {} albums'.format(
    len(medium_subset['artist', 'id'].unique()),
    len(medium_subset['album', 'id'].unique())))
###Counting tracks per artist
artist=pd.DataFrame(index=medium_subset[('artist', 'id')].unique(), columns=['n'])
artist['n']=medium_subset[('artist', 'id')].value_counts()
for row in artist.index:
    b = medium_subset['artist', 'id'] == row
    artist.set_value(row,'#genre',len(medium_subset.loc[ b, ('track', 'genre_top')].unique()))
len(medium_subset[('artist', 'id')].unique())
artist['#genre'].value_counts()               
sns.distplot(artist[artist.values > 50],kde=False, rug=False, color='k', hist_kws=dict(alpha=0.4))
plt.figure()
artist['n'].sort_values(ascending=False).plot.bar(color='k', alpha=0.4)
plt.ylabel('#artist')
plt.tight_layout()

#Data split 
SPLITS = ['training', 'validation', 'test']
counts = [sum((medium_subset['set', 'split'] == split)) for split in SPLITS]
ratios = np.array(counts[0] / counts[1:])
print('#train    #val   #test  val_ratio test_ratio')
print('{:7d} {:7d} {:7d} {:8.2f} {:9.2f}'.format(*counts, *ratios))

#Data distribution according to genre 
d = genres.reset_index().set_index('title')
d = d.loc[medium_subset[('track','genre_top')].unique()]
d['#tracks']=medium_subset['track', 'genre_top'].value_counts()
for split in SPLITS:
    b = medium_subset['set', 'split'] == split
    d['#' + split] = medium_subset.loc[ b, ('track', 'genre_top')].value_counts()     
d['val_ratio'] = d['#training'] / d['#validation']
d['test_ratio'] = d['#training'] / d['#test']
 
#average number of artist per genre
for column in d.index:
    b = medium_subset['track', 'genre_top'] == column
    d.set_value(column,'#artist',len(medium_subset.loc[ b, ('artist', 'id')].unique()))
d['ave_artist']=d['#tracks']/d['#artist']
 
d.sort_values(by=['#tracks'], ascending=False)

plt.figure()
d['#tracks'].sort_values(ascending=False).plot.bar(color='k', alpha=0.4)
plt.ylabel('#tracks')
plt.tight_layout()

###Technical Data
durations = medium_subset['track', 'duration']
plt.figure(figsize=(10, 4))
p = sns.distplot(durations, kde=False, rug=False, color='k', hist_kws=dict(alpha=0.4))
p.set_xlabel('duration [seconds]')
p.set_ylabel('#tracks')
p.set_xlim(0, 650)
plt.tight_layout()
#plt.show()
#plt.savefig('duration_distribution.pdf')
#durations.describe()

#Bitrate
print('Common bit rates: {}'.format(medium_subset['track', 'bit_rate'].value_counts().head(5).index.tolist()))
print('Average bit rate: {:.0f} kbit/s'.format(medium_subset['track', 'bit_rate'].mean()/1000))
p = sns.distplot(medium_subset['track', 'bit_rate'], kde=False, rug=False)
p.set_xlabel('bit rate')
p.set_ylabel('#tracks');
plt.show()

#User Data
def plot(col0, col1, maxval, subplot=None):
    if col0 == 'track':
        d = medium_subset['track']
    if col0 in ['artist', 'album']:
        d = medium_subset[col0].drop_duplicates('id')
    if subplot:
        plt.subplot(subplot)
    d = d[col1]
    p = sns.distplot(d[d.values < maxval], kde=False, color='k', hist_kws=dict(alpha=0.4))
    p.set_xlim(-1, maxval)
    p.set_xlabel('#' + col1)
    p.set_ylabel('#' + col0 + 's')
    plt.show()

medium_subset['track'].describe()
plt.figure(figsize=(17, 10))
plot('track', 'listens', 10e3, 221)
plot('track', 'interest', 10e3, 222)
plot('track', 'favorites', 100, 223)
plot('track', 'comments', 20, 224)

medium_subset['album'].describe()
plt.figure(figsize=(17, 10))
plot('album', 'listens', 100e3, 221)
plot('album', 'favorites', 100, 223)
plot('album', 'comments', 20, 224)

medium_subset['artist'].describe()
plt.figure(figsize=(17, 5))
plot('artist', 'favorites', 100, 121)
plot('artist', 'comments', 20, 122)

#Checking audio files
AUDIO_DIR = 'C:/MusicClassification/fma_medium'
filename = utils.get_audio_path(AUDIO_DIR,2)
#reading audio file with librosa
x, sr = librosa.load(filename, sr=None, mono=True)
print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

#FFT plot
def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    return xf, vals

xf, vals = custom_fft(x, sr)
plt.figure(figsize=(12, 4))
plt.title('FFT of recording sampled with ' + str(sr) + ' Hz')
plt.plot(xf, vals)
plt.xlabel('Frequency')

#Short time fourier transform
n_fft=2048
hop_length=1024
stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length,window='hamm'))
log_stft = librosa.amplitude_to_db(stft)

librosa.display.specshow(log_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear');
plt.colorbar(format='%+2.0f dB')
plt.title('Power spectrogram')
plt.tight_layout()

librosa.display.specshow(log_stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f dB')
plt.title('Power spectrogram')
plt.tight_layout()