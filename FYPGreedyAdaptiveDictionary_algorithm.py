#!/usr/bin/env python
# coding: utf-8

# In[154]:

# importing libraries 
from __future__ import division
from numpy import linalg as LA
import numpy as np
np.set_printoptions(threshold=np.inf)
import sys
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy.io.wavfile import write
import sounddevice as sd
# to play the audio back
from numpy import array, linspace
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import numpy as np


# In[155]:


# source: https://github.com/DavideNardone/Greedy-Adaptive-Dictionary
# paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5776648


# In[195]:


# the function to separate the time series signal into frames
# inputs: signal = audio, L = number of samples in each frame, M = number of overlapping samples between frames
# output = X_tmp = segmented signal

def buffer(signal, L, M):
    if M >= L:
        print('Error: Overlapping windows cannot be larger than frame length!')
        sys.exit()

    len_signal = len(signal)

    print('The signal length is %s: ' % (len_signal))

    K = np.ceil(len_signal / L).astype('int')  # num_frames

    # sense checking
    print('The number of frames \'K\' is %s: ' % (K))
    print('The length of each frame \'L\' is %s: ' % (L))
    frame_dur = L/fs
    print('The duration of each frame is %s: ' %(frame_dur))
    signal_dur = len(signal) /fs
    print('The duration the signal is %s: ' %(signal_dur))

    X_tmp = []
    mag_X_tmp = [] 
    frame_amp = []
    frame_var = []
    
    k = 1
    # a loop to separate the signal into frames and append to X_tmp
    while (True):
        start_ind = ((k - 1) * (L - M) + 1) - 1
        end_ind = ((k * L) - (k - 1) * M)
        if start_ind == len_signal:
            break
        elif (end_ind > len_signal): # if next frame segment would end beyond the length of the input signal
            # print ('k=%s, [%s, %s] ' % (k, start_ind, end_ind - 1))
            val_in = len_signal - start_ind 
            tmp_seg = np.zeros(L)
            tmp_seg[:val_in] = signal[start_ind:]
            
            X_tmp.append(tmp_seg) # the region in which the last frame oveflows is filled with 0 values
            
            # calculate the average energy of the frame
            frame_amp_k = np.average(np.abs(tmp_seg))
            frame_amp.append(frame_amp_k)

            break # because this is the last frame in the audio
        else:
            # print('k=%s, [%s, %s] ' % (k, start_ind, end_ind - 1))
            X_tmp.append(signal[start_ind:end_ind])
            
            # calculate the average energy of the frame (again)
            frame_amp_k = np.average(np.abs(signal[start_ind:end_ind]))
            frame_amp.append(frame_amp_k)
        
        k += 1
        
    
    return X_tmp, mag_X_tmp, K, frame_amp, frame_var  # GAD algorithm will take this as an input


# complimentary function to convert the segmented signal back into a time series
# inputs: X = signal reconstructed from the dictionary and sparse code from the GAD algorithm, and hop size
def unbuffer(X, hop):
    N, L = X.shape

    T = N + L * hop
    K = np.arange(0, N)
    x = np.zeros(T)
    H = np.hanning(N)
    for k in range(0, L):
        x[K] = x[K] + np.multiply(H, X[:, k])
        K = K + hop

    return x


# In[201]:


# contains the functions for the GAD algorithm
# 1) findResidualColumn: uses the sparsity index to determine which atoms to extract


# 2) interative_GAD: continuously calls findResidualColumn to update the dictionary
# returns: learned dictionary = D, set_ind = set of indices where we extracted the atoms from?

class GAD():
    def __init__(self, X, params, priors, sparse_score, sound_score):

        self.X = X
        self.D = []
        self.params = params
        self.n_iter = self.params['rule_1']['n_iter']  # num_iterations
        self.verbose = self.params['verbose']
        
        self.priors = priors
        self.p_silence = self.priors['prior_1']['p_silence'] # expected % silence
        self.verbose = self.priors['verbose']

        self.K = self.X.shape[0]  # sample length
        self.L = self.X.shape[1]  # maximum atoms to be learned

        self.I = np.arange(0, self.L)
        self.original_I = self.I
        self.set_ind = []
        
        self.sparse_score = sparse_score
        self.sound_score = sound_score
        
        
    # function to determine which atoms we will extract from the segmented signal X_tmp (from buffer())
    def findResidualColumn(self):
        
        # Find residual column of R^l with lowest l1- to l2-norm ration - AKA "sparsity index" - as referenced to in paper
        tmp = [] # for reducing I
        TMP = [] # for non-reducing I
        score1 = 0
        score2 = 0
        score3 = 0
        score4 = 0
        

        # create a list of sparsity index values for each atom and append to list "tmp"
        for k in self.I:
            r_k = self.R[:, k]
            tmp.append(LA.norm(r_k, 1) / LA.norm(r_k, 2)) # append tmp with the sparsity values
            
        ind_k_min0 = tmp.index(sorted(tmp)[0]) # smallest
        ind_k_min1 = tmp.index(sorted(tmp)[1]) # 2nd smallest
        ind_k_min2 = tmp.index(sorted(tmp)[2]) # 3rd smallest
        ind_k_min3 = tmp.index(sorted(tmp)[3]) # 4th smallest
        ind_k_min4 = tmp.index(sorted(tmp)[4]) # 5th smallest
        
        ind_k_mins = [ind_k_min0, ind_k_min1, ind_k_min2, ind_k_min3, ind_k_min4]
        scores = [sparse_score, score1, score2, score3, score4]
        
        for l in self.original_I:    
            R_k = self.R[:, l]
            TMP.append(LA.norm(R_k, 1) / LA.norm(R_k, 2)) # append tmp with the sparsity values
 
        # where does the newest smallest sparsity value match a sparsity value in TMP?
        pos0 = TMP.index(sorted(tmp)[0]) 
        pos1 = TMP.index(sorted(tmp)[1]) 
        pos2 = TMP.index(sorted(tmp)[2])
        pos3 = TMP.index(sorted(tmp)[3])
        pos4 = TMP.index(sorted(tmp)[4])
        poss = [pos0, pos1, pos2, pos3, pos4]
        
        # now return volumes
        vol0 = frame_amp[pos0]
        vol1 = frame_amp[pos1]
        vol2 = frame_amp[pos2]
        vol3 = frame_amp[pos3]
        vol4 = frame_amp[pos4]
        
        vols = [vol0,vol1,vol2,vol3,vol4]
        ind_loudest = vols.index(sorted(vols)[len(vols)-1]) # find loudest
        scores[ind_loudest] += sound_score # bigger score for loudest segment
        ind_top_score = scores.index(sorted(scores)[len(scores)-1])
        
        ind_k_min = ind_k_mins[ind_top_score]
        
        r_k_min = self.R[:, self.I[ind_k_min]] # returns the column with the lowest sparsity value


        # Set the l-th atom to equal to normalized r_k
        psi = r_k_min / LA.norm(r_k_min, 2)

        # Add to the dictionary D and its index and shrinking set I
        self.D.append(psi) # populate the dictionary
        self.set_ind.append(self.I[ind_k_min])

        # Compute the new residual for all columns k
        for k in self.I:
            r_k = self.R[:, k]
            alpha = np.dot(r_k, psi)
            self.R[:, k] = r_k - np.dot(psi, alpha)

        # suspect this step is where the contribution of the atom added on this iteration is
        # removed before the process is repeated    
        self.I = np.delete(self.I, ind_k_min)
        #print(self.I)

    def iterative_GAD(self):

        if self.n_iter > self.L:
            print ('Cannot be learned more than %d atom!' % (self.L))
            sys.exit()

        # Initializating the residual matrix 'R' by using 'X'
        self.R = self.X.copy()

        print("shape of I is the maximum number of atoms to be learned:")
        print(self.I.shape)
        for l in range(0, self.n_iter):

            if self.verbose == True:
                #print('GAD iteration: ', l + 1)

                self.findResidualColumn()

        self.D = np.vstack(self.D).T

        #print("Remaining indices:")
        #print(self.I) # shape of I after GAD iterations have run
        remaining_ind = self.I
        return self.D, self.set_ind, remaining_ind, self.n_iter


# To play new signal
def playAudio(s_rec):
    global sound
    sound = ipd.Audio(s_rec, rate = fs, autoplay = True)
    return sound

def whereAudio(s_rec):
    sound = playAudio(s_rec)
    return sound

# In[218]:


# paramters used for the GAD algorithm: 
# n_iter = total number of iterations, 
# error = algorithm will stop when reconstruction error is small enough

n_iter1 = 5

params = {

        'rule_1': {
            'n_iter':  n_iter1
        },

        'rule_2': {
            'error': 10 ** -7
        },

        'verbose': True
    }

# priors used for the GAD algorithm: 

priors = {

        'prior_1': {
            'p_silence': 0.5  # n_iter
        },

        'verbose': True
    }

# In[220]:


# TESTING WHEN NUMBER OF ATOMS TO BE EXTRACTED < NUMBER OF FRAMES GENERATED - SUMMARY STYLE

path = "C:\\Users\\ellie\\Documents\\Sound recordings\\longer_test_with_silence.wav"

signal, fs = librosa.core.load(path)

# still determining which values and ratios give best results
# was L = 5120 and M = 400/500
L = 3000  # frame length
M = 500  # overlapping windows

# SCORES (1)
sparse_score = 5
sound_score1 = 0
sound_score = sound_score1

#### now each function is called one at a time in order ####
print("RESET")
X_tmp, mag_X_tmp, K, frame_amp, frame_var = buffer(signal, L, M)

# splits signal into k overlapping frames
# each frame has L samples
# each frame has an overlap of M samples

# new matrix LxK
X = np.vstack(X_tmp).T.astype('float')
# run the Greedy Adaptive Dictionary algorithm
alg = GAD(X, params, priors, sparse_score, sound_score)
D, I, remaining_ind, n_iter = alg.iterative_GAD()
# audio frames returned from the GAD algorithm
X_t = np.dot(np.dot(D, D.T), X)
# converted the returned audio frames back into signal format
s_rec = unbuffer(X_t, L - M)
# write and play new sound
write('output.wav', fs, s_rec)
sd.play(s_rec,fs)

# In[221]: Plotting Figures


l_f_a = len(frame_amp)
print("Number of frames #= %d" %l_f_a)
print("Fraction of signal returned:")
print(n_iter/len(frame_amp))

from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

fig = plt.figure(figsize=(20,5))
fig.suptitle("Investigation 1 Frame-Amplitude Profiles: Varying the Number of GAD Algorithm Iterations", fontsize = 20,y=1.05)

ax = plt.subplot(1,2,1)


bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->")

ax.annotate('4$^{th}$ extracted \n frame', xy=(0.86, 0.08), xycoords='axes fraction',
             xytext=(50, 30), textcoords='offset points',
             ha="right", va="bottom",
             bbox=bbox_args,
             arrowprops=arrow_args)
ax.annotate('ignored frame', xy=(0.465, 0.715), xycoords='axes fraction',
             xytext=(20, 40), textcoords='offset points',
             ha="right", va="bottom",
             bbox=bbox_args,
             arrowprops=arrow_args)

ax.title.set_text("Frame-Amplitude Extraction Annotation (Parameter Set 1)")
ax.plot([i for i in range(len(frame_amp))],frame_amp, color = 'black')
for i in range(0,n_iter): # not all frames will make the blue annotation cut (be an atom)
    ax.annotate(str(i),(I[i],frame_amp[I[i]]),color='blue',label='extracted frames')
for j in range(0,len(remaining_ind)):
    ax.annotate("X",(remaining_ind[j],frame_amp[remaining_ind[j]]),color='red',label='ignored frames')


at = AnchoredText("Parameters: \n n_iter = %d \n sound_score = %d \n sparse_score = %d"%(n_iter1,sound_score1,sparse_score),
                  prop=dict(size=8), frameon=True,
                  loc=2,
                  )
ax.add_artist(at)

plt.legend(['Original signal'],loc=1)
plt.xlabel("Frame number")
plt.ylabel("Average frame amplitude")

# In[218]:


# paramters used for the GAD algorithm: 
# n_iter = total number of iterations, 
# error = algorithm will stop when reconstruction error is small enough

n_iter2 = 15

params = {

        'rule_1': {
            'n_iter': n_iter2 
        },

        'rule_2': {
            'error': 10 ** -7
        },

        'verbose': True
    }

# priors used for the GAD algorithm: 

priors = {

        'prior_1': {
            'p_silence': 0.5  # n_iter
        },

        'verbose': True
    }

# In[220]:


# TESTING WHEN NUMBER OF ATOMS TO BE EXTRACTED < NUMBER OF FRAMES GENERATED - SUMMARY STYLE

path = "C:\\Users\\ellie\\Documents\\Sound recordings\\longer_test_with_silence.wav"
#path = "C:\\Users\\ellie\\Documents\\Sound recordings\\withPauses.m4a"
# \
signal, fs = librosa.core.load(path)

# still determining which values and ratios give best results
# was L = 5120 and M = 400/500
L = 3000  # frame length
M = 500  # overlapping windows

# SCORES
sparse_score = 5
sound_score2 = 0
sound_score = sound_score2

#### now each function is called one at a time in order ####
print("RESET")
X_tmp, mag_X_tmp, K, frame_amp, frame_var = buffer(signal, L, M)
print("K is (overlapping frames):")
print(K)
print("length of frame_amp is:")
print(len(frame_amp))


# splits signal into k overlapping frames
# each frame has L samples
# each frame has an overlap of M samples

# new matrix LxK
X = np.vstack(X_tmp).T.astype('float')

alg = GAD(X, params, priors, sparse_score, sound_score)

D, I, remaining_ind, n_iter = alg.iterative_GAD()
print("shape of I")
print(len(I))

# audio frames returned from the GAD algorithm
X_t = np.dot(np.dot(D, D.T), X)

# converted the returned audio frames back into signal format
s_rec2 = unbuffer(X_t, L - M)


ax2 = plt.subplot(1,2,2)
ax2.plot([i for i in range(len(frame_amp))],frame_amp, color = 'black')
ax2.title.set_text("Frame-Amplitude Extraction Annotation (Parameter Set 2)")
for i in range(0,n_iter): # not all frames will make the blue annotation cut (be an atom)
    ax2.annotate(str(i),(I[i],frame_amp[I[i]]),color='green',label='extracted frames')
for j in range(0,len(remaining_ind)):
    ax2.annotate("X",(remaining_ind[j],frame_amp[remaining_ind[j]]),color='red',label='ignored frames')

# at2 = AnchoredText("Parameters: n_iter: %d, L = %d, M = %d,\n loud_score = %d, sparse_score = %d"%(n_iter,L,M,loud_score,sparse_score),
#                   prop=dict(size=8), frameon=True,
#                   loc=2,
#                   )
at2 = AnchoredText("Parameters: \n n_iter = %d \n sound_score = %d \n sparse_score = %d"%(n_iter2,sound_score2,sparse_score),
                  prop=dict(size=8), frameon=True,
                  loc=2,
                  )
at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at2)

ax2.annotate('4$^{th}$ extracted \n frame', xy=(0.86, 0.08), xycoords='axes fraction',
             xytext=(50, 30), textcoords='offset points',
             ha="right", va="bottom",
             bbox=bbox_args,
             arrowprops=arrow_args)
ax2.annotate('ignored frame', xy=(0.465, 0.715), xycoords='axes fraction',
             xytext=(20, 40), textcoords='offset points',
             ha="right", va="bottom",
             bbox=bbox_args,
             arrowprops=arrow_args)

plt.xlabel("Frame number")
plt.ylabel("Average frame amplitude")
plt.legend(['Original signal'],loc=1)
plt.show()

l_f_a = len(frame_amp)
print("Number of frames #= %d" %l_f_a)
print("Fraction of signal returned:")
print(n_iter/len(frame_amp))


# In[99]: Waveplots
          
y_lim = [-0.2,0.2]
## oringal (black) signal
fig2 = plt.figure(figsize=(20,5))


fig2.suptitle("Investigation 1 Waveplot Profiles: Varying the Number of GAD Algorithm Iterations", fontsize = 20, y=1.05)
ax3 = plt.subplot(1,3,1)
ax3.plot(signal,color='black')
ax3.title.set_text("Waveplot of Original Signal")
ax3.set_xlabel("Sample")
ax3.set_ylim(y_lim)
at3 = AnchoredText("Parameters: \n L = %d \n M = %d"%(L,M),
                  prop=dict(size=8), frameon=True,
                  loc=2,                  )
at3.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at3)
plt.legend(['Original signal'],loc=1)


## investigation .1 reconstructed signal waveplot
ax4 = plt.subplot(1,3,2)
ax4.plot(signal,color='lightgrey')
ax4.plot(s_rec,color='blue')
ax4.title.set_text("Waveplot of Reconstructed Signal (Parameter Set 1)")
ax4.set_xlabel("Sample") 
ax4.set_ylim(y_lim)                
at4 = AnchoredText("Parameters: \n n_iter = %d \n sound_score = %d \n sparse_score = %d"%(n_iter1,sound_score1,sparse_score),
                  prop=dict(size=8), frameon=True,
                  loc=2,
                  )
at4.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax4.add_artist(at4)
plt.legend(['Original signal', 'Reconstructed \n signal'],loc=1)

ax5= plt.subplot(1,3,3)
ax5.plot(signal,color='lightgrey')
ax5.plot(s_rec2,color='green')
ax5.title.set_text("Waveplot of Reconstructed Signal (Parameter Set 2)")
ax5.set_xlabel("Sample") 
ax5.set_ylim(y_lim)                
at5 = AnchoredText("Parameters: \n n_iter = %d \n sound_score = %d \n sparse_score = %d"%(n_iter2,sound_score2,sparse_score),
                  prop=dict(size=8), frameon=True,
                  loc=2,
                  )
at5.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax5.add_artist(at5)

plt.legend(['Original signal', 'Reconstructed \n signal'],loc=1)
plt.show()


f_amp = array(frame_amp).reshape(-1, 1)
kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(f_amp)
frame = linspace(0,len(frame_amp))
e = kde.score_samples(frame.reshape(-1,1))
plt.plot(frame, e)

mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]


plt.show()
