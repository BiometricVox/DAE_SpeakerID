"""
Utility functions adapted from: https://github.com/swshon/multi-speakerID.

 Copyright 2018 Roberto Font
				Biometric Vox S.L.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function
import numpy as np
from sklearn.metrics import roc_curve

def load_ivector(filename):
    utt = np.loadtxt(filename,dtype='str',delimiter=',',skiprows=1,usecols=[0])
    ivector = np.loadtxt(filename,dtype='float32',delimiter=',',skiprows=1,usecols=range(1,601))
    spk_id = []
    for iter in range(len(utt)):
        spk_id = np.append(spk_id,utt[iter].split('_')[0])

    return spk_id, utt, ivector

def length_norm(mat):
# length normalization (l2 norm)
# input: mat = [utterances X vector dimension] ex) (float) 8631 X 600

    norm_mat = []
    for line in mat:
        temp = line/np.math.sqrt(sum(np.power(line,2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat

def make_spkvec(mat, spk_label):
# calculating speaker mean vector
# input: mat = [utterances X vector dimension] ex) (float) 8631 X 600
#        spk_label = string vector ex) ['abce','cdgd']

#     for iter in range(len(spk_label)):
#         spk_label[iter] = spk_label[iter].split('_')[0]

    spk_label, spk_index  = np.unique(spk_label,return_inverse=True)
    spk_mean=[]
    mat = np.array(mat)

    # calculating speaker mean i-vector
    for i, spk in enumerate(spk_label):
        spk_mean.append(np.mean(mat[np.nonzero(spk_index==i)],axis=0))
    spk_mean = length_norm(spk_mean)
    return spk_mean, spk_label

def calculate_EER(trials, scores):
# calculating EER of Top-S detector
# input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background)
#        scores = float vector

    # Calculating EER
    fpr,tpr,threshold = roc_curve(trials,scores,pos_label=1)
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    
    print("Top S detector EER is %0.2f%%"% (EER*100))
    return EER

def get_trials_label_with_confusion(identified_label, groundtruth_label,dict4spk,is_trial ):
# determine if the test utterance would make confusion error
# input: identified_label = string vector, identified result of test utterance among multi-target from the detection system 
#        groundtruth_label = string vector, ground truth speaker labels of test utterances
#        dict4spk = dictionary, convert label to target set, ex) train2dev convert train id to dev id

    trials = np.zeros(len(identified_label))
    for iter in range(0,len(groundtruth_label)):
        enroll = identified_label[iter].split('_')[0]
        test = groundtruth_label[iter].split('_')[0]
        if is_trial[iter]:
            if enroll == dict4spk[test]:
                trials[iter]=1 # for Target trial (blacklist speaker)
            else:
                trials[iter]=-1 # for Target trial (backlist speaker), but fail on blacklist classifier
                
        else :
            trials[iter]=0 # for non-target (non-blacklist speaker)
    return trials


def calculate_EER_with_confusion(scores,trials):
# calculating EER of Top-1 detector
# input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background) -1: confusion(blacklist)
#        scores = float vector

    # exclude confusion error (trials==-1)
    scores_wo_confusion = scores[np.nonzero(trials!=-1)[0]]
    trials_wo_confusion = trials[np.nonzero(trials!=-1)[0]]

    # dev_trials contain labels of target. (target=1, non-target=0)
    fpr,tpr,threshold = roc_curve(trials_wo_confusion,scores_wo_confusion,pos_label=1, drop_intermediate=False)
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # EER withouth confusion error
    EER = fpr[np.argmin(np.absolute((fnr-fpr)))]
    
    # Add confusion error to false negative rate(Miss rate)
    total_negative = len(np.nonzero(np.array(trials_wo_confusion)==0)[0])
    total_positive = len(np.nonzero(np.array(trials_wo_confusion)==1)[0])
    fp= fpr*np.float(total_negative)  
    fn= fnr*np.float(total_positive) 
    fn += len(np.nonzero(trials==-1)[0])
    total_positive += len(np.nonzero(trials==-1)[0])
    fpr= fp/total_negative
    fnr= fn/total_positive

    # EER with confusion Error
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    
    print("Top 1 detector EER is %0.2f%% (Total confusion error is %d)"% ((EER*100), len(np.nonzero(trials==-1)[0])))
    return EER
