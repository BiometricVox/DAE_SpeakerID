import numpy as np

from mce_utils import load_ivector, length_norm, make_spkvec, calculate_EER, get_trials_label_with_confusion, calculate_EER_with_confusion

## making dictionary to find blacklist pair between train and test dataset
# bl_match = np.loadtxt('data/bl_matching_dev.csv',dtype='string')
bl_match = np.loadtxt('data/bl_matching.csv',dtype='string')
dev2train={}
dev2id={}
train2dev={}
train2id={}
test2train={}
train2test={}
for iter, line in enumerate(bl_match):
    line_s = line.split(',')
    dev2train[line_s[1].split('_')[-1]]= line_s[3].split('_')[-1]
    dev2id[line_s[1].split('_')[-1]]= line_s[0].split('_')[-1]
    train2dev[line_s[3].split('_')[-1]]= line_s[1].split('_')[-1]
    train2id[line_s[3].split('_')[-1]]= line_s[0].split('_')[-1]
    test2train[line_s[2].split('_')[-1]]= line_s[3].split('_')[-1]
    train2test[line_s[3].split('_')[-1]]= line_s[2].split('_')[-1]
    


# Loading i-vector
trn_bl_id, trn_bl_utt, trn_bl_ivector = load_ivector('data/trn_blacklist.csv')
trn_bg_id, trn_bg_utt, trn_bg_ivector = load_ivector('data/trn_background.csv')
dev_bl_id, dev_bl_utt, dev_bl_ivector = load_ivector('data/dev_blacklist.csv')
dev_bg_id, dev_bg_utt, dev_bg_ivector = load_ivector('data/dev_background.csv')
tst_id, test_utt, tst_ivector = load_ivector('data/tst_evaluation.csv')

# Calculating speaker mean vector
spk_mean, spk_mean_label = make_spkvec(trn_bl_ivector,trn_bl_id)

#length normalization
trn_bl_ivector = length_norm(trn_bl_ivector)
trn_bg_ivector = length_norm(trn_bg_ivector)
dev_bl_ivector = length_norm(dev_bl_ivector)
dev_bg_ivector = length_norm(dev_bg_ivector)
tst_ivector = length_norm(tst_ivector)

# load test set information
filename = 'data/tst_evaluation_keys.csv'
tst_info = np.loadtxt(filename,dtype='string',delimiter=',',skiprows=1,usecols=range(0,3))
tst_trials = []
tst_trials_label = []
tst_ground_truth =[]
for iter in range(len(tst_info)):
    tst_trials_label.extend([tst_info[iter,0]])
    if tst_info[iter,1]=='background':
        tst_trials = np.append(tst_trials,0)
        
    else:
        tst_trials = np.append(tst_trials,1)

        
print '\nDev set score using train set :'
# making trials of Dev set
dev_ivector = np.append(dev_bl_ivector, dev_bg_ivector,axis=0)
dev_trials = np.append( np.ones([len(dev_bl_id), 1]), np.zeros([len(dev_bg_id), 1]))

# Cosine distance scoring
scores = spk_mean.dot(dev_ivector.transpose())

# Multi-target normalization
blscores = spk_mean.dot(trn_bl_ivector.transpose())
mnorm_mu = np.mean(blscores,axis=1)
mnorm_std = np.std(blscores,axis=1)
for iter in range(np.shape(scores)[1]):
    scores[:,iter]= (scores[:,iter] - mnorm_mu) / mnorm_std
dev_scores = np.max(scores,axis=0)

# Top-S detector EER
dev_EER = calculate_EER(dev_trials, dev_scores)

#divide trial label into target and non-target, plus confusion error(blacklist, fail at blacklist detector)
dev_identified_label = spk_mean_label[np.argmax(scores,axis=0)]
dev_trials_label = np.append( dev_bl_id,dev_bg_id)

# Top-1 detector EER
dev_trials_confusion = get_trials_label_with_confusion(dev_identified_label, dev_trials_label, dev2train, dev_trials )
dev_EER_confusion = calculate_EER_with_confusion(dev_scores,dev_trials_confusion)

print '\nTest set score using train set:'
#Cosine distance scoring on Test set
scores = spk_mean.dot(tst_ivector.transpose())

# Multi-target normalization
blscores = spk_mean.dot(trn_bl_ivector.transpose())
mnorm_mu = np.mean(blscores,axis=1)
mnorm_std = np.std(blscores,axis=1)
for iter in range(np.shape(scores)[1]):
    scores[:,iter]= (scores[:,iter] - mnorm_mu) / mnorm_std
tst_scores = np.max(scores,axis=0)

# top-S detector EER
tst_EER = calculate_EER(tst_trials, tst_scores)

#divide trial label into target and non-target, plus confusion error(blacklist, fail at blacklist detector)
tst_identified_label = spk_mean_label[np.argmax(scores,axis=0)]

# Top-1 detector EER
tst_trials_confusion = get_trials_label_with_confusion(tst_identified_label, tst_trials_label, test2train, tst_trials )
tst_EER_confusion = calculate_EER_with_confusion(tst_scores,tst_trials_confusion)


print '\nTest set score using train + dev set:'
# get dev set id consistent with Train set
dev_bl_id_along_trnset = []
for iter in range(len(dev_bl_id)):
    dev_bl_id_along_trnset.extend([dev2train[dev_bl_id[iter]]])

# Calculating speaker mean vector
spk_mean, spk_mean_label = make_spkvec(np.append(trn_bl_ivector,dev_bl_ivector,0),np.append(trn_bl_id,dev_bl_id_along_trnset))

#Cosine distance scoring on Test set
scores = spk_mean.dot(tst_ivector.transpose())
# tst_scores = np.max(scores,axis=0)


# Multi-target normalization
blscores = spk_mean.dot(np.append(trn_bl_ivector.transpose(),dev_bl_ivector.transpose(),axis=1))
mnorm_mu = np.mean(blscores,axis=1)
mnorm_std = np.std(blscores,axis=1)
for iter in range(np.shape(scores)[1]):
    scores[:,iter]= (scores[:,iter] - mnorm_mu) / mnorm_std
tst_scores = np.max(scores,axis=0)

# top-S detector EER
tst_EER = calculate_EER(tst_trials, tst_scores)

#divide trial label into target and non-target, plus confusion error(blacklist, fail at blacklist detector)
tst_identified_label = spk_mean_label[np.argmax(scores,axis=0)]

# Top-1 detector EER
tst_trials_confusion = get_trials_label_with_confusion(tst_identified_label, tst_trials_label, test2train,tst_trials )
tst_EER_confusion = calculate_EER_with_confusion(tst_scores,tst_trials_confusion)
