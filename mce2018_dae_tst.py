import numpy as np

from mce_utils import load_ivector, length_norm, make_spkvec, calculate_EER, get_trials_label_with_confusion, calculate_EER_with_confusion

import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras import metrics
from keras import optimizers

def get_DAE(nu=2000):
  iv_dim = 600
  inputs = Input(shape=(iv_dim,))
  x = Dense(nu)(inputs)
  x = Activation('tanh')(x)
  x = Dense(iv_dim)(x)
  out = Activation('linear')(x)
  model = Model(inputs=inputs, outputs=out)
  
  return model

# Making dictionary to find blacklist pair between train and test dataset
bl_match = np.loadtxt('data/bl_matching.csv',dtype='str')
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
    
    
# load test set information
filename = 'data/tst_evaluation_keys.csv'
tst_info = np.loadtxt(filename,dtype='str',delimiter=',',skiprows=1,usecols=range(0,3))
tst_trials = []
tst_trials_label = []
tst_ground_truth =[]
for iter in range(len(tst_info)):
    tst_trials_label.extend([tst_info[iter,0]])
    if tst_info[iter,1]=='background':
        tst_trials = np.append(tst_trials,0)
        
    else:
        tst_trials = np.append(tst_trials,1)
    

# Fix random seed to make results reproducible
seed = 134
np.random.seed(seed)

# Loading i-vector
trn_bl_id, trn_bl_utt, trn_bl_ivector = load_ivector('data/trn_blacklist.csv')
trn_bg_id, trn_bg_utt, trn_bg_ivector = load_ivector('data/trn_background.csv')
dev_bl_id, dev_bl_utt, dev_bl_ivector = load_ivector('data/dev_blacklist.csv')
dev_bg_id, dev_bg_utt, dev_bg_ivector = load_ivector('data/dev_background.csv')
tst_id, test_utt, tst_ivector = load_ivector('data/tst_evaluation.csv')

# length normalization
trn_bl_ivector = length_norm(trn_bl_ivector)
trn_bg_ivector = length_norm(trn_bg_ivector)
dev_bl_ivector = length_norm(dev_bl_ivector)
dev_bg_ivector = length_norm(dev_bg_ivector)
tst_ivector = length_norm(tst_ivector)

# Inputs to DAE are ivectors and targets are the speaker-level mean ivectors
train_spk_ids = pd.DataFrame({'spk_ids': trn_bg_id})
train_ivs = pd.DataFrame(trn_bg_ivector)

X_train = train_ivs.values
Y_train = (train_ivs.groupby(train_spk_ids['spk_ids']).transform('mean')).as_matrix()

# DAE training
model = get_DAE()

model.compile(loss='cosine_proximity',
              optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06),
              metrics=[metrics.mean_squared_error])

num_examples = X_train.shape[0]
num_epochs = 5
batch_size = 512
num_batch_per_epoch = num_examples / batch_size

model.fit(x=X_train,y=Y_train,batch_size=batch_size,epochs=num_epochs)

# Compute DAE-transformed embeddings from ivectors
trn_bl_embeddings = model.predict(trn_bl_ivector,batch_size=batch_size)
trn_bg_embeddings = model.predict(trn_bg_ivector,batch_size=batch_size)
dev_bl_embeddings = model.predict(dev_bl_ivector,batch_size=batch_size)
dev_bg_embeddings = model.predict(dev_bg_ivector,batch_size=batch_size)
tst_embeddings = model.predict(tst_ivector,batch_size=batch_size)

# Calculating speaker mean vector
spk_mean, spk_mean_label = make_spkvec(trn_bl_embeddings,trn_bl_id)

# length normalization
trn_bl_embeddings = length_norm(trn_bl_embeddings)
trn_bg_embeddings = length_norm(trn_bg_embeddings)
dev_bl_embeddings = length_norm(dev_bl_embeddings)
dev_bg_embeddings = length_norm(dev_bg_embeddings)
tst_embeddings = length_norm(tst_embeddings)
        
print('Dev set score using train set :')

# making trials of Dev set
dev_embeddings = np.append(dev_bl_embeddings, dev_bg_embeddings,axis=0)
dev_trials = np.append( np.ones([len(dev_bl_id), 1]), np.zeros([len(dev_bg_id), 1]))

# Cosine distance scoring
scores = spk_mean.dot(dev_embeddings.transpose())
dev_scores = np.max(scores,axis=0)

# Top-S detector EER
dev_EER = calculate_EER(dev_trials, dev_scores)

#divide trial label into target and non-target, plus confusion error(blacklist, fail at blacklist detector)
dev_identified_label = spk_mean_label[np.argmax(scores,axis=0)]
dev_trials_label = np.append( dev_bl_id,dev_bg_id)

# Top-1 detector EER
dev_trials_confusion = get_trials_label_with_confusion(dev_identified_label, dev_trials_label, dev2train, dev_trials )
dev_EER_confusion = calculate_EER_with_confusion(dev_scores,dev_trials_confusion)

print('Test set score using train set:')

#Cosine distance scoring on Test set
scores = spk_mean.dot(tst_embeddings.transpose())
tst_scores = np.max(scores,axis=0)

# top-S detector EER
tst_EER = calculate_EER(tst_trials, tst_scores)

#divide trial label into target and non-target, plus confusion error(blacklist, fail at blacklist detector)
tst_identified_label = spk_mean_label[np.argmax(scores,axis=0)]

# Top-1 detector EER
tst_trials_confusion = get_trials_label_with_confusion(tst_identified_label, tst_trials_label, test2train, tst_trials )
tst_EER_confusion = calculate_EER_with_confusion(tst_scores,tst_trials_confusion)


print('Test set score using train + dev set:')

# get dev set id consistent with Train set
dev_bl_id_along_trnset = []
for iter in range(len(dev_bl_id)):
    dev_bl_id_along_trnset.extend([dev2train[dev_bl_id[iter]]])

# Calculating speaker mean vector
spk_mean, spk_mean_label = make_spkvec(np.append(trn_bl_embeddings,dev_bl_embeddings,0),np.append(trn_bl_id,dev_bl_id_along_trnset))

#Cosine distance scoring on Test set
scores = spk_mean.dot(tst_embeddings.transpose())
tst_scores = np.max(scores,axis=0)

# top-S detector EER
tst_EER = calculate_EER(tst_trials, tst_scores)

#divide trial label into target and non-target, plus confusion error(blacklist, fail at blacklist detector)
tst_identified_label = spk_mean_label[np.argmax(scores,axis=0)]

# Top-1 detector EER
tst_trials_confusion = get_trials_label_with_confusion(tst_identified_label, tst_trials_label, test2train,tst_trials )
tst_EER_confusion = calculate_EER_with_confusion(tst_scores,tst_trials_confusion)
